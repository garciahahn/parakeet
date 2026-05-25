#!/usr/bin/env python3
"""
Voice-to-Text — Push-to-talk local transcription using NVIDIA Parakeet v3.

Usage:
    python transcribe.py                              Push-to-talk (Parakeet, auto-detect language)
    python transcribe.py --model canary --lang es      Push-to-talk with forced Spanish
    python transcribe.py --model nb-whisper            Push-to-talk Norwegian (bokmål/nynorsk)
    python transcribe.py file audio.wav                Transcribe a file → audio.txt
    python transcribe.py file audio.wav -o out.md      → markdown output
    python transcribe.py file audio.wav --model canary --lang fr   Force French
    python transcribe.py file audio.wav --model nb-whisper         Norwegian transcription

Models:
    parakeet   (default) nvidia/parakeet-tdt-0.6b-v3    Auto-detects language (25 EU languages)
    canary               nvidia/canary-1b-v2            Accepts --lang hint for explicit language
    nb-whisper           NbAiLab/nb-whisper-large       Norwegian (bokmål + nynorsk); --lang no|nb|nn

Hotkeys (push-to-talk mode):
    Hold  Ctrl+`             Record voice
    Press Ctrl+Shift+`       Cycle output mode (clipboard → auto-type → both)
    Press Ctrl+Shift+Q       Quit

Requires:
    - Windows 10+ (Linux/macOS need root for global hotkeys)
    - Python 3.10–3.12
    - NVIDIA GPU recommended (CPU works but is much slower)

First run downloads the model (cached after that).
"""

import argparse
import os
import sys
import time
import tempfile
import threading
from datetime import datetime
from pathlib import Path

# Use the OS certificate store (fixes corporate proxy / self-signed CA issues)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


import numpy as np


def _ensure_ffmpeg_on_path() -> str:
    """Return a path to an `ffmpeg` binary, and make sure it's usable.

    Prefers a system `ffmpeg` if present. Otherwise falls back to the binary
    shipped by `imageio-ffmpeg` (a pip package — no system install required).
    """
    from shutil import which
    sys_ffmpeg = which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg is required to decode compressed audio but was not found. "
            "Install it system-wide, or `pip install imageio-ffmpeg`."
        ) from e


def _decode_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    """Decode any audio/video file to a mono float32 numpy array at `target_sr`.

    Uses ffmpeg under the hood (system binary if available, otherwise the one
    shipped by `imageio-ffmpeg`). Returns the raw PCM samples in [-1, 1].
    """
    import subprocess
    ffmpeg = _ensure_ffmpeg_on_path()
    cmd = [
        ffmpeg, "-nostdin", "-loglevel", "error",
        "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(target_sr),
        "-",
    ]
    # CREATE_NO_WINDOW on Windows so no console pops up from the bundled binary.
    creationflags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.run(cmd, capture_output=True, check=True, creationflags=creationflags)
    return np.frombuffer(proc.stdout, dtype=np.float32)


import sounddevice as sd
import soundfile as sf
import pyperclip
import keyboard
import torch


# ─── Configuration ──────────────────────────────────────────────────────────────

MODELS = {
    # NeMo models — 25 EU languages, no Norwegian.
    "parakeet":   {"backend": "nemo",    "name": "nvidia/parakeet-tdt-0.6b-v3"},
    "canary":     {"backend": "nemo",    "name": "nvidia/canary-1b-v2"},
    # Whisper fine-tuned by the National Library of Norway — best for Norwegian.
    "nb-whisper": {"backend": "whisper", "name": "NbAiLab/nb-whisper-large"},
}
DEFAULT_MODEL = "parakeet"

# Models that accept a --lang hint.
LANG_AWARE_MODELS = {"canary", "nb-whisper"}

SAMPLE_RATE = 16_000                          # 16 kHz mono, required by model
CHANNELS = 1

PUSH_TO_TALK_KEY = "`"            # hold Ctrl + this key to record
TOGGLE_MODE_HOTKEY = "ctrl+shift+`"
QUIT_HOTKEY = "ctrl+shift+q"

OUTPUT_MODES = ("clipboard", "auto-type", "both")

# ────────────────────────────────────────────────────────────────────────────────

_recording = False
_audio_frames: list[np.ndarray] = []
_stream: sd.InputStream | None = None
_mode_idx = 0
_lock = threading.Lock()
model = None
_active_model_key: str | None = None
_active_lang: str | None = None


# ─── Model ──────────────────────────────────────────────────────────────────────

def _active_backend() -> str:
    return MODELS[_active_model_key]["backend"] if _active_model_key else ""


def load_model(model_key: str = DEFAULT_MODEL):
    """Download (first run) and load the ASR model into GPU/CPU."""
    global model, _active_model_key
    if model is not None and _active_model_key == model_key:
        return
    spec = MODELS[model_key]
    backend = spec["backend"]
    model_name = spec["name"]
    print(f"  ⏳ Loading {model_key} model (first run downloads, cached after that)…")

    if backend == "nemo":
        import nemo.collections.asr as nemo_asr
        # NeMo caches models under ~/.cache/torch/NeMo after the first download.
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    elif backend == "whisper":
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            return_timestamps=False,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    _active_model_key = model_key
    print("  ✅ Model ready.\n")


# Formats that libsndfile / NeMo can read natively without ffmpeg.
_NEMO_NATIVE_FORMATS = {".wav", ".flac", ".ogg"}


def _ensure_nemo_compatible(audio_paths: list[str]) -> tuple[list[str], list[str]]:
    """Convert non-native audio files to temp WAVs so NeMo never invokes torchaudio/ffmpeg.

    Returns (paths_for_nemo, temp_files_to_cleanup).
    """
    out_paths: list[str] = []
    tmp_files: list[str] = []
    for p in audio_paths:
        if Path(p).suffix.lower() in _NEMO_NATIVE_FORMATS:
            out_paths.append(p)
        else:
            pcm = _decode_audio(p, target_sr=SAMPLE_RATE)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, pcm, SAMPLE_RATE)
            tmp.close()
            out_paths.append(tmp.name)
            tmp_files.append(tmp.name)
    return out_paths, tmp_files


def run_transcription(audio_paths: list[str], lang: str | None = None) -> list[str]:
    """Run inference on audio files. Backend-aware."""
    backend = _active_backend()

    if backend == "whisper":
        # NB-Whisper expects a Whisper language code. Map nb/nn → no (Norwegian).
        whisper_lang = lang or "no"
        if whisper_lang in ("nb", "nn"):
            whisper_lang = "no"
        generate_kwargs = {"language": whisper_lang, "task": "transcribe"}
        # Decode audio ourselves (ffmpeg via imageio-ffmpeg fallback) so the
        # pipeline gets a raw float32 array at 16 kHz and never shells out to
        # a bare `ffmpeg` command (which fails on systems without ffmpeg on PATH).
        texts = []
        for p in audio_paths:
            audio = _decode_audio(p, target_sr=16_000)
            out = model(
                {"array": audio, "sampling_rate": 16_000},
                generate_kwargs=generate_kwargs,
            )
            texts.append(out["text"].strip() if isinstance(out, dict) else str(out))
        return texts

    # NeMo backend (parakeet / canary) — pre-convert non-native formats to WAV
    # so NeMo/lhotse never tries to use torchaudio's ffmpeg bindings.
    safe_paths, tmp_files = _ensure_nemo_compatible(audio_paths)
    try:
        if _active_model_key == "canary" and lang:
            override_cfg = {"source_lang": lang, "target_lang": lang}
            results = model.transcribe(safe_paths, override_config=override_cfg)
        else:
            results = model.transcribe(safe_paths)
        return [r.text if hasattr(r, "text") else str(r) for r in results]
    finally:
        for f in tmp_files:
            os.unlink(f)


# ─── VAD chunking (silero-vad) ──────────────────────────────────────────────────

_vad_model = None
_get_speech_timestamps = None


def load_vad_model():
    """Load Silero VAD from torch hub (cached after first download)."""
    global _vad_model, _get_speech_timestamps
    if _vad_model is not None:
        return
    print("  ⏳ Loading Silero VAD model…")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True,
    )
    _vad_model = vad_model
    _get_speech_timestamps = utils[0]  # (get_speech_timestamps, save_audio, read_audio, ...)
    print("  ✅ VAD ready.")


def chunk_audio(
    audio_path: str,
    min_silence_ms: int = 600,
    padding_ms: int = 200,
) -> list[str]:
    """Split an audio file at silent gaps using Silero VAD.

    Returns a list of temp WAV file paths, one per speech chunk.
    Caller is responsible for cleaning them up.
    """
    load_vad_model()

    ext = Path(audio_path).suffix.lower()
    if ext in _NEMO_NATIVE_FORMATS:
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # Resample to 16 kHz if needed
        if sr != SAMPLE_RATE:
            import torchaudio
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav_t)
            wav = wav_t.squeeze().numpy()
    else:
        # Decode compressed formats (m4a, mp3, etc.) via ffmpeg
        wav = _decode_audio(audio_path, target_sr=SAMPLE_RATE)

    wav_tensor = torch.from_numpy(wav)

    speech_ts = _get_speech_timestamps(
        wav_tensor, _vad_model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
    )

    # If VAD returned nothing (e.g. very quiet file), fall back to whole file
    if not speech_ts:
        return [audio_path]

    # Merge segments that are close together to avoid splitting mid-sentence
    merge_gap = int(SAMPLE_RATE * padding_ms / 1000)
    merged: list[dict] = [speech_ts[0]]
    for seg in speech_ts[1:]:
        if seg["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    # Pad each segment slightly so words at edges aren't clipped
    pad_samples = int(SAMPLE_RATE * padding_ms / 1000)
    total_samples = len(wav)
    chunk_paths: list[str] = []
    for i, seg in enumerate(merged):
        start = max(0, seg["start"] - pad_samples)
        end = min(total_samples, seg["end"] + pad_samples)
        chunk = wav[start:end]
        tmp = tempfile.NamedTemporaryFile(suffix=f"_chunk{i}.wav", delete=False)
        sf.write(tmp.name, chunk, SAMPLE_RATE)
        tmp.close()
        chunk_paths.append(tmp.name)

    return chunk_paths


# ─── Output mode ────────────────────────────────────────────────────────────────

def current_mode() -> str:
    return OUTPUT_MODES[_mode_idx]


def toggle_mode():
    global _mode_idx
    _mode_idx = (_mode_idx + 1) % len(OUTPUT_MODES)
    print(f"  🔄 Output mode → {current_mode()}")


# ─── Recording ──────────────────────────────────────────────────────────────────

def _audio_callback(indata, frames, time_info, status):
    """sounddevice callback — appends audio chunks while recording."""
    if _recording:
        _audio_frames.append(indata.copy())


def start_recording():
    global _recording, _stream, _audio_frames
    with _lock:
        if _recording:
            return
        _recording = True
        _audio_frames = []
    print("  🎙️  Recording… (release to stop)", flush=True)
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=_audio_callback,
    )
    _stream.start()


# ─── Transcription ──────────────────────────────────────────────────────────────

def stop_and_transcribe():
    global _recording, _stream
    with _lock:
        if not _recording:
            return
        _recording = False
    if _stream:
        _stream.stop()
        _stream.close()
        _stream = None

    if not _audio_frames:
        print("  ⚠️  No audio captured.")
        return

    audio = np.concatenate(_audio_frames, axis=0)

    # Save to a temporary WAV so NeMo can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        print("  ⏳ Transcribing…", end=" ", flush=True)
        texts = run_transcription([tmp_path], lang=_active_lang)
        text = texts[0]
        print("done.")

        if not text.strip():
            print("  (empty transcription)")
            return

        print(f'  📝 "{text}"')
        deliver(text)
    finally:
        os.unlink(tmp_path)


# ─── Text delivery ──────────────────────────────────────────────────────────────

def deliver(text: str):
    """Send transcribed text to clipboard, active window, or both."""
    mode = current_mode()

    if mode in ("clipboard", "both"):
        pyperclip.copy(text)
        print("  📋 Copied to clipboard.")

    if mode in ("auto-type", "both"):
        # Paste via Ctrl+V for full Unicode support in any app.
        # In "auto-type" mode we restore the original clipboard afterwards.
        old_clip = ""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            pass

        if mode == "auto-type":
            pyperclip.copy(text)

        time.sleep(0.15)
        keyboard.send("ctrl+v")

        if mode == "auto-type":
            time.sleep(0.15)
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

        print("  ⌨️  Pasted into active window.")


# ─── File transcription ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4"}


def transcribe_file(
    audio_path: str,
    output_path: str,
    fmt: str,
    lang: str | None = None,
    chunk: bool = False,
    min_silence_ms: int = 600,
):
    """Transcribe an audio file and write the result to a txt or md file."""
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        print(f"  ❌ File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    ext = Path(audio_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(
            f"  ❌ Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if chunk:
        print(f"  🔪 Splitting {os.path.basename(audio_path)} at silent gaps…")
        chunk_paths = chunk_audio(audio_path, min_silence_ms=min_silence_ms)
        print(f"  ✂️  {len(chunk_paths)} chunk(s) detected.")

        all_texts: list[str] = []
        tmp_paths = [p for p in chunk_paths if p != audio_path]
        try:
            for i, cp in enumerate(chunk_paths, 1):
                print(f"  ⏳ Transcribing chunk {i}/{len(chunk_paths)}…", end=" ", flush=True)
                texts = run_transcription([cp], lang=lang)
                print("done.")
                if texts[0].strip():
                    all_texts.append(texts[0].strip())
        finally:
            for p in tmp_paths:
                os.unlink(p)

        text = "\n\n".join(all_texts)
    else:
        print(f"  ⏳ Transcribing {os.path.basename(audio_path)}…", end=" ", flush=True)
        texts = run_transcription([audio_path], lang=lang)
        text = texts[0]
        print("done.")

    if not text.strip():
        print("  ⚠️  Transcription was empty — no output file written.")
        return

    if fmt == "md":
        content = (
            f"# Transcript\n\n"
            f"- **Source:** `{os.path.basename(audio_path)}`\n"
            f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"---\n\n"
            f"{text}\n"
        )
    else:
        content = text + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f'  📝 "{text[:80]}{"…" if len(text) > 80 else ""}"')
    print(f"  💾 Saved to {output_path}")


# ─── Hotkey handler ─────────────────────────────────────────────────────────────

def on_key_event(e):
    """Global key hook: Ctrl+` push-to-talk."""
    if e.event_type == keyboard.KEY_DOWN and e.name == PUSH_TO_TALK_KEY:
        # Only trigger when Ctrl is held WITHOUT Shift (avoid collision with toggle)
        if keyboard.is_pressed("ctrl") and not keyboard.is_pressed("shift"):
            start_recording()
    elif e.event_type == keyboard.KEY_UP and e.name == PUSH_TO_TALK_KEY:
        if _recording:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()


# ─── Main ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🦜 Parakeet — local speech-to-text with NVIDIA Parakeet v3",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help="ASR model to use (default: parakeet). Use 'canary' with --lang for explicit "
             "European language, or 'nb-whisper' for Norwegian (bokmål/nynorsk).",
    )
    parser.add_argument(
        "--lang",
        default=None,
        metavar="CODE",
        help="Language code (e.g. en, es, fr for canary; no, nb, nn for nb-whisper). "
             "Only used with --model canary or --model nb-whisper.",
    )

    sub = parser.add_subparsers(dest="command")

    file_cmd = sub.add_parser(
        "file",
        help="Transcribe an audio file to txt or markdown",
    )
    file_cmd.add_argument(
        "audio",
        help="Path to an audio file (.wav, .mp3, .flac, .ogg, .m4a, .webm, .mp4)",
    )
    file_cmd.add_argument(
        "-o", "--output",
        help="Output file path (default: <audio_name>.txt)",
    )
    file_cmd.add_argument(
        "--format",
        choices=["txt", "md"],
        default=None,
        help="Output format; inferred from --output extension when given (default: txt)",
    )
    file_cmd.add_argument(
        "--chunk",
        action="store_true",
        help="Split audio at silent gaps (via Silero VAD) before transcribing. "
             "Recommended for files longer than ~5 minutes.",
    )
    file_cmd.add_argument(
        "--min-silence",
        type=int,
        default=600,
        metavar="MS",
        help="Minimum silence duration in ms to split on (default: 600). "
             "Lower = more splits, higher = fewer.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_key = args.model
    lang = args.lang

    if lang and model_key not in LANG_AWARE_MODELS:
        # Norwegian codes → switch to nb-whisper; otherwise → canary.
        if lang.lower() in ("no", "nb", "nn", "nob", "nno"):
            print("  ⚠️  --lang is Norwegian. Switching to --model nb-whisper.")
            model_key = "nb-whisper"
        else:
            print("  ⚠️  --lang is only supported with --model canary or nb-whisper. "
                  "Switching to canary.")
            model_key = "canary"

    if args.command == "file":
        # Determine output format and path
        fmt = args.format
        if args.output:
            out_path = args.output
            if fmt is None:
                fmt = "md" if out_path.endswith(".md") else "txt"
        else:
            stem = Path(args.audio).stem
            fmt = fmt or "txt"
            out_path = f"{stem}.{fmt}"

        print("=" * 56)
        print("    🦜 Parakeet — File Transcription")
        print("=" * 56)
        load_model(model_key)
        transcribe_file(
            args.audio, out_path, fmt,
            lang=lang,
            chunk=args.chunk,
            min_silence_ms=args.min_silence,
        )
        return

    # Default: push-to-talk mode
    global _active_lang
    _active_lang = lang

    print("=" * 56)
    print("    🦜 Voice-to-Text  (NVIDIA Parakeet v3)")
    print("=" * 56)

    load_model(model_key)

    print(f"  Model: {model_key}" + (f"  |  Language: {lang}" if lang else ""))
    print(f"  Hotkeys:")
    print(f"    Hold  Ctrl+{PUSH_TO_TALK_KEY}             →  Record")
    print(f"    Press {TOGGLE_MODE_HOTKEY}   →  Toggle output mode")
    print(f"    Press {QUIT_HOTKEY}   →  Quit")
    print()
    print(f"  Output mode: {current_mode()}")
    print("─" * 56)
    print("  Listening…\n")

    keyboard.hook(on_key_event)
    keyboard.add_hotkey(TOGGLE_MODE_HOTKEY, toggle_mode, suppress=True)
    keyboard.add_hotkey(QUIT_HOTKEY, lambda: os._exit(0), suppress=True)

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        pass

    print("\n  👋 Goodbye!")


if __name__ == "__main__":
    main()
