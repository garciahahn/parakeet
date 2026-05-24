# 🦜 Parakeet

Local speech-to-text powered by [NVIDIA Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). Runs entirely on your machine — no API keys, no cloud, no data leaves your device.

Two modes of operation:

- **Push-to-talk** — hold a hotkey, speak, release to transcribe and paste.
- **File transcription** — point it at an audio file, get a `.txt` or `.md` transcript.

Two models available:

| Model | Flag | Size | Language |
|---|---|---|---|
| **Parakeet** (default) | `--model parakeet` | ~2.5 GB | Auto-detects (25 EU languages) |
| **Canary** | `--model canary` | ~5 GB | Explicit `--lang` hint for best accuracy |
| **NB-Whisper** | `--model nb-whisper` | ~3 GB | Norwegian (bokmål + nynorsk) — by the National Library of Norway |

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10 – 3.12 |
| OS | Windows 10+ recommended. Linux/macOS require root for global hotkeys (push-to-talk mode). |
| GPU | NVIDIA GPU strongly recommended. CPU works but is much slower. |
| Disk | ~2.5 GB per model (downloaded automatically on first run, cached permanently in `~/.cache/torch/NeMo/`) |

## Installation

### 1. Clone the repo

```bash
git clone <repo-url> parakeet
cd parakeet
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch + torchaudio

Install PyTorch and torchaudio **before** the other dependencies. Pick the command that matches your setup:

```bash
# CUDA 12.1 (recommended if you have an NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU-only (slow, but works without a GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. (Linux) Install PortAudio

`sounddevice` requires the PortAudio system library:

```bash
# Debian / Ubuntu
sudo apt install libportaudio2

# Fedora
sudo dnf install portaudio
```

## Usage

### Push-to-talk mode (default)

```bash
python transcribe.py
```

The first run downloads the model (~2.5 GB, cached permanently after that). After that, the app listens for hotkeys:

| Hotkey | Action |
|---|---|
| Hold `Ctrl + `` ` | Record voice |
| Press `Ctrl + Shift + `` ` | Cycle output mode (clipboard → auto-type → both) |
| Press `Ctrl + Shift + Q` | Quit |

**Output modes:**

- **clipboard** — copies the transcription to your clipboard.
- **auto-type** — pastes the transcription into the active window via `Ctrl+V`.
- **both** — copies to clipboard *and* pastes into the active window.

### File transcription

Transcribe an audio file and save the result as plain text or markdown.

```bash
# Basic — outputs recording.txt
python transcribe.py file recording.wav

# Custom output path
python transcribe.py file recording.wav -o transcript.txt

# Markdown output (inferred from .md extension)
python transcribe.py file recording.wav -o transcript.md

# Explicit format flag
python transcribe.py file recording.wav --format md
```

**Supported audio formats:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.webm`, `.mp4`

**Markdown output** includes a header with the source filename and date:

```markdown
# Transcript

- **Source:** `recording.wav`
- **Date:** 2026-04-17 10:30

---

Hello, this is the transcribed text…
```

### Chunking long audio files

For long recordings, use `--chunk` to automatically split the audio at silent gaps using [Silero VAD](https://github.com/snakers4/silero-vad) before transcribing. Each chunk is transcribed independently and the results are joined.

```bash
# Split at silences and transcribe each chunk
python transcribe.py file long_meeting.wav --chunk

# Require longer silences before splitting (default: 600 ms)
python transcribe.py file podcast.mp3 --chunk --min-silence 1000

# Combine with other options
python transcribe.py --model canary --lang en file interview.wav --chunk -o interview.md
```

The `--min-silence` flag controls how long a quiet gap must be (in milliseconds) before the audio is split there. Lower values produce more, shorter chunks; higher values keep more audio together.

### Choosing a model & language

By default, **Parakeet** is used and auto-detects the spoken language. If auto-detection isn't working well for your audio, switch to **Canary** and specify the language explicitly:

```bash
# Push-to-talk with forced Spanish
python transcribe.py --model canary --lang es

# File transcription with forced French
python transcribe.py --model canary --lang fr file interview.mp3 -o interview.md
```

If you pass `--lang` without `--model canary`, the tool auto-switches to Canary for you.

### Norwegian transcription

Parakeet and Canary do **not** support Norwegian. Use **NB-Whisper** — a Whisper model fine-tuned by the National Library of Norway on 66 000+ hours of Norwegian audio:

```bash
# Push-to-talk Norwegian
python transcribe.py --model nb-whisper

# File transcription (bokmål or nynorsk — both are handled by language code "no")
python transcribe.py --model nb-whisper file intervju.mp3 -o intervju.md

# Shortcut: passing --lang no|nb|nn auto-selects nb-whisper
python transcribe.py --lang no file opptak.wav
```

<details>
<summary>Supported language codes</summary>

`bg` Bulgarian, `cs` Czech, `da` Danish, `de` German, `el` Greek,
`en` English, `es` Spanish, `et` Estonian, `fi` Finnish, `fr` French,
`hr` Croatian, `hu` Hungarian, `it` Italian, `lt` Lithuanian, `lv` Latvian,
`mt` Maltese, `nl` Dutch, `pl` Polish, `pt` Portuguese, `ro` Romanian,
`ru` Russian, `sk` Slovak, `sl` Slovenian, `sv` Swedish, `uk` Ukrainian

</details>

### Help

```bash
python transcribe.py --help
python transcribe.py file --help
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch first — see step 3 above. |
| Model download fails behind a proxy | `pip install truststore` — it's already in `requirements.txt` and auto-patches SSL. |
| `sounddevice` import error on Linux | Install PortAudio: `sudo apt install libportaudio2` |
| Hotkeys don't work on Linux/macOS | The `keyboard` library requires root: `sudo python transcribe.py` |
| Transcription is slow | Make sure PyTorch is using your GPU: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`. |
| Wrong language detected | Use `--model canary --lang <code>` to force a specific language. |

## License

See [LICENSE](LICENSE) if present, otherwise contact the repository owner.
