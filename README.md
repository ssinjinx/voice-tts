# Voice TTS — Paul Harvey News Pipeline

Takes a news article, rewrites it in Paul Harvey's storytelling style via a local LLM, then synthesizes the audio using Qwen3-TTS with voice cloning from a real Paul Harvey audio clip.

## How It Works

### Step 1 — LLM Rewrite (Ollama)

The article is sent to a local Ollama model (`minimax-m2.5:cloud` by default) with a system prompt that rewrites it in Paul Harvey's style:

- Opens with a dramatic hook
- Builds suspense, withholds the key reveal
- Uses "Page 2..." as a mid-story transition
- Closes with "And now you know... the rest of the story. Good day!"
- Inserts `[pause]` markers for dramatic beats

The `[pause]` markers are stripped and replaced with commas before TTS synthesis.

### Step 2 — TTS Synthesis (Qwen3-TTS)

Two modes are available:

**Voice Clone (default)** — Uses `Qwen3-TTS-12Hz-1.7B-Base` with a reference audio clip (`~/Music/harveyclip.wav`) to clone Paul Harvey's voice. This produces significantly better results than voice design.

**Voice Design** — Uses `Qwen3-TTS-12Hz-1.7B-VoiceDesign` with a text description of the desired voice (warm, gravelly baritone, mid-century radio cadence). No audio sample needed, but output doesn't sound like Harvey. Use `--no-clone` to select this mode.

### Pipeline Diagram

```
Article (.txt or stdin)
    → Ollama LLM (minimax-m2.5:cloud)
    → Paul Harvey-style script with [pause] markers
    → Strip [pause] markers → clean text
    → Qwen3-TTS (voice clone from harveyclip.wav)
    → Output audio (.wav)
```

## Requirements

- **OS:** Linux (tested on Ubuntu with kernel 6.17)
- **GPU:** AMD Radeon RX 7900 XTX (24GB VRAM) with ROCm
- **ROCm:** 7.2.0 installed at `/opt/rocm-7.2.0`
- **Python:** 3.12 via conda (`voice-tts` env)
- **Ollama:** Running locally on port 11434 with a model pulled (e.g. `minimax-m2.5:cloud`)
- **Reference audio:** A Paul Harvey `.wav` clip at `~/Music/harveyclip.wav` (13 seconds, used for voice cloning)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/ssinjinx/voice-tts.git
cd voice-tts

# 2. Run setup (creates conda env, installs ROCm PyTorch + qwen-tts)
bash setup.sh

# 3. Make sure Ollama is running with a model available
ollama pull minimax-m2.5:cloud
ollama serve  # if not already running

# 4. Place a Paul Harvey audio clip for voice cloning
# Default path: ~/Music/harveyclip.wav
# Any clean .wav clip of Paul Harvey speaking (10-15 seconds works well)
```

### What setup.sh does

1. Creates a conda env named `voice-tts` with Python 3.12
2. Installs PyTorch with **ROCm 6.3** wheels (backward-compatible with ROCm 7.2.0) — this must be installed before `qwen-tts` so PyTorch uses the GPU build, not CPU/CUDA
3. Installs `qwen-tts` and `soundfile`

### Important: ROCm + PyTorch

The AMD 7900 XTX requires ROCm PyTorch wheels. The setup script installs `torch` from `https://download.pytorch.org/whl/rocm6.3`. If you skip this step or install regular PyTorch, the model will fall back to CPU (extremely slow).

Verify GPU is detected:
```bash
conda run -n voice-tts python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True AMD Radeon RX 7900 XTX
```

## Usage

```bash
conda activate voice-tts

# Basic usage — reads article, rewrites as Paul Harvey, clones his voice
python article_to_harvey.py article.txt output.wav

# Read from stdin
echo "Your article text here..." | python article_to_harvey.py - output.wav

# Print the LLM-generated script before synthesizing
python article_to_harvey.py article.txt output.wav --print-script

# Use a different Ollama model for the rewrite
python article_to_harvey.py article.txt output.wav --model qwen3:4b

# Use a different reference audio clip
python article_to_harvey.py article.txt output.wav --clone-audio /path/to/clip.wav

# Use voice design instead of voice cloning (generic voice, no reference audio)
python article_to_harvey.py article.txt output.wav --no-clone

# Use the smaller 0.6B TTS model (less VRAM, lower quality)
python article_to_harvey.py article.txt output.wav --tts-size 0.6B
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `article` | *(required)* | Path to article text file, or `-` for stdin |
| `output` | *(required)* | Output audio file path (.wav) |
| `--clone-audio PATH` | `~/Music/harveyclip.wav` | Reference audio clip for voice cloning |
| `--no-clone` | off | Use voice design mode instead of cloning |
| `--model NAME` | `minimax-m2.5:cloud` | Ollama model for the Paul Harvey rewrite |
| `--tts-size` | `1.7B` | Qwen3-TTS model size (`1.7B` or `0.6B`) |
| `--print-script` | off | Print the rewritten script before synthesis |

## Output

- Format: 16-bit PCM WAV, mono, 24kHz sample rate
- A ~500 word article produces roughly 2 minutes of audio (~6MB)
- First run downloads the Qwen3-TTS model (~3GB), subsequent runs use the cached model

## Troubleshooting

**MIOpen workspace warnings** — Harmless ROCm noise during inference. Does not affect output quality.

**flash-attn warning** — Flash attention is not available for ROCm. The model falls back to standard PyTorch attention automatically. No action needed.

**Model loading to RAM instead of VRAM** — If `torch.cuda.is_available()` returns `False`, you have the wrong PyTorch build. Reinstall with ROCm wheels:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

**Ollama connection refused** — Make sure Ollama is running: `ollama serve`

## Tech Stack

| Component | Tool |
|-----------|------|
| TTS Model | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B (Base for clone, VoiceDesign for text-described voice) |
| LLM | [Ollama](https://ollama.com) running locally (default: minimax-m2.5:cloud) |
| GPU | AMD Radeon RX 7900 XTX via ROCm 7.2.0 |
| Python | 3.12 in conda `voice-tts` env |
| Audio | `soundfile` for WAV output |
