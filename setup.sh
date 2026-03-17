#!/usr/bin/env bash
# Set up the voice-tts conda environment

set -e

conda create -n voice-tts python=3.12 -y
conda activate voice-tts

pip install -U anthropic qwen-tts

# Optional but recommended: reduces VRAM usage
# Requires compatible GPU (CUDA) and torch already installed
# MAX_JOBS=4 pip install -U flash-attn --no-build-isolation

echo ""
echo "Setup complete. Activate with: conda activate voice-tts"
echo "Set your Anthropic API key: export ANTHROPIC_API_KEY=sk-..."
