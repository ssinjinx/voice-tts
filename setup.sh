#!/usr/bin/env bash
# Set up the voice-tts conda environment
#
# Prerequisites:
#   - conda/miniforge installed
#   - ROCm installed (tested with 7.2.0 at /opt/rocm-7.2.0)
#   - Ollama running with a model pulled (e.g. ollama pull minimax-m2.5:cloud)

set -e

conda create -n voice-tts python=3.12 -y
conda activate voice-tts

# ROCm PyTorch — MUST install before qwen-tts so it picks up the GPU build.
# rocm6.3 wheels are backward-compatible with ROCm 7.2.0.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

pip install -U qwen-tts soundfile

echo ""
echo "Setup complete. Activate with: conda activate voice-tts"
echo "Verify GPU: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
