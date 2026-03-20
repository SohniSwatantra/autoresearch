#!/bin/bash
set -e

echo "=============================================="
echo "Nosana MCP Autoresearch Setup"
echo "=============================================="

# System dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    git curl python3 python3-pip python3-venv wget \
    build-essential zstd

# Install ollama
echo "Installing ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start ollama server in background
echo "Starting ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 10

# Pull Qwen model
echo "Pulling Qwen 3.5 9B model..."
ollama pull qwen3.5:9b

# Verify ollama is working
echo "Verifying ollama..."
ollama list

# Install uv (Python package manager)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Use the workspace (repo already cloned here by job config)
cd /workspace

# Install crawler dependencies (not in pyproject.toml)
echo "Installing crawler dependencies..."
pip install requests beautifulsoup4 pandas

# Install autoresearch dependencies
echo "Installing autoresearch dependencies..."
uv sync

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Show available VRAM
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Total VRAM: {props.total_mem / 1024**3:.1f} GB')
else:
    print('No CUDA GPU available!')
"

echo ""
echo "Setup complete! Starting pipeline..."
echo ""

# Run the full pipeline
bash run_pipeline.sh
