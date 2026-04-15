#!/bin/bash
#SBATCH --job-name=soc_setup
#SBATCH --account=advdls25
#SBATCH --partition=inv-soc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=setup_env_%j.out
#SBATCH --error=setup_env_%j.err

set -e

ENV_NAME="soc"
REQUIREMENTS="$HOME/soc/requirements.txt"

echo "=== SOC Environment Setup ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

module load miniconda3/24.3.0

echo "=== Creating conda environment: $ENV_NAME ==="
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists — removing and rebuilding"
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=3.11 -y
echo "Environment created."
echo ""

echo "=== Activating environment ==="
source activate $ENV_NAME

echo "=== Installing PyTorch with CUDA 12.6 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== Installing remaining requirements ==="
pip install -r $REQUIREMENTS

echo ""
echo "=== Verifying installation ==="
python - <<'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import transformers
print(f"Transformers: {transformers.__version__}")

import pandas
print(f"Pandas: {pandas.__version__}")

import anthropic
print(f"Anthropic SDK: {anthropic.__version__}")

print("")
print("All packages verified successfully.")
EOF

echo ""
echo "=== Setup complete ==="
echo "To use this environment in future jobs, add to your script:"
echo "  module load miniconda3/24.3.0"
echo "  source activate soc"
