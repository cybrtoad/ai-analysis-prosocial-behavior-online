#!/bin/bash
#SBATCH --job-name=soc_download_weights
#SBATCH --account=advdls25
#SBATCH --partition=inv-soc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_weights_%j.out
#SBATCH --error=logs/download_weights_%j.err

# Pre-download DeBERTa-v3-large and DeBERTa-v3-base weights from HuggingFace.
#
# Run this ONCE before submitting the training jobs.  It caches the model
# weights to ~/.cache/huggingface so compute nodes don't need internet access
# during training.
#
# Submit via OOD Job Composer or after SSH login:
#   sbatch scripts/download_weights.sh

set -e

echo "=== DeBERTa Weight Pre-Download ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo ""

module load miniconda3/24.3.0
source activate soc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Working directory: $(pwd)"

mkdir -p logs

python scripts/download_deberta_weights.py

echo ""
echo "=== Download complete: $(date) ==="
echo "You can now submit:"
echo "  sbatch scripts/train_deberta_large.sh"
echo "  sbatch scripts/train_deberta_base.sh"
