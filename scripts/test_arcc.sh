#!/bin/bash
#SBATCH --job-name=soc_test
#SBATCH --account=advdls25
#SBATCH --partition=inv-soc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --output=test_arcc_%j.out
#SBATCH --error=test_arcc_%j.err

echo "=== ARCC Test Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

echo "=== Available partitions ==="
sinfo -o "%P %l %G %D %N" 2>/dev/null

echo ""
echo "=== Checking project allocation ==="
sacctmgr show assoc user=$USER format=Account,User,Partition -n

echo ""
echo "=== Python / PyTorch check ==="
module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
echo "Loaded modules:"
module list 2>&1
python3 - <<'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        x = torch.tensor([1.0, 2.0]).cuda()
        print(f"GPU tensor test: {x.sum().item()} (expected 3.0)")
except ImportError:
    print("PyTorch not found — may need to activate conda env")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers not found")
EOF

echo ""
echo "=== Disk quotas ==="
df -h $HOME 2>/dev/null
df -h /gscratch/$USER 2>/dev/null || true

echo ""
echo "=== Done ==="
