#!/bin/bash
#SBATCH --job-name=soc_deberta_base
#SBATCH --account=advdls25
#SBATCH --partition=inv-soc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_base_%j.out
#SBATCH --error=logs/train_base_%j.err

# DeBERTa-v3-base three-stage training (comparison baseline)
# ~4 hours total on 1x L40S GPU
#
# Identical pipeline to train_deberta_large.sh; only the model name
# and resource allocation differ.  Submit both jobs simultaneously:
#   sbatch scripts/train_deberta_large.sh
#   sbatch scripts/train_deberta_base.sh

set -e

echo "=== DeBERTa-v3-base Training (Comparison Baseline) ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo ""

module load miniconda3/24.3.0
source activate soc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Working directory: $(pwd)"
echo ""

MODEL="microsoft/deberta-v3-base"
TAG="deberta-v3-base"
OUT="output/models/$TAG"
TRAIN_SPLIT="output/splits/train.parquet"
VAL_SPLIT="output/splits/val.parquet"
PHASE2_DATA="output/phase2/phase2_claude_scores.parquet"

mkdir -p "$OUT" logs

# ------------------------------------------------------------------
# Stage 1 — Broad initialization on ~100K Phase 2 Claude scores
# ------------------------------------------------------------------
echo "--- Stage 1 ---"
if [ -f "$PHASE2_DATA" ]; then
    echo "Phase 2 data found. Running Stage 1..."
    python src/train_deberta.py \
        --model-name "$MODEL" \
        --stage 1 \
        --train-data "$PHASE2_DATA" \
        --val-data "$VAL_SPLIT" \
        --output-dir "$OUT" \
        --fp16
    STAGE2_CKPT="$OUT/stage1/best"
    echo "Stage 1 complete."
else
    echo "Phase 2 data not found — skipping Stage 1 (cold start for Stage 2)."
    STAGE2_CKPT=""
fi
echo ""

# ------------------------------------------------------------------
# Stage 2 — Quality refinement on ~3K Phase 1 gold labels
# ------------------------------------------------------------------
echo "--- Stage 2 ---"
if [ -n "$STAGE2_CKPT" ]; then
    python src/train_deberta.py \
        --model-name "$MODEL" \
        --stage 2 \
        --train-data "$TRAIN_SPLIT" \
        --val-data "$VAL_SPLIT" \
        --checkpoint "$STAGE2_CKPT" \
        --output-dir "$OUT" \
        --fp16
else
    python src/train_deberta.py \
        --model-name "$MODEL" \
        --stage 2 \
        --train-data "$TRAIN_SPLIT" \
        --val-data "$VAL_SPLIT" \
        --output-dir "$OUT" \
        --fp16
fi
echo "Stage 2 complete."
echo ""

# ------------------------------------------------------------------
# Stage 3 — Human alignment + holdout predictions
# ------------------------------------------------------------------
echo "--- Stage 3 ---"
HUMAN_TRAIN="output/human_labels/human_train.parquet"
HUMAN_VAL="output/human_labels/human_val.parquet"

if [ -f "$HUMAN_TRAIN" ] && [ -f "$HUMAN_VAL" ]; then
    echo "Human annotation data found. Running Stage 3..."
    python src/train_deberta.py \
        --model-name "$MODEL" \
        --stage 3 \
        --train-data "$HUMAN_TRAIN" \
        --val-data "$HUMAN_VAL" \
        --checkpoint "$OUT/stage2/best" \
        --output-dir "$OUT" \
        --fp16 \
        --predict-holdout
    echo "Stage 3 complete."
    echo ""
    echo "Holdout predictions saved to $OUT/stage3/holdout_predictions.csv"
    echo "Run: python src/evaluate_claude_deberta_agreement.py \\"
    echo "    --predictions $OUT/stage3/holdout_predictions.csv \\"
    echo "    --model-label $TAG"
else
    echo "Human annotation data not found — Stage 3 skipped."
    echo "Run manually after human annotation is complete:"
    echo ""
    echo "  python src/train_deberta.py \\"
    echo "      --model-name $MODEL --stage 3 \\"
    echo "      --train-data $HUMAN_TRAIN \\"
    echo "      --val-data $HUMAN_VAL \\"
    echo "      --checkpoint $OUT/stage2/best \\"
    echo "      --output-dir $OUT --fp16 --predict-holdout"
fi

# ------------------------------------------------------------------
# Test-set scoring — runs only if Stage 3 produced a checkpoint
# ------------------------------------------------------------------
TEST_SPLIT="output/splits/test.parquet"
TEST_PREDS="$OUT/stage3/test_predictions.csv"
STAGE3_CKPT="$OUT/stage3/best"

echo ""
echo "--- Test-set scoring ---"
if [ -d "$STAGE3_CKPT" ] && [ -f "$TEST_SPLIT" ]; then
    echo "Scoring test split with Stage 3 checkpoint..."
    python src/score_with_deberta.py \
        --checkpoint "$STAGE3_CKPT" \
        --model-name "$MODEL" \
        --input     "$TEST_SPLIT" \
        --output    "$TEST_PREDS"
    echo "Test predictions saved to $TEST_PREDS"
else
    echo "Stage 3 checkpoint or test split not found — skipping test scoring."
    echo "Run manually:"
    echo "  python src/score_with_deberta.py \\"
    echo "      --checkpoint $STAGE3_CKPT \\"
    echo "      --model-name $MODEL \\"
    echo "      --input     $TEST_SPLIT \\"
    echo "      --output    $TEST_PREDS"
fi

echo ""
echo "=== DeBERTa-v3-base training complete: $(date) ==="
