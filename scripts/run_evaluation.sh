#!/bin/bash
# scripts/run_evaluation.sh
# =========================
# Orchestration script: run all post-training evaluation steps in sequence.
#
# Prerequisites (must all exist before running):
#   output/models/deberta-v3-large/stage3/test_predictions.csv
#   output/models/deberta-v3-base/stage3/test_predictions.csv
#   output/models/deberta-v3-large/stage3/holdout_predictions.csv
#   output/models/deberta-v3-base/stage3/holdout_predictions.csv
#   output/human_labels/human_train.parquet
#   output/human_labels/human_val.parquet
#
# If test predictions are missing, this script will generate them first
# using score_with_deberta.py (requires Stage 3 checkpoints).
#
# Steps executed:
#   1. Score test set with large model     (if predictions absent)
#   2. Score test set with base model      (if predictions absent)
#   3. DeBERTa-large vs. humans            evaluate_deberta_vs_humans.py
#   4. DeBERTa-base  vs. humans            evaluate_deberta_vs_humans.py
#   5. DeBERTa-large vs. Claude holdout    evaluate_claude_deberta_agreement.py
#   6. DeBERTa-base  vs. Claude holdout    evaluate_claude_deberta_agreement.py
#   7. Compare large vs. base              compare_deberta_models.py
#
# Usage:
#   bash scripts/run_evaluation.sh
#
# Run from the project root or ARCC after training jobs finish.

set -e

# Navigate to project root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Working directory: $(pwd)"
echo ""

# Activate conda env if on ARCC (no-op on local machine if already active)
if command -v module &>/dev/null; then
    module load miniconda3/24.3.0
    source activate soc
fi

LARGE_MODEL="microsoft/deberta-v3-large"
BASE_MODEL="microsoft/deberta-v3-base"
LARGE_TAG="deberta-v3-large"
BASE_TAG="deberta-v3-base"
LARGE_OUT="output/models/$LARGE_TAG"
BASE_OUT="output/models/$BASE_TAG"
TEST_SPLIT="output/splits/test.parquet"
LARGE_TEST_PREDS="$LARGE_OUT/stage3/test_predictions.csv"
BASE_TEST_PREDS="$BASE_OUT/stage3/test_predictions.csv"
LARGE_HOLDOUT_PREDS="$LARGE_OUT/stage3/holdout_predictions.csv"
BASE_HOLDOUT_PREDS="$BASE_OUT/stage3/holdout_predictions.csv"

echo "=== Prosocial Interventions — Post-Training Evaluation ==="
echo "Date: $(date)"
echo ""

# ------------------------------------------------------------------
# Verify required files
# ------------------------------------------------------------------
MISSING=0
for f in \
    "$LARGE_OUT/stage3/best" \
    "$BASE_OUT/stage3/best" \
    "$TEST_SPLIT" \
    "$LARGE_HOLDOUT_PREDS" \
    "$BASE_HOLDOUT_PREDS" \
    "output/human_labels/human_train.parquet" \
    "output/human_labels/human_val.parquet"
do
    if [ ! -e "$f" ]; then
        echo "ERROR: Missing required file/dir: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Resolve missing files then re-run this script."
    exit 1
fi
echo "All required files present."
echo ""

# ------------------------------------------------------------------
# Step 1 & 2 — Generate test predictions if not already present
# ------------------------------------------------------------------
echo "--- Step 1/2: Test-set scoring ---"

if [ ! -f "$LARGE_TEST_PREDS" ]; then
    echo "Generating test predictions for large model..."
    python src/score_with_deberta.py \
        --checkpoint "$LARGE_OUT/stage3/best" \
        --model-name "$LARGE_MODEL" \
        --input     "$TEST_SPLIT" \
        --output    "$LARGE_TEST_PREDS"
    echo "Saved: $LARGE_TEST_PREDS"
else
    echo "Large model test predictions already exist — skipping."
fi

if [ ! -f "$BASE_TEST_PREDS" ]; then
    echo "Generating test predictions for base model..."
    python src/score_with_deberta.py \
        --checkpoint "$BASE_OUT/stage3/best" \
        --model-name "$BASE_MODEL" \
        --input     "$TEST_SPLIT" \
        --output    "$BASE_TEST_PREDS"
    echo "Saved: $BASE_TEST_PREDS"
else
    echo "Base model test predictions already exist — skipping."
fi
echo ""

# ------------------------------------------------------------------
# Step 3 — DeBERTa-large vs. human consensus
# ------------------------------------------------------------------
echo "--- Step 3: DeBERTa-large vs. human consensus ---"
python src/evaluate_deberta_vs_humans.py \
    --predictions "$LARGE_TEST_PREDS" \
    --model-label "$LARGE_TAG"
echo "Report: output/artifacts/human_${LARGE_TAG}_eval_report.txt"
echo ""

# ------------------------------------------------------------------
# Step 4 — DeBERTa-base vs. human consensus
# ------------------------------------------------------------------
echo "--- Step 4: DeBERTa-base vs. human consensus ---"
python src/evaluate_deberta_vs_humans.py \
    --predictions "$BASE_TEST_PREDS" \
    --model-label "$BASE_TAG"
echo "Report: output/artifacts/human_${BASE_TAG}_eval_report.txt"
echo ""

# ------------------------------------------------------------------
# Step 5 — DeBERTa-large vs. Claude (holdout set)
# ------------------------------------------------------------------
echo "--- Step 5: DeBERTa-large vs. Claude holdout ---"
python src/evaluate_claude_deberta_agreement.py \
    --predictions "$LARGE_HOLDOUT_PREDS" \
    --model-label "$LARGE_TAG"
echo "Report: output/artifacts/claude_deberta_${LARGE_TAG}_agreement_report.txt"
echo ""

# ------------------------------------------------------------------
# Step 6 — DeBERTa-base vs. Claude (holdout set)
# ------------------------------------------------------------------
echo "--- Step 6: DeBERTa-base vs. Claude holdout ---"
python src/evaluate_claude_deberta_agreement.py \
    --predictions "$BASE_HOLDOUT_PREDS" \
    --model-label "$BASE_TAG"
echo "Report: output/artifacts/claude_deberta_${BASE_TAG}_agreement_report.txt"
echo ""

# ------------------------------------------------------------------
# Step 7 — Compare large vs. base
# ------------------------------------------------------------------
echo "--- Step 7: Compare DeBERTa-large vs. DeBERTa-base ---"
python src/compare_deberta_models.py \
    --large-human  "output/artifacts/human_${LARGE_TAG}_eval_report.json" \
    --base-human   "output/artifacts/human_${BASE_TAG}_eval_report.json" \
    --large-claude "output/artifacts/claude_deberta_${LARGE_TAG}_agreement_report.json" \
    --base-claude  "output/artifacts/claude_deberta_${BASE_TAG}_agreement_report.json"
echo "Report: output/artifacts/deberta_model_comparison.txt"
echo ""

echo "=== Evaluation complete: $(date) ==="
echo ""
echo "Summary of outputs:"
echo "  output/artifacts/human_${LARGE_TAG}_eval_report.txt"
echo "  output/artifacts/human_${BASE_TAG}_eval_report.txt"
echo "  output/artifacts/claude_deberta_${LARGE_TAG}_agreement_report.txt"
echo "  output/artifacts/claude_deberta_${BASE_TAG}_agreement_report.txt"
echo "  output/artifacts/deberta_model_comparison.txt"
echo "  output/artifacts/phase2_tables/model_comparison_table.csv"
