#!/bin/bash
# scripts/sync_to_arcc.sh
# =======================
# Sync local output data files to ARCC scratch space.
#
# Run from the project root on your LOCAL machine in an interactive terminal
# (Duo 2FA is required — this cannot run unattended).
#
# Usage
# -----
#   bash scripts/sync_to_arcc.sh splits    # Push Phase 1 splits right now
#   bash scripts/sync_to_arcc.sh phase2    # Push Phase 2 scores when ready
#   bash scripts/sync_to_arcc.sh human     # Push human annotation labels when ready
#   bash scripts/sync_to_arcc.sh all       # Push everything listed above
#
# Remote layout
# -------------
#   /gscratch/tlinse/soc/output/splits/        Phase 1 train/val/test/holdout
#   /gscratch/tlinse/soc/output/phase2/        Phase 2 Claude-scored 100K records
#   /gscratch/tlinse/soc/output/human_labels/  Human annotation consensus splits
#
# Notes
# -----
#   - Code changes are handled via git push/pull; this script is for data only.
#   - rsync skips files that are already up to date (safe to re-run).
#   - The -P flag shows progress and resumes interrupted transfers.

set -e

REMOTE_HOST="arcc"
REMOTE_ROOT="/gscratch/tlinse/soc"

LOCAL_SPLITS="output/splits/"
LOCAL_PHASE2="output/phase2/"
LOCAL_HUMAN="output/human_labels/"

REMOTE_SPLITS="$REMOTE_HOST:$REMOTE_ROOT/output/splits/"
REMOTE_PHASE2="$REMOTE_HOST:$REMOTE_ROOT/output/phase2/"
REMOTE_HUMAN="$REMOTE_HOST:$REMOTE_ROOT/output/human_labels/"

RSYNC="rsync -avz --progress"

# Make sure remote output directories exist
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_ROOT/output/splits $REMOTE_ROOT/output/phase2 $REMOTE_ROOT/output/human_labels"

STAGE="${1:-help}"

case "$STAGE" in

  splits)
    echo "=== Syncing Phase 1 splits → ARCC ==="
    $RSYNC "$LOCAL_SPLITS" "$REMOTE_SPLITS"
    echo "Done."
    ;;

  phase2)
    if [ ! -f "output/phase2/phase2_claude_scores.parquet" ]; then
      echo "ERROR: output/phase2/phase2_claude_scores.parquet not found."
      echo "Phase 2 scoring is still in progress — run this after it completes."
      exit 1
    fi
    echo "=== Syncing Phase 2 Claude scores → ARCC ==="
    $RSYNC "$LOCAL_PHASE2" "$REMOTE_PHASE2"
    echo "Done."
    ;;

  human)
    if [ ! -f "output/human_labels/human_train.parquet" ]; then
      echo "ERROR: output/human_labels/human_train.parquet not found."
      echo "Run src/process_human_annotations.py first."
      exit 1
    fi
    echo "=== Syncing human annotation labels → ARCC ==="
    $RSYNC "$LOCAL_HUMAN" "$REMOTE_HUMAN"
    echo "Done."
    ;;

  all)
    bash "$0" splits
    echo ""
    # Phase 2 and human are synced only if their output files exist
    if [ -f "output/phase2/phase2_claude_scores.parquet" ]; then
      bash "$0" phase2
      echo ""
    else
      echo "Skipping phase2 sync (phase2_claude_scores.parquet not yet available)."
      echo ""
    fi
    if [ -f "output/human_labels/human_train.parquet" ]; then
      bash "$0" human
      echo ""
    else
      echo "Skipping human sync (human_train.parquet not yet available)."
      echo ""
    fi
    echo "=== All available data synced ==="
    ;;

  help|*)
    echo "Usage: bash scripts/sync_to_arcc.sh [splits|phase2|human|all]"
    echo ""
    echo "  splits   Push output/splits/ to ARCC (Phase 1 gold labels + holdout)"
    echo "  phase2   Push output/phase2/ to ARCC (Phase 2 Claude scores, ~100K records)"
    echo "  human    Push output/human_labels/ to ARCC (human consensus splits)"
    echo "  all      Push whichever of the above are locally available"
    exit 0
    ;;

esac
