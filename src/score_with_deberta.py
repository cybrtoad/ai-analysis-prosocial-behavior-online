"""
src/score_with_deberta.py
==========================
Load a trained DeBERTa checkpoint and score any parquet file.

Used to generate test-set predictions after training is complete.
The output CSV feeds directly into evaluate_deberta_vs_humans.py and
compare_deberta_models.py.

Usage
-----
  # Score the Phase 1 test set with the Stage 3 large model
  python src/score_with_deberta.py \\
      --checkpoint output/models/deberta-v3-large/stage3/best \\
      --model-name microsoft/deberta-v3-large \\
      --input     output/splits/test.parquet \\
      --output    output/models/deberta-v3-large/stage3/test_predictions.csv

  # Score the Phase 1 test set with the Stage 3 base model
  python src/score_with_deberta.py \\
      --checkpoint output/models/deberta-v3-base/stage3/best \\
      --model-name microsoft/deberta-v3-base \\
      --input     output/splits/test.parquet \\
      --output    output/models/deberta-v3-base/stage3/test_predictions.csv

  # Score an arbitrary parquet (e.g., new data or the Phase 2 sample)
  python src/score_with_deberta.py \\
      --checkpoint output/models/deberta-v3-large/stage3/best \\
      --model-name microsoft/deberta-v3-large \\
      --input     output/phase2/phase2_sample.parquet \\
      --output    output/models/deberta-v3-large/stage3/phase2_predictions.csv

Output format
-------------
  CSV with columns: record_id, empathy, constructiveness, respect,
  social_cohesion, prosociality_composite
  One row per record in the input parquet.

Notes
-----
  Imports DeBERTaRegressor, ProsocialityDataset, _collate, and evaluate()
  directly from train_deberta.py — no code duplication.
  Requires a CUDA-capable GPU for reasonable throughput on large inputs.
  CPU inference is supported but will be slow for >10K records.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Reuse model classes and utilities from train_deberta.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_deberta import DeBERTaRegressor, ProsocialityDataset, _collate, DIMENSIONS

ROOT    = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "score_with_deberta.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(
    model: DeBERTaRegressor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list]:
    """Run inference on all batches; return record_ids and per-dimension predictions."""
    model.eval()
    preds_all  = {dim: [] for dim in DIMENSIONS}
    ids_all    = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        out = model(input_ids, attention_mask, token_type_ids)

        for i, dim in enumerate(DIMENSIONS):
            preds_all[dim].extend(out[:, i].cpu().numpy().tolist())
        ids_all.extend(batch["record_id"])

    return {"record_ids": ids_all, "preds": preds_all}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a parquet file with a trained DeBERTa checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to saved checkpoint directory (contains encoder/ and regressor.pt).",
    )
    parser.add_argument(
        "--model-name", required=True,
        choices=["microsoft/deberta-v3-large", "microsoft/deberta-v3-base"],
        help="HuggingFace model identifier (must match the checkpoint).",
    )
    parser.add_argument(
        "--input", required=True,
        help="Parquet file to score.  Must contain pre_intervention_text, "
             "intervention_text, post_intervention_text, record_id.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max token length (default: 512; must match training).",
    )
    args = parser.parse_args()

    ckpt_path  = Path(args.checkpoint)
    input_path = Path(args.input)
    out_path   = Path(args.output)

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    log.info("=== Score with DeBERTa ===")
    log.info(f"  checkpoint : {ckpt_path}")
    log.info(f"  model      : {args.model_name}")
    log.info(f"  input      : {input_path}")
    log.info(f"  output     : {out_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  device     : {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU        : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_parquet(input_path)
    log.info(f"Loaded {len(df):,} records from {input_path.name}")

    # Drop invalid triples if present
    if "invalid_triple" in df.columns:
        n_before = len(df)
        df = df[~df["invalid_triple"].fillna(False)].copy()
        log.info(f"Dropped {n_before - len(df):,} invalid triples → {len(df):,} records")

    # ------------------------------------------------------------------
    # Tokenizer + dataset
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    dataset   = ProsocialityDataset(df, tokenizer, args.max_length)
    loader    = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    log.info(f"Loading model from {ckpt_path}")
    model = DeBERTaRegressor.load(ckpt_path, args.model_name).to(device)
    log.info(f"Model loaded  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    log.info("Running inference...")
    results = run_inference(model, loader, device)

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    rows = {"record_id": results["record_ids"]}
    for dim in DIMENSIONS:
        rows[dim] = results["preds"][dim]
    rows["prosociality_composite"] = [
        float(np.nanmean([results["preds"][d][i] for d in DIMENSIONS]))
        for i in range(len(results["record_ids"]))
    ]

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    log.info(f"Saved {len(out_df):,} predictions → {out_path}")
    log.info(f"  Predicted composite — mean: {out_df['prosociality_composite'].mean():.3f}  "
             f"std: {out_df['prosociality_composite'].std():.3f}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
