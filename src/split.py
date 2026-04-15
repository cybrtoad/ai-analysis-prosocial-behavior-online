"""
Step 8: Create Train / Val / Test Splits + Claude-DeBERTa Agreement Holdout
=============================================================================
Reads output/labels/gold_labels_merged.parquet (880 valid records after dropping
invalid_triple=True).  First extracts a stratified holdout set (~10%) that is
excluded from ALL DeBERTa training stages and reserved for post-training
Claude-DeBERTa agreement validation.  Then performs a stratified 80/10/10 split
of the remaining records by (platform × intervention_type), saving:

  output/splits/claude_agreement_holdout.parquet  — held out from all training
  output/splits/train.parquet
  output/splits/val.parquet
  output/splits/test.parquet
  output/splits/split_metadata.json               — counts + score statistics

Usage:
  python src/split.py
  python src/split.py --random-state 7       # override seed
  python src/split.py --holdout-frac 0.10    # default holdout fraction

Notes:
  - The claude_agreement_holdout is NEVER used during DeBERTa training (Stages 1–3).
    It is used ONLY after training to evaluate how closely DeBERTa replicates
    Claude's scores.  See src/evaluate_claude_deberta_agreement.py.
  - Strata with fewer than 5 records are not split for the holdout; those records
    go entirely to the main pool.
  - Some strata are very small (wikipedia/positive_reinforcement, wikipedia/reframe).
    The 80/10/10 script rounds to guarantee at least 1 record in val and test per
    stratum when the stratum has ≥ 3 records; strata with < 3 records go to train.
  - The final split sizes may differ slightly from 80/10/10 due to rounding.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "output" / "labels"
SPLITS_DIR = ROOT / "output" / "splits"
LOG_DIR    = ROOT / "logs"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "split.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for dim in DIMENSIONS:
        col = df[dim].dropna()
        stats[dim] = {
            "n":    int(len(col)),
            "mean": round(float(col.mean()), 4),
            "std":  round(float(col.std()),  4),
            "min":  round(float(col.min()),  4),
            "max":  round(float(col.max()),  4),
        }
    return stats


def _stratum_split(df: pd.DataFrame, val_frac: float, test_frac: float,
                   random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test; if too small all go to train."""
    n = len(df)
    # We need at least 3 records to guarantee 1 in val and 1 in test
    if n < 3:
        return df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    n_val  = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))

    # ensure we don't ask for more than available
    if n_val + n_test >= n:
        n_val  = max(1, (n - 1) // 2)
        n_test = max(1, (n - 1) - n_val)

    temp_frac  = (n_val + n_test) / n
    train_df, holdout = train_test_split(
        df, test_size=temp_frac, random_state=random_state
    )
    val_share = n_val / (n_val + n_test)
    val_df, test_df = train_test_split(
        holdout, test_size=1 - val_share, random_state=random_state
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _extract_holdout(
    df: pd.DataFrame,
    holdout_frac: float,
    random_state: int,
    strata_key: list[str],
    min_stratum_size: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified extraction of a holdout set for Claude-DeBERTa agreement testing.

    Records from strata with fewer than min_stratum_size rows are not split;
    they go entirely to the main pool to avoid over-fragmenting small groups.

    Returns (holdout_df, remaining_df).  The holdout is EXCLUDED from all
    DeBERTa training stages and used only for post-training validation.
    """
    holdout_parts: list[pd.DataFrame] = []
    remaining_parts: list[pd.DataFrame] = []

    for _key, grp in df.groupby(strata_key):
        n = len(grp)
        n_holdout = round(n * holdout_frac)
        if n < min_stratum_size or n_holdout < 1:
            remaining_parts.append(grp)
        else:
            main_grp, holdout_grp = train_test_split(
                grp, test_size=n_holdout, random_state=random_state
            )
            holdout_parts.append(holdout_grp)
            remaining_parts.append(main_grp)

    holdout = (
        pd.concat(holdout_parts, ignore_index=True)
        if holdout_parts
        else pd.DataFrame(columns=df.columns)
    )
    remaining = (
        pd.concat(remaining_parts, ignore_index=True)
        if remaining_parts
        else pd.DataFrame(columns=df.columns)
    )
    return holdout, remaining


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Claude-DeBERTa holdout + 80/10/10 stratified train/val/test splits"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.10,
        help="Fraction of valid records to reserve as Claude-DeBERTa agreement holdout "
             "(default: 0.10).  These records are excluded from all DeBERTa training stages.",
    )
    args = parser.parse_args()

    log.info("=== Step 8: Create Claude-DeBERTa Holdout + Train/Val/Test Splits ===")
    log.info(f"Random state: {args.random_state}")
    log.info(f"Holdout fraction: {args.holdout_frac:.0%}")

    # ------------------------------------------------------------------
    # 1. Load merged labels
    # ------------------------------------------------------------------
    merged_path = LABELS_DIR / "gold_labels_merged.parquet"
    if not merged_path.exists():
        log.error(f"Missing: {merged_path}  — run validate_labels.py first")
        sys.exit(1)

    df = pd.read_parquet(merged_path)
    log.info(f"Loaded {len(df):,} records from {merged_path.name}")

    # ------------------------------------------------------------------
    # 2. Keep only valid triples
    # ------------------------------------------------------------------
    df_valid = df[~df["invalid_triple"].fillna(False)].copy()
    n_dropped = len(df) - len(df_valid)
    log.info(f"Dropped {n_dropped:,} invalid triples → {len(df_valid):,} valid records")

    # ------------------------------------------------------------------
    # 3. Extract Claude-DeBERTa agreement holdout BEFORE any training splits
    # ------------------------------------------------------------------
    strata_key = ["platform", "intervention_type"]

    holdout, df_main = _extract_holdout(
        df_valid,
        holdout_frac=args.holdout_frac,
        random_state=args.random_state,
        strata_key=strata_key,
    )
    log.info(
        f"\nClaude-DeBERTa agreement holdout: {len(holdout):,} records "
        f"({len(holdout)/len(df_valid)*100:.1f}% of valid)\n"
        f"Remaining for train/val/test: {len(df_main):,} records"
    )
    log.info(
        "NOTE: The holdout is excluded from all DeBERTa training stages. "
        "Use src/evaluate_claude_deberta_agreement.py after training completes."
    )

    # ------------------------------------------------------------------
    # 4. Stratified 80/10/10 split of the remaining records
    # ------------------------------------------------------------------
    train_parts, val_parts, test_parts = [], [], []

    strata_counts = df_main.groupby(strata_key).size()
    log.info("\nStratum sizes (main pool after holdout extraction):")
    for (plat, itype), n in strata_counts.items():
        log.info(f"  {plat:15s} / {itype:25s}  n={n}")

    for (plat, itype), grp in df_main.groupby(strata_key):
        tr, va, te = _stratum_split(
            grp.copy(),
            val_frac=0.10,
            test_frac=0.10,
            random_state=args.random_state,
        )
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        log.info(
            f"  {plat}/{itype}: "
            f"train={len(tr)} val={len(va)} test={len(te)}"
        )

    train = pd.concat(train_parts, ignore_index=True)
    val   = pd.concat(val_parts,   ignore_index=True)
    test  = pd.concat(test_parts,  ignore_index=True)

    n_total = len(train) + len(val) + len(test)
    log.info(
        f"\nFinal split sizes: "
        f"train={len(train)} ({len(train)/n_total*100:.1f}%)  "
        f"val={len(val)} ({len(val)/n_total*100:.1f}%)  "
        f"test={len(test)} ({len(test)/n_total*100:.1f}%)  "
        f"total={n_total}"
    )

    # ------------------------------------------------------------------
    # 5. Verify no overlap across all four sets
    # ------------------------------------------------------------------
    ids_holdout = set(holdout["record_id"])
    ids_train   = set(train["record_id"])
    ids_val     = set(val["record_id"])
    ids_test    = set(test["record_id"])

    overlaps = {
        "holdout∩train": ids_holdout & ids_train,
        "holdout∩val":   ids_holdout & ids_val,
        "holdout∩test":  ids_holdout & ids_test,
        "train∩val":     ids_train & ids_val,
        "train∩test":    ids_train & ids_test,
        "val∩test":      ids_val   & ids_test,
    }
    any_overlap = any(len(v) > 0 for v in overlaps.values())
    if any_overlap:
        details = "  ".join(f"{k}={len(v)}" for k, v in overlaps.items() if v)
        log.error(f"Overlapping record_ids detected!  {details}")
        sys.exit(1)
    else:
        log.info("Overlap check passed — no record appears in more than one set.")

    # ------------------------------------------------------------------
    # 6. Save parquets
    # ------------------------------------------------------------------
    holdout_path = SPLITS_DIR / "claude_agreement_holdout.parquet"
    holdout.to_parquet(holdout_path, index=False)
    log.info(f"Saved holdout → {holdout_path}  ({len(holdout):,} rows)")

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = SPLITS_DIR / f"{split_name}.parquet"
        split_df.to_parquet(out, index=False)
        log.info(f"Saved {split_name:5s} → {out}  ({len(split_df):,} rows)")

    # ------------------------------------------------------------------
    # 7. Build and save metadata JSON
    # ------------------------------------------------------------------
    metadata = {
        "random_state": args.random_state,
        "holdout_frac": args.holdout_frac,
        "total_valid_records": int(len(df_valid)),
        "holdout_n": int(len(holdout)),
        "holdout_purpose": (
            "Claude-DeBERTa agreement validation. Excluded from all DeBERTa "
            "training stages. Use src/evaluate_claude_deberta_agreement.py "
            "after training completes."
        ),
        "splits": {},
    }

    for split_name, split_df in [
        ("claude_agreement_holdout", holdout),
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        by_strat = (
            split_df.groupby(strata_key)
            .size()
            .rename("count")
            .reset_index()
            .to_dict(orient="records")
        )
        metadata["splits"][split_name] = {
            "n": int(len(split_df)),
            "pct": round(len(split_df) / len(df_valid) * 100, 2),
            "score_stats": _score_stats(split_df),
            "by_stratum": by_strat,
        }

    meta_path = SPLITS_DIR / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata → {meta_path}")

    # ------------------------------------------------------------------
    # 8. Print summary table
    # ------------------------------------------------------------------
    log.info("\n--- Score statistics per set (means) ---")
    header = f"{'Set':26s}  {'n':>5s}  " + "  ".join(f"{d[:4]:>6s}" for d in DIMENSIONS)
    log.info(header)
    for set_name, set_df in [
        ("claude_agreement_holdout", holdout),
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        means = "  ".join(f"{set_df[d].mean():6.3f}" for d in DIMENSIONS)
        log.info(f"{set_name:26s}  {len(set_df):>5d}  {means}")

    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()
