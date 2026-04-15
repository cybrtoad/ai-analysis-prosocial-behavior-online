"""
Step 7: Validate Label Quality
================================
Joins gold_labels with labeling_sample, then produces a thorough quality report:

  1. Overall score statistics
  2. Score distributions by platform × intervention_type
  3. Invalid-triple breakdown by platform
  4. Inter-dimension correlations
  5. Confidence distribution
  6. Surprising records (high/low confidence outliers worth manual review)
  7. Saves a clean merged parquet: output/labels/gold_labels_merged.parquet

Input:
  output/labels/gold_labels.parquet
  output/processed/labeling_sample.parquet

Output:
  output/labels/gold_labels_merged.parquet  — labels joined to sample metadata
  logs/validate_labels.log
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "output" / "labels"
PROC_DIR   = ROOT / "output" / "processed"
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "validate_labels.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(value: float, max_val: float = 5.0, width: int = 20) -> str:
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 7: Validate Label Quality ===")

    # ------------------------------------------------------------------
    # 1. Load and merge
    # ------------------------------------------------------------------
    labels = pd.read_parquet(LABELS_DIR / "gold_labels.parquet")
    sample = pd.read_parquet(PROC_DIR   / "labeling_sample.parquet")

    log.info(f"Labels:  {len(labels):,} rows")
    log.info(f"Sample:  {len(sample):,} rows")

    merged = sample.merge(labels, on="record_id", how="inner")
    log.info(f"Merged:  {len(merged):,} rows")

    if len(merged) < len(labels):
        log.warning(
            f"  {len(labels) - len(merged)} label rows had no matching sample record"
        )

    # Save merged parquet for downstream use
    out_path = LABELS_DIR / "gold_labels_merged.parquet"
    merged.to_parquet(out_path, index=False)
    log.info(f"Saved merged -> {out_path}")

    # ------------------------------------------------------------------
    # Subset: valid vs invalid triples
    # ------------------------------------------------------------------
    valid   = merged[~merged["invalid_triple"].fillna(False)]
    invalid = merged[merged["invalid_triple"].fillna(False)]

    log.info(f"\nValid triples:   {len(valid):,}  ({len(valid)/len(merged)*100:.1f}%)")
    log.info(f"Invalid triples: {len(invalid):,}  ({len(invalid)/len(merged)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Invalid-triple breakdown by platform
    # ------------------------------------------------------------------
    _section("INVALID TRIPLE BREAKDOWN BY PLATFORM")
    inv_by_plat = (
        merged.groupby("platform")["invalid_triple"]
        .agg(total="count", n_invalid="sum")
        .assign(pct_invalid=lambda d: d["n_invalid"] / d["total"] * 100)
    )
    log.info("\n" + inv_by_plat.to_string())

    _section("INVALID TRIPLE BREAKDOWN BY PLATFORM × INTERVENTION TYPE")
    inv_detail = (
        merged.groupby(["platform", "intervention_type"])["invalid_triple"]
        .agg(total="count", n_invalid="sum")
        .assign(pct_invalid=lambda d: d["n_invalid"] / d["total"] * 100)
    )
    log.info("\n" + inv_detail.to_string())

    if "intervener_role" in merged.columns:
        _section("INVALID TRIPLE BREAKDOWN BY INTERVENER ROLE")
        inv_role = (
            merged.groupby("intervener_role")["invalid_triple"]
            .agg(total="count", n_invalid="sum")
            .assign(pct_invalid=lambda d: d["n_invalid"] / d["total"] * 100)
        )
        log.info("\n" + inv_role.to_string())

    # ------------------------------------------------------------------
    # 3. Overall score statistics (valid records only)
    # ------------------------------------------------------------------
    _section("OVERALL SCORE STATISTICS (valid records only)")
    for dim in DIMENSIONS:
        s = valid[dim].describe()
        log.info(
            f"\n  {dim.upper()}\n"
            f"    n={int(s['count']):,}  mean={s['mean']:.3f}  std={s['std']:.3f}  "
            f"median={valid[dim].median():.1f}  "
            f"min={s['min']:.1f}  max={s['max']:.1f}\n"
            f"    bar (mean): {_bar(s['mean'])}"
        )

    # ------------------------------------------------------------------
    # 4. Score distributions by platform (valid records)
    # ------------------------------------------------------------------
    _section("SCORE MEANS BY PLATFORM (valid records)")
    plat_means = valid.groupby("platform")[DIMENSIONS].mean().round(3)
    log.info("\n" + plat_means.to_string())

    _section("SCORE MEANS BY INTERVENTION TYPE (valid records)")
    type_means = valid.groupby("intervention_type")[DIMENSIONS].mean().round(3)
    log.info("\n" + type_means.to_string())

    if "intervener_role" in valid.columns:
        _section("SCORE MEANS BY INTERVENER ROLE (valid records)")
        role_means = valid.groupby("intervener_role")[DIMENSIONS].mean().round(3)
        log.info("\n" + role_means.to_string())

    _section("SCORE MEANS BY PLATFORM × INTERVENTION TYPE (valid records)")
    cross_means = (
        valid.groupby(["platform", "intervention_type"])[DIMENSIONS]
        .mean()
        .round(3)
    )
    log.info("\n" + cross_means.to_string())

    # ------------------------------------------------------------------
    # 5. Score value-count histograms (valid records)
    # ------------------------------------------------------------------
    _section("SCORE FREQUENCY DISTRIBUTIONS (valid records)")
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    for dim in DIMENSIONS:
        counts = valid[dim].value_counts().sort_index()
        log.info(f"\n  {dim}:")
        for score, cnt in counts.items():
            bar_w = int(round(cnt / len(valid) * 40))
            log.info(f"    {score:.1f}  {'█' * bar_w}  {cnt:4d}  ({cnt/len(valid)*100:5.1f}%)")

    # ------------------------------------------------------------------
    # 6. Inter-dimension correlations (valid records)
    # ------------------------------------------------------------------
    _section("INTER-DIMENSION CORRELATIONS (Pearson, valid records)")
    corr = valid[DIMENSIONS].corr().round(3)
    log.info("\n" + corr.to_string())

    # ------------------------------------------------------------------
    # 7. Confidence distribution
    # ------------------------------------------------------------------
    _section("CONFIDENCE DISTRIBUTION (all records)")
    conf = merged["confidence"].dropna()
    log.info(
        f"\n  n={len(conf):,}  mean={conf.mean():.3f}  std={conf.std():.3f}  "
        f"median={conf.median():.3f}  min={conf.min():.3f}  max={conf.max():.3f}"
    )
    for lo, hi in [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        n = ((conf >= lo) & (conf < hi)).sum()
        log.info(f"    [{lo:.1f}, {hi:.1f})  {'█' * int(n/len(conf)*40):40s}  {n:4d}  ({n/len(conf)*100:5.1f}%)")

    # ------------------------------------------------------------------
    # 8. Surprising / review-worthy records
    # ------------------------------------------------------------------
    _section("REVIEW CANDIDATES")

    # a) Low confidence valid records
    low_conf_valid = valid[valid["confidence"] < 0.65].sort_values("confidence")
    log.info(f"\n  Low-confidence valid records (conf < 0.65): {len(low_conf_valid):,}")
    for _, row in low_conf_valid.head(10).iterrows():
        scores = {d: row[d] for d in DIMENSIONS}
        log.info(
            f"    {row['record_id']}  [{row['platform']}/{row['intervention_type']}]  "
            f"conf={row['confidence']:.2f}  scores={scores}"
        )

    # b) Score outliers: high respect but low empathy (interesting asymmetry)
    asym = valid[(valid["respect"] >= 4.0) & (valid["empathy"] <= 1.5)]
    log.info(f"\n  High-respect / low-empathy records (resp>=4, emp<=1.5): {len(asym):,}")
    for _, row in asym.head(10).iterrows():
        scores = {d: row[d] for d in DIMENSIONS}
        log.info(
            f"    {row['record_id']}  [{row['platform']}/{row['intervention_type']}]  "
            f"conf={row['confidence']:.2f}  scores={scores}"
        )

    # c) High-scoring records (mean score >= 4.5) — exemplars
    valid = valid.copy()
    valid["mean_score"] = valid[DIMENSIONS].mean(axis=1)
    top = valid[valid["mean_score"] >= 4.5].sort_values("mean_score", ascending=False)
    log.info(f"\n  Top-scoring records (mean >= 4.5): {len(top):,}")
    for _, row in top.head(10).iterrows():
        scores = {d: row[d] for d in DIMENSIONS}
        log.info(
            f"    {row['record_id']}  [{row['platform']}/{row['intervention_type']}]  "
            f"mean={row['mean_score']:.2f}  conf={row['confidence']:.2f}  scores={scores}"
        )

    # d) All-1.0 valid records (suspicious even for valid triples)
    all_ones_valid = valid[(valid[DIMENSIONS] == 1.0).all(axis=1)]
    log.info(
        f"\n  Valid records scored all-1.0: {len(all_ones_valid):,} "
        f"({len(all_ones_valid)/len(valid)*100:.1f}% of valid)"
    )

    # ------------------------------------------------------------------
    # 9. Error / malformed-JSON records
    # ------------------------------------------------------------------
    errors = merged[merged["label_error"].notna()]
    _section(f"LABEL ERRORS ({len(errors):,} records)")
    if len(errors):
        log.info("\n" + errors[["record_id", "platform", "label_error"]].to_string(index=False))
    else:
        log.info("\n  None — all records labeled successfully.")

    # ------------------------------------------------------------------
    # 10. Summary verdict
    # ------------------------------------------------------------------
    _section("SUMMARY")
    n_valid = len(valid)
    n_total = len(merged)
    usable_pct = n_valid / n_total * 100

    log.info(f"\n  Total labeled:          {n_total:,}")
    log.info(f"  Valid (usable):         {n_valid:,}  ({usable_pct:.1f}%)")
    log.info(f"  Invalid triples:        {len(invalid):,}  ({100-usable_pct:.1f}%)")
    log.info(f"  Label errors:           {len(errors):,}")
    log.info(f"  Avg confidence (valid): {valid['confidence'].mean():.3f}")

    log.info("\n  Score ranges (valid):")
    for dim in DIMENSIONS:
        log.info(
            f"    {dim:18s}  mean={valid[dim].mean():.2f}  "
            f"std={valid[dim].std():.2f}"
        )

    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()
