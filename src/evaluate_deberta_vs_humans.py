"""
src/evaluate_deberta_vs_humans.py
==================================
Compare DeBERTa predictions to human consensus scores on the test set.

This is the PRIMARY validation of the trained model: ICC(2,1) between DeBERTa
outputs and human rater consensus, broken down by dimension and platform.

How the test-set human scores are obtained
------------------------------------------
The 340 human-annotated records are a stratified subset of the Phase 1 sample.
They are distributed across the train/val/test splits (and the Claude-DeBERTa
holdout) by record_id.  After process_human_annotations.py produces the
consensus scores (human_train.parquet + human_val.parquet), this script joins
those consensus scores with test.parquet on record_id to identify the test-set
human-annotated records.  Those records — typically ~30-40 of the 340 — are
the evaluation gold standard for this script.

Usage
-----
  python src/evaluate_deberta_vs_humans.py \\
      --predictions output/models/deberta-v3-large/stage3/test_predictions.csv \\
      --model-label deberta-v3-large

  python src/evaluate_deberta_vs_humans.py \\
      --predictions output/models/deberta-v3-base/stage3/test_predictions.csv \\
      --model-label deberta-v3-base \\
      --human-dir   output/human_labels \\
      --test-split  output/splits/test.parquet \\
      --output-dir  output/artifacts

Outputs
-------
  output/artifacts/human_{model_label}_eval_report.txt
  output/artifacts/human_{model_label}_eval_report.json

Metrics reported
----------------
  ICC(2,1)       two-way random effects, single rater, absolute agreement
  Pearson r      linear correlation
  Spearman rho   rank-order correlation
  MAE            mean absolute error on 1-5 scale
  Bias           mean(Human - DeBERTa); positive = humans score higher
  95% LoA        Bland-Altman limits of agreement
  Per-platform breakdown
  Per-intervention-type breakdown (from test.parquet)

ICC interpretation benchmarks (Koo & Mae, 2016)
  < 0.50 poor  |  0.50-0.75 moderate  |  0.75-0.90 good  |  > 0.90 excellent
  Target: >= 0.75 per dimension
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
SPLITS_DIR    = ROOT / "output" / "splits"
HUMAN_DIR     = ROOT / "output" / "human_labels"
ARTIFACTS_DIR = ROOT / "output" / "artifacts"
LOG_DIR       = ROOT / "logs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "evaluate_deberta_vs_humans.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]

ICC_BENCHMARKS   = "<0.50 poor | 0.50-0.75 moderate | 0.75-0.90 good | >0.90 excellent"


# ---------------------------------------------------------------------------
# Metrics (shared with evaluate_claude_deberta_agreement.py)
# ---------------------------------------------------------------------------

def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def icc_2_1(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single rater, absolute agreement."""
    a, b = _clean_pair(
        np.asarray(rater1, dtype=float), np.asarray(rater2, dtype=float)
    )
    n, k = len(a), 2
    if n < 3:
        return float("nan")

    data        = np.column_stack([a, b])
    grand_mean  = data.mean()
    row_means   = data.mean(axis=1)
    col_means   = data.mean(axis=0)

    ss_subjects = k * np.sum((row_means - grand_mean) ** 2)
    ss_raters   = n * np.sum((col_means  - grand_mean) ** 2)
    ss_total    = np.sum((data - grand_mean) ** 2)
    ss_error    = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters   = ss_raters   / (k - 1)
    ms_error    = ss_error    / ((n - 1) * (k - 1))

    denom = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    return float("nan") if denom == 0 else float((ms_subjects - ms_error) / denom)


def dimension_metrics(human: pd.Series, deberta: pd.Series) -> dict:
    """Full agreement metrics for one dimension."""
    a, b = _clean_pair(
        human.to_numpy(dtype=float), deberta.to_numpy(dtype=float)
    )
    n = len(a)
    if n < 3:
        return {"n": n, "error": "insufficient data (need >= 3 records)"}

    diffs    = a - b    # Human - DeBERTa
    bias     = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    return {
        "n":           n,
        "icc_2_1":     round(icc_2_1(a, b), 4),
        "pearson_r":   round(float(pd.Series(a).corr(pd.Series(b), method="pearson")),  4),
        "spearman_rho":round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 4),
        "mae":         round(float(np.mean(np.abs(a - b))), 4),
        "bias_human_minus_deberta": round(bias, 4),
        "std_diff":    round(std_diff, 4),
        "loa_lower":   round(bias - 1.96 * std_diff, 4),
        "loa_upper":   round(bias + 1.96 * std_diff, 4),
        "human_mean":  round(float(np.mean(a)), 4),
        "deberta_mean":round(float(np.mean(b)), 4),
        "human_std":   round(float(np.std(a, ddof=1)), 4),
        "deberta_std": round(float(np.std(b, ddof=1)), 4),
    }


def group_breakdown(df: pd.DataFrame, group_col: str) -> dict:
    """Per-group MAE and bias for each dimension."""
    result = {}
    for val, grp in df.groupby(group_col):
        result[str(val)] = {}
        for dim in DIMENSIONS:
            a, b = _clean_pair(
                grp[f"human_{dim}"].to_numpy(dtype=float),
                grp[f"deberta_{dim}"].to_numpy(dtype=float),
            )
            if len(a) < 2:
                result[str(val)][dim] = {"n": len(a), "error": "insufficient data"}
            else:
                result[str(val)][dim] = {
                    "n":    len(a),
                    "mae":  round(float(np.mean(np.abs(a - b))), 4),
                    "bias": round(float(np.mean(a - b)), 4),
                    "icc_2_1": round(icc_2_1(a, b), 4),
                }
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _icc_label(v: float) -> str:
    if np.isnan(v):  return "(insufficient data)"
    if v < 0.50:     return "poor"
    if v < 0.75:     return "moderate"
    if v < 0.90:     return "good"
    return                  "excellent"


def _dim_row(dim: str, m: dict) -> str:
    if "error" in m:
        return f"  {dim:20s}  {m['n']:>4d}  {'(insufficient data)':>50s}"
    return (
        f"  {dim:20s}  {m['n']:>4d}  {m['icc_2_1']:>8.3f}  "
        f"{m['pearson_r']:>7.3f}  {m['spearman_rho']:>8.3f}  "
        f"{m['mae']:>6.3f}  {m['bias_human_minus_deberta']:>+7.3f}  "
        f"[{m['loa_lower']:+.3f}, {m['loa_upper']:+.3f}]  "
        f"{_icc_label(m['icc_2_1'])}"
        + (" ✓" if m['icc_2_1'] >= 0.75 else "")
    )


def format_report(results: dict) -> str:
    W = 82
    model = results.get("model_label", "DeBERTa")
    lines = [
        "=" * W,
        f"  DeBERTa vs Human Consensus — Evaluation Report",
        f"  Model: {model}",
        "=" * W,
        f"  Test records (with human scores): {results['n_matched']:,}  "
        f"(of {results['n_test']:,} test, {results['n_human']:,} human-annotated)",
        f"  Human consensus source: {results['human_source']}",
        f"  Predictions file:       {results['predictions_path']}",
        "",
        "  Per-Dimension Agreement (Human vs DeBERTa)",
        "  " + "-" * (W - 2),
        f"  {'Dimension':20s}  {'n':>4s}  {'ICC(2,1)':>8s}  {'Pearson':>7s}  "
        f"{'Spearman':>8s}  {'MAE':>6s}  {'Bias':>7s}  {'95% LoA':>16s}",
        "  " + "-" * (W - 2),
    ]

    n_met = 0
    for dim in DIMENSIONS:
        lines.append(_dim_row(dim, results["dimensions"][dim]))
        m = results["dimensions"][dim]
        if "icc_2_1" in m and m["icc_2_1"] >= 0.75:
            n_met += 1

    target = "✓ ALL DIMENSIONS MET" if n_met == 4 else f"✗ {n_met}/4 dimensions met"
    lines += [
        "  " + "-" * (W - 2),
        f"  Target ICC >= 0.75:  {target}",
        "",
        f"  Bias = mean(Human - {model}).  Positive = humans score higher.",
        f"  LoA  = 95% limits of agreement (Bland-Altman).",
        f"  ICC  {ICC_BENCHMARKS}",
        "",
        "  Score Distributions",
        "  " + "-" * (W - 2),
        f"  {'Dimension':20s}  {'Human μ':>8s}  {'Human σ':>7s}  "
        f"{'DeBERTa μ':>9s}  {'DeBERTa σ':>9s}",
        "  " + "-" * (W - 2),
    ]
    for dim in DIMENSIONS:
        m = results["dimensions"][dim]
        if "error" not in m:
            lines.append(
                f"  {dim:20s}  {m['human_mean']:>8.3f}  {m['human_std']:>7.3f}  "
                f"{m['deberta_mean']:>9.3f}  {m['deberta_std']:>9.3f}"
            )

    if results.get("by_platform"):
        lines += ["", "  Per-Platform ICC(2,1) (averaged across dimensions)",
                  "  " + "-" * (W - 2)]
        for platform, dim_data in results["by_platform"].items():
            iccs = [v["icc_2_1"] for v in dim_data.values() if "icc_2_1" in v]
            mean_icc = np.mean(iccs) if iccs else float("nan")
            maes = [v["mae"] for v in dim_data.values() if "mae" in v]
            mean_mae = np.mean(maes) if maes else float("nan")
            lines.append(
                f"  {platform:20s}  mean ICC = {mean_icc:.3f}  mean MAE = {mean_mae:.3f}"
            )

    if results.get("by_intervention_type"):
        lines += ["", "  Per-Intervention-Type ICC(2,1) (averaged across dimensions)",
                  "  " + "-" * (W - 2)]
        for itype, dim_data in results["by_intervention_type"].items():
            iccs = [v["icc_2_1"] for v in dim_data.values() if "icc_2_1" in v]
            mean_icc = np.mean(iccs) if iccs else float("nan")
            lines.append(f"  {itype:25s}  mean ICC = {mean_icc:.3f}")

    lines += ["", "=" * W]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DeBERTa predictions to human consensus scores on the test set."
    )
    parser.add_argument("--predictions", required=True,
                        help="Predictions CSV from score_with_deberta.py.  "
                             "Required columns: record_id + 4 dimension scores.")
    parser.add_argument("--model-label", default="deberta",
                        help="Short model label for filenames and report headers "
                             "(e.g. 'deberta-v3-large').  Default: 'deberta'.")
    parser.add_argument("--human-dir", default=str(HUMAN_DIR),
                        help=f"Directory containing human_train.parquet and human_val.parquet "
                             f"(default: {HUMAN_DIR}).")
    parser.add_argument("--test-split", default=str(SPLITS_DIR / "test.parquet"),
                        help=f"Phase 1 test split parquet (default: output/splits/test.parquet).")
    parser.add_argument("--output-dir", default=str(ARTIFACTS_DIR),
                        help=f"Output directory (default: {ARTIFACTS_DIR}).")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    human_dir  = Path(args.human_dir)
    pred_path  = Path(args.predictions)
    test_path  = Path(args.test_split)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Evaluate DeBERTa vs Human Consensus ===")

    # ------------------------------------------------------------------
    # Load human consensus scores (combine train + val → all 340 records)
    # ------------------------------------------------------------------
    human_parts = []
    for fname in ["human_train.parquet", "human_val.parquet"]:
        fpath = human_dir / fname
        if fpath.exists():
            human_parts.append(pd.read_parquet(fpath))
        else:
            log.warning(f"Human consensus file not found: {fpath}")

    if not human_parts:
        log.error(f"No human consensus files found in {human_dir}")
        log.error("Run src/process_human_annotations.py first.")
        sys.exit(1)

    df_human = pd.concat(human_parts, ignore_index=True)
    log.info(f"Human consensus: {len(df_human):,} records from {human_dir.name}/")

    # ------------------------------------------------------------------
    # Load test split (to identify which records are in the test set)
    # ------------------------------------------------------------------
    if not test_path.exists():
        log.error(f"Test split not found: {test_path}")
        sys.exit(1)
    df_test = pd.read_parquet(test_path)
    log.info(f"Test split: {len(df_test):,} records")

    # Find human-annotated records that are in the test split
    test_ids  = set(df_test["record_id"])
    human_ids = set(df_human["record_id"])
    eval_ids  = test_ids & human_ids
    log.info(
        f"Human-annotated records in test split: {len(eval_ids):,}  "
        f"({len(eval_ids)/len(test_ids)*100:.1f}% of test, "
        f"{len(eval_ids)/len(human_ids)*100:.1f}% of human-annotated)"
    )

    if len(eval_ids) < 5:
        log.warning(
            f"Only {len(eval_ids)} evaluation records — results may not be reliable.  "
            "This is expected if the test split is small relative to the annotation set."
        )

    # Subset human consensus to eval records
    df_eval_human = df_human[df_human["record_id"].isin(eval_ids)].copy()

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    if not pred_path.exists():
        log.error(f"Predictions file not found: {pred_path}")
        log.error("Run src/score_with_deberta.py first.")
        sys.exit(1)

    df_pred = (
        pd.read_parquet(pred_path)
        if pred_path.suffix.lower() == ".parquet"
        else pd.read_csv(pred_path)
    )
    log.info(f"Predictions: {len(df_pred):,} records from {pred_path.name}")

    required_cols = {"record_id"} | set(DIMENSIONS)
    missing = required_cols - set(df_pred.columns)
    if missing:
        log.error(f"Predictions file missing columns: {missing}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Three-way join: test ∩ human ∩ predictions
    # ------------------------------------------------------------------
    rename_human   = {dim: f"human_{dim}"   for dim in DIMENSIONS}
    rename_deberta = {dim: f"deberta_{dim}" for dim in DIMENSIONS}

    df_h = df_eval_human[["record_id"] + DIMENSIONS].rename(columns=rename_human)
    df_p = df_pred[["record_id"] + DIMENSIONS].rename(columns=rename_deberta)

    # Also bring in platform + intervention_type from test.parquet for breakdowns
    test_meta = df_test[["record_id", "platform", "intervention_type"]].drop_duplicates("record_id")
    df_merged = df_h.merge(df_p, on="record_id", how="inner")
    df_merged = df_merged.merge(test_meta, on="record_id", how="left")

    n_matched = len(df_merged)
    log.info(f"Matched records for evaluation: {n_matched:,}")

    if n_matched < 3:
        log.error("Fewer than 3 matched records — cannot compute ICC.  "
                  "Check that record_ids align across test split, human consensus, and predictions.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Per-dimension metrics
    # ------------------------------------------------------------------
    dim_results: dict = {}
    for dim in DIMENSIONS:
        m = dimension_metrics(df_merged[f"human_{dim}"], df_merged[f"deberta_{dim}"])
        dim_results[dim] = m
        if "error" not in m:
            log.info(
                f"  {dim:20s}  ICC={m['icc_2_1']:.3f}  "
                f"Pearson={m['pearson_r']:.3f}  MAE={m['mae']:.3f}  "
                f"Bias={m['bias_human_minus_deberta']:+.3f}"
            )
        else:
            log.warning(f"  {dim}: {m['error']}")

    # ------------------------------------------------------------------
    # Group breakdowns
    # ------------------------------------------------------------------
    by_platform = (
        group_breakdown(df_merged, "platform")
        if "platform" in df_merged.columns and df_merged["platform"].notna().any()
        else {}
    )
    by_itype = (
        group_breakdown(df_merged, "intervention_type")
        if "intervention_type" in df_merged.columns and df_merged["intervention_type"].notna().any()
        else {}
    )

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    file_label = args.model_label.replace("/", "-").replace(" ", "_")

    results = {
        "model_label":        args.model_label,
        "n_matched":          n_matched,
        "n_test":             len(df_test),
        "n_human":            len(df_human),
        "human_source":       str(human_dir),
        "predictions_path":   str(pred_path),
        "dimensions":         dim_results,
        "by_platform":        by_platform,
        "by_intervention_type": by_itype,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    report = format_report(results)
    txt_path  = out_dir / f"human_{file_label}_eval_report.txt"
    json_path = out_dir / f"human_{file_label}_eval_report.json"

    with open(txt_path,  "w", encoding="utf-8") as f:
        f.write(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"Report (text) → {txt_path}")
    log.info(f"Report (JSON) → {json_path}")
    print("\n" + report)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
