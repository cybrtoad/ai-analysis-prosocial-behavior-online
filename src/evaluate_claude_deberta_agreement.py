"""
Post-Training Validation: Claude-DeBERTa Agreement
====================================================
Computes agreement between Claude's original scores and DeBERTa's predictions
on the held-out Claude-DeBERTa agreement set.

Run this AFTER completing all three DeBERTa training stages, not before.
The holdout set (output/splits/claude_agreement_holdout.parquet) was excluded
from all training stages precisely for this comparison.

Usage:
  python src/evaluate_claude_deberta_agreement.py \\
      --predictions output/artifacts/deberta_holdout_predictions.csv

  python src/evaluate_claude_deberta_agreement.py \\
      --predictions output/artifacts/deberta_holdout_predictions.parquet \\
      --holdout output/splits/claude_agreement_holdout.parquet \\
      --output-dir output/artifacts

Predictions file format (CSV or parquet):
  record_id, empathy, constructiveness, respect, social_cohesion
  One row per record in the holdout set.

Metrics reported per dimension:
  ICC(2,1)    Intraclass Correlation Coefficient — two-way random effects,
              single rater, absolute agreement.  Matches the metric used for
              human inter-rater reliability, so results are directly comparable.
  Pearson r   Linear correlation
  Spearman ρ  Rank-order correlation
  MAE         Mean Absolute Error (1–5 scale)
  Bias        Mean of (Claude − DeBERTa).  Positive = Claude scores higher.
  LoA lower   Lower 95% limit of agreement = bias − 1.96 × SD(differences)
  LoA upper   Upper 95% limit of agreement = bias + 1.96 × SD(differences)

Also reported:
  - Score distribution comparison (mean, std) per dimension
  - Per-platform and per-intervention-type MAE/bias breakdowns

ICC interpretation benchmarks (Koo & Mae, 2016):
  < 0.50 = poor   0.50–0.75 = moderate   0.75–0.90 = good   > 0.90 = excellent

Why this holdout exists:
  The three DeBERTa training stages pull the model progressively toward human
  judgment, which may move it away from Claude's scores in places where Claude
  and human raters disagreed.  The holdout lets us quantify the net Claude-
  DeBERTa divergence after all training is complete and decompose it by
  dimension and platform.  Systematic divergence signals where Claude's
  automated scoring differs from the human-calibrated model.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
SPLITS_DIR   = ROOT / "output" / "splits"
ARTIFACTS_DIR = ROOT / "output" / "artifacts"
LOG_DIR      = ROOT / "logs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "evaluate_claude_deberta_agreement.log", encoding="utf-8"
        ),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _clean_pair(
    rater1: np.ndarray, rater2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either value is NaN and return cleaned pair."""
    r1 = np.asarray(rater1, dtype=float)
    r2 = np.asarray(rater2, dtype=float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    return r1[mask], r2[mask]


def icc_2_1(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects model, single rater, absolute agreement.

    Uses the standard two-way ANOVA decomposition:
      ICC = (MS_subjects - MS_error) /
            (MS_subjects + (k-1)*MS_error + k*(MS_raters - MS_error)/n)

    Returns NaN if computation is degenerate (zero variance, too few records).
    """
    c, d = _clean_pair(rater1, rater2)
    n = len(c)
    if n < 3:
        return float("nan")

    k = 2
    data = np.column_stack([c, d])           # n × 2

    grand_mean  = data.mean()
    row_means   = data.mean(axis=1)          # subject means
    col_means   = data.mean(axis=0)          # rater means

    ss_subjects = k * np.sum((row_means - grand_mean) ** 2)
    ss_raters   = n * np.sum((col_means  - grand_mean) ** 2)
    ss_total    = np.sum((data - grand_mean) ** 2)
    ss_error    = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters   = ss_raters   / (k - 1)
    ms_error    = ss_error    / ((n - 1) * (k - 1))

    denominator = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if denominator == 0:
        return float("nan")

    return float((ms_subjects - ms_error) / denominator)


def bland_altman_stats(rater1: np.ndarray, rater2: np.ndarray) -> dict:
    """
    Bland-Altman agreement statistics.
    Difference convention: Claude − DeBERTa (positive = Claude scores higher).
    """
    c, d = _clean_pair(rater1, rater2)
    diffs     = c - d
    bias      = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))
    loa_lower = bias - 1.96 * std_diff
    loa_upper = bias + 1.96 * std_diff
    return {
        "bias":      round(bias,      4),
        "std_diff":  round(std_diff,  4),
        "loa_lower": round(loa_lower, 4),
        "loa_upper": round(loa_upper, 4),
    }


def dimension_metrics(
    claude_scores: pd.Series, deberta_scores: pd.Series
) -> dict:
    """Compute all agreement metrics for one dimension."""
    c, d = _clean_pair(
        claude_scores.to_numpy(dtype=float),
        deberta_scores.to_numpy(dtype=float),
    )
    n = len(c)
    if n < 3:
        return {"n": n, "error": "insufficient data (need ≥ 3 records)"}

    ba = bland_altman_stats(c, d)

    return {
        "n":                        n,
        "icc_2_1":                  round(icc_2_1(c, d), 4),
        "pearson_r":                round(float(pd.Series(c).corr(pd.Series(d), method="pearson")),  4),
        "spearman_rho":             round(float(pd.Series(c).corr(pd.Series(d), method="spearman")), 4),
        "mae":                      round(float(np.mean(np.abs(c - d))), 4),
        "bias_claude_minus_deberta": ba["bias"],
        "std_diff":                 ba["std_diff"],
        "loa_lower":                ba["loa_lower"],
        "loa_upper":                ba["loa_upper"],
        "claude_mean":              round(float(np.mean(c)), 4),
        "deberta_mean":             round(float(np.mean(d)), 4),
        "claude_std":               round(float(np.std(c, ddof=1)), 4),
        "deberta_std":              round(float(np.std(d, ddof=1)), 4),
    }


def group_breakdown(df: pd.DataFrame, group_col: str) -> dict:
    """Per-group MAE and bias for each dimension."""
    result = {}
    for group_val, grp in df.groupby(group_col):
        result[str(group_val)] = {}
        for dim in DIMENSIONS:
            c, d = _clean_pair(
                grp[f"claude_{dim}"].to_numpy(dtype=float),
                grp[f"deberta_{dim}"].to_numpy(dtype=float),
            )
            if len(c) < 2:
                result[str(group_val)][dim] = {"n": len(c), "error": "insufficient data"}
            else:
                result[str(group_val)][dim] = {
                    "n":    len(c),
                    "mae":  round(float(np.mean(np.abs(c - d))), 4),
                    "bias": round(float(np.mean(c - d)), 4),
                }
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

ICC_BENCHMARKS = "<0.50 poor | 0.50–0.75 moderate | 0.75–0.90 good | >0.90 excellent"


def _dim_row(dim: str, m: dict) -> str:
    if "error" in m:
        return f"  {dim:20s}  {m['n']:>5d}  {'(insufficient data)':>50s}"
    return (
        f"  {dim:20s}  {m['n']:>5d}  {m['icc_2_1']:>8.3f}  "
        f"{m['pearson_r']:>7.3f}  {m['spearman_rho']:>8.3f}  "
        f"{m['mae']:>6.3f}  {m['bias_claude_minus_deberta']:>+7.3f}  "
        f"[{m['loa_lower']:+.3f}, {m['loa_upper']:+.3f}]"
    )


def format_report(results: dict) -> str:
    w = 78
    model_label = results.get("model_label", "DeBERTa")
    lines = [
        "=" * w,
        f"  Claude-{model_label} Agreement Report",
        "=" * w,
        f"  Model:                   {model_label}",
        f"  Holdout records matched: {results['n_matched']:,} "
        f"(of {results['n_holdout']:,} in holdout, "
        f"{results['n_predictions']:,} in predictions)",
        f"  Holdout file:            {results['holdout_path']}",
        f"  Predictions file:        {results['predictions_path']}",
        "",
        "  Per-Dimension Agreement",
        "  " + "-" * (w - 2),
        f"  {'Dimension':20s}  {'n':>5s}  {'ICC(2,1)':>8s}  {'Pearson':>7s}  "
        f"{'Spearman':>8s}  {'MAE':>6s}  {'Bias':>7s}  {'95% LoA':>16s}",
        "  " + "-" * (w - 2),
    ]

    for dim in DIMENSIONS:
        lines.append(_dim_row(dim, results["dimensions"][dim]))

    lines += [
        "",
        f"  Bias = mean(Claude − {model_label}).  Positive bias → Claude scores higher.",
        f"  LoA  = 95% limits of agreement (Bland-Altman).  Narrow LoA → consistent.",
        f"  ICC  {ICC_BENCHMARKS}",
        "",
        "  Score Distributions",
        "  " + "-" * (w - 2),
        f"  {'Dimension':20s}  {'Claude μ':>8s}  {'Claude σ':>8s}  "
        f"{'DeBERTa μ':>9s}  {'DeBERTa σ':>9s}",
        "  " + "-" * (w - 2),
    ]

    for dim in DIMENSIONS:
        m = results["dimensions"][dim]
        if "error" not in m:
            lines.append(
                f"  {dim:20s}  {m['claude_mean']:>8.3f}  {m['claude_std']:>8.3f}  "
                f"{m['deberta_mean']:>9.3f}  {m['deberta_std']:>9.3f}"
            )

    if results.get("by_platform"):
        lines += ["", "  Per-Platform Mean MAE (averaged across dimensions)", "  " + "-" * (w - 2)]
        for platform, dim_data in results["by_platform"].items():
            maes = [v["mae"] for v in dim_data.values() if "mae" in v]
            mean_mae = np.mean(maes) if maes else float("nan")
            lines.append(f"  {platform:20s}  MAE = {mean_mae:.3f}")

    if results.get("by_intervention_type"):
        lines += ["", "  Per-Intervention-Type Mean MAE (averaged across dimensions)", "  " + "-" * (w - 2)]
        for itype, dim_data in results["by_intervention_type"].items():
            maes = [v["mae"] for v in dim_data.values() if "mae" in v]
            mean_mae = np.mean(maes) if maes else float("nan")
            lines.append(f"  {itype:25s}  MAE = {mean_mae:.3f}")

    lines += ["", "=" * w]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Claude-DeBERTa agreement on the held-out agreement set"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "Path to DeBERTa predictions file (CSV or parquet).  "
            "Required columns: record_id, empathy, constructiveness, "
            "respect, social_cohesion."
        ),
    )
    parser.add_argument(
        "--holdout",
        default=str(SPLITS_DIR / "claude_agreement_holdout.parquet"),
        help="Path to holdout parquet (default: output/splits/claude_agreement_holdout.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACTS_DIR),
        help="Directory for output report files (default: output/artifacts/)",
    )
    parser.add_argument(
        "--model-label",
        default="deberta",
        help=(
            "Short label for the model being evaluated, used in report headers and "
            "output filenames (e.g. 'deberta-v3-large', 'deberta-v3-base'). "
            "Default: 'deberta'."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Claude-DeBERTa Agreement Evaluation ===")

    # Load holdout (Claude scores)
    holdout_path = Path(args.holdout)
    if not holdout_path.exists():
        log.error(f"Holdout file not found: {holdout_path}")
        log.error("Run src/split.py first to generate the holdout set.")
        sys.exit(1)

    df_holdout = pd.read_parquet(holdout_path)
    log.info(f"Loaded holdout: {len(df_holdout):,} records from {holdout_path.name}")

    # Rename Claude dimension columns to avoid collision after merge
    rename_claude = {dim: f"claude_{dim}" for dim in DIMENSIONS}
    df_holdout = df_holdout.rename(columns=rename_claude)

    # Load DeBERTa predictions
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        log.error(f"Predictions file not found: {pred_path}")
        sys.exit(1)

    df_pred = (
        pd.read_parquet(pred_path)
        if pred_path.suffix.lower() == ".parquet"
        else pd.read_csv(pred_path)
    )
    log.info(f"Loaded predictions: {len(df_pred):,} records from {pred_path.name}")

    # Validate prediction columns
    required_cols = {"record_id"} | set(DIMENSIONS)
    missing = required_cols - set(df_pred.columns)
    if missing:
        log.error(f"Predictions file is missing required columns: {missing}")
        log.error(f"Expected: record_id + {DIMENSIONS}")
        sys.exit(1)

    # Rename DeBERTa prediction columns and merge
    rename_deberta = {dim: f"deberta_{dim}" for dim in DIMENSIONS}
    df_pred_slim = df_pred[["record_id"] + DIMENSIONS].rename(columns=rename_deberta)
    df = df_holdout.merge(df_pred_slim, on="record_id", how="inner")

    n_matched  = len(df)
    n_holdout  = len(df_holdout)
    n_pred     = len(df_pred)

    log.info(
        f"Matched {n_matched:,} records "
        f"({n_matched / n_holdout * 100:.1f}% of holdout, "
        f"{n_matched / n_pred * 100:.1f}% of predictions)"
    )
    if n_matched < 10:
        log.warning("Fewer than 10 matched records — results may not be meaningful.")

    # Per-dimension overall metrics
    dim_results: dict = {}
    for dim in DIMENSIONS:
        m = dimension_metrics(df[f"claude_{dim}"], df[f"deberta_{dim}"])
        dim_results[dim] = m
        if "error" not in m:
            log.info(
                f"  {dim:20s}  ICC={m['icc_2_1']:.3f}  "
                f"Pearson={m['pearson_r']:.3f}  "
                f"MAE={m['mae']:.3f}  "
                f"Bias={m['bias_claude_minus_deberta']:+.3f}"
            )
        else:
            log.warning(f"  {dim}: {m['error']}")

    # Per-platform breakdown
    by_platform = group_breakdown(df, "platform") if "platform" in df.columns else {}

    # Per-intervention-type breakdown
    by_itype = (
        group_breakdown(df, "intervention_type")
        if "intervention_type" in df.columns
        else {}
    )

    # Sanitise label for use in filenames (replace spaces and slashes)
    file_label = args.model_label.replace("/", "-").replace(" ", "_")

    results = {
        "model_label":         args.model_label,
        "n_matched":           n_matched,
        "n_holdout":           n_holdout,
        "n_predictions":       n_pred,
        "holdout_path":        str(holdout_path),
        "predictions_path":    str(pred_path),
        "dimensions":          dim_results,
        "by_platform":         by_platform,
        "by_intervention_type": by_itype,
    }

    # Save JSON
    json_path = out_dir / f"claude_{file_label}_agreement_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved JSON report → {json_path}")

    # Save and print text report
    report = format_report(results)
    txt_path = out_dir / f"claude_{file_label}_agreement_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Saved text report → {txt_path}")

    print("\n" + report)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
