"""
src/compare_claude_vs_humans.py
================================
Compare Claude's automated scores to human consensus scores for the 340
annotated records.  This is the primary validation of Claude's labeling
quality before (or after) scaling to ~100K Phase 2 records.

How it works
------------
Claude scored the 340 annotation records and saved those scores to
human-validation/claude_scores_EMBARGOED.csv (held back from annotators
to prevent anchoring).  After annotation is complete and
process_human_annotations.py has produced consensus scores, this script
joins the two on record_id and computes agreement metrics.

Usage
-----
  python src/compare_claude_vs_humans.py

  python src/compare_claude_vs_humans.py \\
      --claude-scores human-validation/claude_scores_EMBARGOED.csv \\
      --human-dir     output/human_labels \\
      --output-dir    output/artifacts

Outputs
-------
  output/artifacts/claude_human_validation_report.txt
  output/artifacts/claude_human_validation_report.json

Metrics reported
----------------
  ICC(2,1)       two-way random effects, single rater, absolute agreement
                 Target: >= 0.70 (slightly lower threshold than DeBERTa, since
                 Claude is the scoring instrument, not the final model)
  Pearson r      linear correlation
  Spearman rho   rank-order correlation
  MAE            mean absolute error on 1-5 scale
  Bias           mean(Claude - Human); positive = Claude scores higher
  95% LoA        Bland-Altman limits of agreement
  % within 0.5   proportion of records where Claude and human agree within 0.5
  % within 1.0   proportion within 1.0 point
  By-platform and by-intervention-type breakdowns

Calibration verdict
-------------------
  GOOD (ICC >= 0.70 on all dimensions):
    Claude's automated scoring is well-calibrated.  Phase 2 scores can be
    used with confidence.  Report the ICC as a validation metric.

  ACCEPTABLE (ICC >= 0.60 on all, >= 0.70 on mean):
    Proceed with Phase 2, but note the calibration gap as a limitation.
    Consider weighting human-annotated records more heavily in analysis.

  POOR (ICC < 0.60 on any dimension):
    Systematic divergence detected.  If Phase 2 has not yet completed:
    revise the labeling prompt and re-run.  If Phase 2 is already complete:
    document the specific dimensions and conditions where Claude diverges,
    and treat those results with additional caution.

Note: Since Phase 2 scoring may already be in progress by the time this
script runs, the verdict is informational rather than blocking.  Systematic
Claude-human divergence is documented as a methodological finding.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
EMBARGOED     = ROOT / "human-validation" / "claude_scores_EMBARGOED.csv"
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
        logging.FileHandler(LOG_DIR / "compare_claude_vs_humans.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]

# Calibration thresholds
ICC_GOOD       = 0.70
ICC_ACCEPTABLE = 0.60

ICC_BENCHMARKS = "<0.50 poor | 0.50-0.75 moderate | 0.75-0.90 good | >0.90 excellent"


# ---------------------------------------------------------------------------
# Metrics (same formulas as other evaluation scripts)
# ---------------------------------------------------------------------------

def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def icc_2_1(rater1: np.ndarray, rater2: np.ndarray) -> float:
    a, b = _clean_pair(
        np.asarray(rater1, dtype=float), np.asarray(rater2, dtype=float)
    )
    n, k = len(a), 2
    if n < 3:
        return float("nan")
    data       = np.column_stack([a, b])
    grand_mean = data.mean()
    row_means  = data.mean(axis=1)
    col_means  = data.mean(axis=0)
    ss_s = k * np.sum((row_means - grand_mean) ** 2)
    ss_r = n * np.sum((col_means  - grand_mean) ** 2)
    ss_e = np.sum((data - grand_mean) ** 2) - ss_s - ss_r
    ms_s = ss_s / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_s + (k - 1) * ms_e + k * (ms_r - ms_e) / n
    return float("nan") if denom == 0 else float((ms_s - ms_e) / denom)


def dimension_metrics(claude: pd.Series, human: pd.Series) -> dict:
    a, b = _clean_pair(
        claude.to_numpy(dtype=float), human.to_numpy(dtype=float)
    )
    n = len(a)
    if n < 3:
        return {"n": n, "error": "insufficient data (need >= 3 records)"}

    diffs    = a - b    # Claude - Human
    bias     = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    return {
        "n":                  n,
        "icc_2_1":            round(icc_2_1(a, b), 4),
        "pearson_r":          round(float(pd.Series(a).corr(pd.Series(b), method="pearson")),  4),
        "spearman_rho":       round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 4),
        "mae":                round(float(np.mean(np.abs(a - b))), 4),
        "pct_within_half":    round(float(np.mean(np.abs(a - b) <= 0.5)), 4),
        "pct_within_one":     round(float(np.mean(np.abs(a - b) <= 1.0)), 4),
        "bias_claude_minus_human": round(bias, 4),
        "std_diff":           round(std_diff, 4),
        "loa_lower":          round(bias - 1.96 * std_diff, 4),
        "loa_upper":          round(bias + 1.96 * std_diff, 4),
        "claude_mean":        round(float(np.mean(a)), 4),
        "human_mean":         round(float(np.mean(b)), 4),
        "claude_std":         round(float(np.std(a, ddof=1)), 4),
        "human_std":          round(float(np.std(b, ddof=1)), 4),
    }


def group_breakdown(df: pd.DataFrame, group_col: str) -> dict:
    result = {}
    for val, grp in df.groupby(group_col):
        result[str(val)] = {}
        for dim in DIMENSIONS:
            a, b = _clean_pair(
                grp[f"claude_{dim}"].to_numpy(dtype=float),
                grp[f"human_{dim}"].to_numpy(dtype=float),
            )
            if len(a) < 2:
                result[str(val)][dim] = {"n": len(a), "error": "insufficient data"}
            else:
                result[str(val)][dim] = {
                    "n":    len(a),
                    "icc_2_1": round(icc_2_1(a, b), 4),
                    "mae":  round(float(np.mean(np.abs(a - b))), 4),
                    "bias": round(float(np.mean(a - b)), 4),
                }
    return result


# ---------------------------------------------------------------------------
# Calibration verdict
# ---------------------------------------------------------------------------

def calibration_verdict(dim_results: dict) -> tuple[str, str]:
    """
    Returns (verdict_code, verdict_text).
    verdict_code: 'GOOD' | 'ACCEPTABLE' | 'POOR'
    """
    iccs = [m["icc_2_1"] for m in dim_results.values() if "icc_2_1" in m]
    if not iccs:
        return "UNKNOWN", "Insufficient data to assess calibration."

    mean_icc = float(np.mean(iccs))
    min_icc  = float(np.min(iccs))
    dims_below_good = [d for d, m in dim_results.items()
                       if m.get("icc_2_1", 0) < ICC_GOOD]
    dims_below_acc  = [d for d, m in dim_results.items()
                       if m.get("icc_2_1", 0) < ICC_ACCEPTABLE]

    if min_icc >= ICC_GOOD:
        code = "GOOD"
        text = (
            f"All dimensions meet ICC >= {ICC_GOOD} (mean ICC = {mean_icc:.3f}).  "
            "Claude's automated scoring is well-calibrated against human raters.  "
            "Phase 2 scores can be used with confidence.  "
            "Report the ICC values as a validation metric in the thesis."
        )
    elif not dims_below_acc and mean_icc >= ICC_GOOD:
        code = "ACCEPTABLE"
        text = (
            f"Mean ICC = {mean_icc:.3f} (>= {ICC_GOOD}), but dimensions "
            f"{dims_below_good} are below the {ICC_GOOD} threshold.  "
            "Proceed with Phase 2, noting the calibration gap as a limitation.  "
            "Consider documenting per-dimension uncertainty in the thesis."
        )
    elif dims_below_acc:
        code = "POOR"
        text = (
            f"Dimensions {dims_below_acc} have ICC < {ICC_ACCEPTABLE} — "
            "systematic divergence between Claude and human raters detected.  "
            "If Phase 2 scoring is not yet complete: revise the labeling prompt "
            "and re-run before proceeding.  "
            "If Phase 2 is already complete: document the affected dimensions "
            "explicitly as a methodological limitation and treat those results "
            "with additional caution in the analysis."
        )
    else:
        code = "ACCEPTABLE"
        text = (
            f"Mean ICC = {mean_icc:.3f}, minimum ICC = {min_icc:.3f}.  "
            "Borderline calibration — proceed with documented caveats."
        )

    return code, text


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _icc_label(v: float) -> str:
    if np.isnan(v):  return "(n/a)"
    if v < 0.50:     return "poor"
    if v < 0.75:     return "moderate"
    if v < 0.90:     return "good"
    return                  "excellent"


def format_report(results: dict) -> str:
    W = 82
    lines = [
        "=" * W,
        "  Claude vs Human Consensus — Calibration Validation Report",
        "  Prosocial Interventions in Online Moderation",
        "=" * W,
        f"  Matched records:    {results['n_matched']:,}  "
        f"(of {results['n_claude']:,} Claude, {results['n_human']:,} human-annotated)",
        f"  Claude scores:      {results['claude_path']}",
        f"  Human consensus:    {results['human_source']}",
        "",
        "  Per-Dimension Agreement",
        "  " + "-" * (W - 2),
        f"  {'Dimension':20s}  {'n':>4s}  {'ICC(2,1)':>8s}  {'Pearson':>7s}  "
        f"{'Spearman':>8s}  {'MAE':>5s}  {'Bias':>6s}  {'<=0.5':>6s}  {'<=1.0':>6s}",
        "  " + "-" * (W - 2),
    ]

    for dim in DIMENSIONS:
        m = results["dimensions"][dim]
        if "error" in m:
            lines.append(f"  {dim:20s}  {m['n']:>4d}  (insufficient data)")
            continue
        flag = " ✓" if m["icc_2_1"] >= ICC_GOOD else (
               " ~" if m["icc_2_1"] >= ICC_ACCEPTABLE else " ✗")
        lines.append(
            f"  {dim:20s}  {m['n']:>4d}  {m['icc_2_1']:>8.3f}{flag}  "
            f"{m['pearson_r']:>7.3f}  {m['spearman_rho']:>8.3f}  "
            f"{m['mae']:>5.3f}  {m['bias_claude_minus_human']:>+6.3f}  "
            f"{m['pct_within_half']*100:>5.1f}%  {m['pct_within_one']*100:>5.1f}%"
        )

    iccs = [m["icc_2_1"] for m in results["dimensions"].values() if "icc_2_1" in m]
    mean_icc = float(np.mean(iccs)) if iccs else float("nan")
    lines += [
        "  " + "-" * (W - 2),
        f"  Mean ICC: {mean_icc:.3f}   "
        f"Target >= {ICC_GOOD} (✓) / >= {ICC_ACCEPTABLE} (~) / < {ICC_ACCEPTABLE} (✗)",
        "",
        "  Bias = mean(Claude − Human).  Positive = Claude scores higher.",
        "  LoA = 95% limits of agreement (Bland-Altman).",
        f"  ICC benchmarks: {ICC_BENCHMARKS}",
        "",
    ]

    # Score distributions
    lines += [
        "  Score Distributions",
        "  " + "-" * (W - 2),
        f"  {'Dimension':20s}  {'Claude μ':>8s}  {'Claude σ':>8s}  "
        f"{'Human μ':>7s}  {'Human σ':>7s}  {'LoA':>18s}",
        "  " + "-" * (W - 2),
    ]
    for dim in DIMENSIONS:
        m = results["dimensions"][dim]
        if "error" not in m:
            lines.append(
                f"  {dim:20s}  {m['claude_mean']:>8.3f}  {m['claude_std']:>8.3f}  "
                f"{m['human_mean']:>7.3f}  {m['human_std']:>7.3f}  "
                f"[{m['loa_lower']:+.3f}, {m['loa_upper']:+.3f}]"
            )

    # Platform breakdown
    if results.get("by_platform"):
        lines += ["", "  Per-Platform Mean ICC (across dimensions)",
                  "  " + "-" * (W - 2)]
        for platform, dim_data in results["by_platform"].items():
            iccs_p = [v["icc_2_1"] for v in dim_data.values() if "icc_2_1" in v]
            maes_p = [v["mae"]     for v in dim_data.values() if "mae"     in v]
            mean_i = float(np.mean(iccs_p)) if iccs_p else float("nan")
            mean_m = float(np.mean(maes_p)) if maes_p else float("nan")
            lines.append(f"  {platform:20s}  mean ICC = {mean_i:.3f}  mean MAE = {mean_m:.3f}")

    # Intervention type breakdown
    if results.get("by_intervention_type"):
        lines += ["", "  Per-Intervention-Type Mean ICC (across dimensions)",
                  "  " + "-" * (W - 2)]
        for itype, dim_data in results["by_intervention_type"].items():
            iccs_t = [v["icc_2_1"] for v in dim_data.values() if "icc_2_1" in v]
            mean_i = float(np.mean(iccs_t)) if iccs_t else float("nan")
            lines.append(f"  {itype:25s}  mean ICC = {mean_i:.3f}")

    # Verdict
    code  = results.get("calibration_verdict_code", "UNKNOWN")
    text  = results.get("calibration_verdict_text", "")
    lines += [
        "",
        "  Calibration Verdict",
        "  " + "-" * (W - 2),
        f"  {code}",
        "",
    ]
    for line in text.splitlines():
        lines.append(f"  {line}")

    lines += ["", "=" * W]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Claude's automated scores to human consensus for the 340 "
            "annotated records.  Validates Claude calibration before/after Phase 2."
        )
    )
    parser.add_argument(
        "--claude-scores", default=str(EMBARGOED),
        help=f"Claude embargoed scores CSV (default: {EMBARGOED}).",
    )
    parser.add_argument(
        "--human-dir", default=str(HUMAN_DIR),
        help=f"Directory with human_train.parquet + human_val.parquet "
             f"(default: {HUMAN_DIR}).",
    )
    parser.add_argument(
        "--output-dir", default=str(ARTIFACTS_DIR),
        help=f"Output directory (default: {ARTIFACTS_DIR}).",
    )
    args = parser.parse_args()

    claude_path = Path(args.claude_scores)
    human_dir   = Path(args.human_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Claude vs Human Consensus Calibration Validation ===")

    # ------------------------------------------------------------------
    # Load Claude scores
    # ------------------------------------------------------------------
    if not claude_path.exists():
        log.error(f"Claude scores file not found: {claude_path}")
        sys.exit(1)

    df_claude = pd.read_csv(claude_path)
    df_claude.columns = [c.strip().lower() for c in df_claude.columns]

    for dim in DIMENSIONS:
        df_claude[dim] = pd.to_numeric(df_claude[dim], errors="coerce")

    # Drop invalid triples — Claude flagged these as not genuine intervention sequences
    if "invalid_triple" in df_claude.columns:
        n_before = len(df_claude)
        df_claude = df_claude[~df_claude["invalid_triple"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )].copy()
        log.info(
            f"Dropped {n_before - len(df_claude)} invalid triples from Claude scores"
        )

    log.info(f"Claude scores: {len(df_claude):,} valid records")

    # ------------------------------------------------------------------
    # Load human consensus (combine train + val → all annotated records)
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
    log.info(f"Human consensus: {len(df_human):,} records")

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    rename_claude = {dim: f"claude_{dim}" for dim in DIMENSIONS}
    rename_human  = {dim: f"human_{dim}"  for dim in DIMENSIONS}

    df_c = df_claude[["record_id"] + DIMENSIONS].rename(columns=rename_claude)
    df_h = df_human[["record_id"] + DIMENSIONS].rename(columns=rename_human)

    df = df_c.merge(df_h, on="record_id", how="inner")
    n_matched = len(df)
    log.info(f"Matched records: {n_matched:,}")

    if n_matched < 5:
        log.error(
            f"Only {n_matched} matched records.  "
            "Check that record_ids align between Claude scores and human consensus."
        )
        sys.exit(1)

    # Bring in platform + intervention_type for breakdowns
    # Try to get these from Claude scores file or human consensus
    for meta_col in ["platform", "intervention_type"]:
        if meta_col in df_claude.columns:
            df = df.merge(
                df_claude[["record_id", meta_col]].drop_duplicates("record_id"),
                on="record_id", how="left",
            )

    # If not in Claude scores, try to get from splits
    for meta_col in ["platform", "intervention_type"]:
        if meta_col not in df.columns:
            # Infer platform from record_id prefix
            if meta_col == "platform":
                df["platform"] = df["record_id"].str.split("_").str[0]
            # intervention_type: try gold_labels_merged.parquet
            elif meta_col == "intervention_type":
                merged_path = ROOT / "output" / "labels" / "gold_labels_merged.parquet"
                if merged_path.exists():
                    gold = pd.read_parquet(merged_path)[["record_id", "intervention_type"]]
                    df = df.merge(gold, on="record_id", how="left")
                else:
                    df["intervention_type"] = "unknown"

    # ------------------------------------------------------------------
    # Per-dimension metrics
    # ------------------------------------------------------------------
    dim_results: dict = {}
    for dim in DIMENSIONS:
        m = dimension_metrics(df[f"claude_{dim}"], df[f"human_{dim}"])
        dim_results[dim] = m
        if "error" not in m:
            log.info(
                f"  {dim:20s}  ICC={m['icc_2_1']:.3f}  "
                f"Pearson={m['pearson_r']:.3f}  MAE={m['mae']:.3f}  "
                f"Bias={m['bias_claude_minus_human']:+.3f}"
            )
        else:
            log.warning(f"  {dim}: {m['error']}")

    # ------------------------------------------------------------------
    # Group breakdowns
    # ------------------------------------------------------------------
    by_platform = (
        group_breakdown(df, "platform")
        if "platform" in df.columns and df["platform"].notna().any()
        else {}
    )
    by_itype = (
        group_breakdown(df, "intervention_type")
        if "intervention_type" in df.columns and df["intervention_type"].notna().any()
        else {}
    )

    # ------------------------------------------------------------------
    # Calibration verdict
    # ------------------------------------------------------------------
    verdict_code, verdict_text = calibration_verdict(dim_results)
    log.info(f"\nCalibration verdict: {verdict_code}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "n_matched":              n_matched,
        "n_claude":               len(df_claude),
        "n_human":                len(df_human),
        "claude_path":            str(claude_path),
        "human_source":           str(human_dir),
        "dimensions":             dim_results,
        "by_platform":            by_platform,
        "by_intervention_type":   by_itype,
        "calibration_verdict_code": verdict_code,
        "calibration_verdict_text": verdict_text,
    }

    report = format_report(results)
    txt_path  = out_dir / "claude_human_validation_report.txt"
    json_path = out_dir / "claude_human_validation_report.json"

    with open(txt_path,  "w", encoding="utf-8") as f:
        f.write(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nReport (text) → {txt_path}")
    log.info(f"Report (JSON) → {json_path}")
    print("\n" + report)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
