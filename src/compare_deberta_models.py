"""
src/compare_deberta_models.py
==============================
Side-by-side comparison of DeBERTa-v3-large vs DeBERTa-v3-base across all
evaluation types.  Answers the thesis question: does model scale meaningfully
affect prosociality scoring accuracy?

Reads the JSON reports produced by:
  - src/evaluate_deberta_vs_humans.py  (DeBERTa vs human consensus, test set)
  - src/evaluate_claude_deberta_agreement.py  (DeBERTa vs Claude, holdout set)

Produces a single comparison report with three sections:
  1. DeBERTa vs Human Consensus — primary validation
  2. DeBERTa vs Claude Scores   — secondary instrument comparison
  3. Scale Effect Summary        — large vs base delta, thesis framing guidance

Usage
-----
  python src/compare_deberta_models.py \\
      --large-human-report   output/artifacts/human_deberta-v3-large_eval_report.json \\
      --base-human-report    output/artifacts/human_deberta-v3-base_eval_report.json \\
      --large-claude-report  output/artifacts/claude_deberta-v3-large_agreement_report.json \\
      --base-claude-report   output/artifacts/claude_deberta-v3-base_agreement_report.json

Outputs
-------
  output/artifacts/deberta_model_comparison.txt
  output/artifacts/deberta_model_comparison.json
  output/artifacts/phase2_tables/model_comparison_table.csv  (thesis-ready table)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "output" / "artifacts"
TABLES_DIR    = ARTIFACTS_DIR / "phase2_tables"
LOG_DIR       = ROOT / "logs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "compare_deberta_models.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]

ICC_BENCHMARKS = "<0.50 poor | 0.50-0.75 moderate | 0.75-0.90 good | >0.90 excellent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path, label: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        log.warning(f"{label} report not found: {p}  (section will be skipped)")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _icc_label(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if v < 0.50:   return "poor"
    if v < 0.75:   return "moderate"
    if v < 0.90:   return "good ✓"
    return                "excellent ✓"


def _get_dim_metric(report: dict | None, dim: str, metric: str) -> float | None:
    if report is None:
        return None
    m = report.get("dimensions", {}).get(dim, {})
    return m.get(metric)


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a  "
    return f"{v:>{decimals + 4}.{decimals}f}"


# ---------------------------------------------------------------------------
# Comparison table builders
# ---------------------------------------------------------------------------

def build_dimension_table(
    large_report: dict | None,
    base_report:  dict | None,
    metric:       str,
    label:        str,
) -> pd.DataFrame:
    """One row per dimension, columns: dimension, large_{metric}, base_{metric}, delta."""
    rows = []
    for dim in DIMENSIONS:
        l_val = _get_dim_metric(large_report, dim, metric)
        b_val = _get_dim_metric(base_report,  dim, metric)
        delta = (l_val - b_val) if (l_val is not None and b_val is not None) else None
        rows.append({
            "dimension":           dim,
            f"large_{metric}":     l_val,
            f"base_{metric}":      b_val,
            f"delta_{metric}":     delta,
        })
    # Add mean row
    l_vals = [r[f"large_{metric}"] for r in rows if r[f"large_{metric}"] is not None]
    b_vals = [r[f"base_{metric}"]  for r in rows if r[f"base_{metric}"]  is not None]
    rows.append({
        "dimension":        "MEAN",
        f"large_{metric}":  round(float(np.mean(l_vals)), 4) if l_vals else None,
        f"base_{metric}":   round(float(np.mean(b_vals)), 4) if b_vals else None,
        f"delta_{metric}":  round(float(np.mean(l_vals) - np.mean(b_vals)), 4)
                            if (l_vals and b_vals) else None,
    })
    return pd.DataFrame(rows)


def scale_effect_interpretation(
    large_human: dict | None,
    base_human:  dict | None,
) -> str:
    """
    Generate a thesis-ready interpretation of the scale effect.
    Benchmarks: delta ICC < 0.05 = negligible; 0.05-0.10 = modest; > 0.10 = substantial.
    """
    if large_human is None or base_human is None:
        return "Scale effect cannot be assessed — one or both human evaluation reports are missing."

    icc_deltas = []
    mae_deltas = []
    for dim in DIMENSIONS:
        l_icc = _get_dim_metric(large_human, dim, "icc_2_1")
        b_icc = _get_dim_metric(base_human,  dim, "icc_2_1")
        l_mae = _get_dim_metric(large_human, dim, "mae")
        b_mae = _get_dim_metric(base_human,  dim, "mae")
        if l_icc is not None and b_icc is not None:
            icc_deltas.append(l_icc - b_icc)
        if l_mae is not None and b_mae is not None:
            mae_deltas.append(l_mae - b_mae)   # negative = large is better

    if not icc_deltas:
        return "Insufficient data to assess scale effect."

    mean_icc_delta = float(np.mean(icc_deltas))
    mean_mae_delta = float(np.mean(mae_deltas)) if mae_deltas else float("nan")

    # Did large meet the 0.75 target?
    l_met = sum(
        1 for d in DIMENSIONS
        if (_get_dim_metric(large_human, d, "icc_2_1") or 0) >= 0.75
    )
    b_met = sum(
        1 for d in DIMENSIONS
        if (_get_dim_metric(base_human,  d, "icc_2_1") or 0) >= 0.75
    )

    if abs(mean_icc_delta) < 0.05:
        scale_verdict = (
            "negligible (Δ ICC < 0.05): DeBERTa-v3-base is a viable alternative "
            "to DeBERTa-v3-large for prosociality scoring.  The base model's "
            "accessibility advantage (fits on consumer 16 GB VRAM) outweighs "
            "the marginal accuracy gain from the larger model."
        )
    elif abs(mean_icc_delta) < 0.10:
        scale_verdict = (
            f"modest (Δ ICC = {mean_icc_delta:+.3f}): DeBERTa-v3-large scores "
            "moderately higher than base.  Report both models; recommend large "
            "for research applications and base for resource-constrained deployment."
        )
    else:
        scale_verdict = (
            f"substantial (Δ ICC = {mean_icc_delta:+.3f}): DeBERTa-v3-large "
            "meaningfully outperforms DeBERTa-v3-base.  The larger model is "
            "recommended for prosociality scoring tasks."
        )

    lines = [
        f"Mean ICC delta (large − base): {mean_icc_delta:+.4f}",
        f"Mean MAE delta (large − base): {mean_mae_delta:+.4f}  "
        f"(negative = large has lower error)",
        f"Dimensions meeting ICC >= 0.75: large = {l_met}/4,  base = {b_met}/4",
        f"Scale effect: {scale_verdict}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    large_human:  dict | None,
    base_human:   dict | None,
    large_claude: dict | None,
    base_claude:  dict | None,
) -> str:
    W = 86
    lines = [
        "=" * W,
        "  DeBERTa Model Comparison: large vs base",
        "  Primary: DeBERTa vs Human Consensus (test set)",
        "  Secondary: DeBERTa vs Claude Scores (holdout set)",
        "=" * W,
        "",
    ]

    # ---- Section 1: vs Humans ----
    lines += [
        "  Section 1 — DeBERTa vs Human Consensus (PRIMARY VALIDATION)",
        "  " + "-" * (W - 2),
        f"  {'':20s}  {'large ICC':>9s}  {'base ICC':>8s}  {'Δ ICC':>7s}  "
        f"{'large MAE':>9s}  {'base MAE':>8s}  {'Δ MAE':>7s}",
        "  " + "-" * (W - 2),
    ]

    for dim in DIMENSIONS:
        l_icc = _get_dim_metric(large_human, dim, "icc_2_1")
        b_icc = _get_dim_metric(base_human,  dim, "icc_2_1")
        l_mae = _get_dim_metric(large_human, dim, "mae")
        b_mae = _get_dim_metric(base_human,  dim, "mae")
        d_icc = (l_icc - b_icc) if (l_icc and b_icc) else None
        d_mae = (l_mae - b_mae) if (l_mae and b_mae) else None
        lines.append(
            f"  {dim:20s}  {_fmt(l_icc)}  {_fmt(b_icc)}  {_fmt(d_icc, 4)}  "
            f"{_fmt(l_mae)}  {_fmt(b_mae)}  {_fmt(d_mae, 4)}"
        )

    lines += [
        "  " + "-" * (W - 2),
        "",
        "  Scale Effect (large vs base on human scores):",
    ]
    for line in scale_effect_interpretation(large_human, base_human).splitlines():
        lines.append(f"    {line}")

    lines += [
        "",
        f"  ICC benchmarks: {ICC_BENCHMARKS}",
        "  Target: ICC >= 0.75 per dimension",
        "",
    ]

    # ---- Section 2: vs Claude ----
    lines += [
        "  Section 2 — DeBERTa vs Claude Scores (SECONDARY: instrument comparison)",
        "  (Claude-DeBERTa agreement holdout — excluded from all training stages)",
        "  " + "-" * (W - 2),
        f"  {'':20s}  {'large ICC':>9s}  {'base ICC':>8s}  {'Δ ICC':>7s}  "
        f"{'large bias':>10s}  {'base bias':>9s}",
        "  " + "-" * (W - 2),
    ]
    for dim in DIMENSIONS:
        l_icc  = _get_dim_metric(large_claude, dim, "icc_2_1")
        b_icc  = _get_dim_metric(base_claude,  dim, "icc_2_1")
        # Bias key differs between the two report formats
        l_bias = _get_dim_metric(large_claude, dim, "bias_claude_minus_deberta")
        b_bias = _get_dim_metric(base_claude,  dim, "bias_claude_minus_deberta")
        d_icc  = (l_icc - b_icc) if (l_icc and b_icc) else None
        lines.append(
            f"  {dim:20s}  {_fmt(l_icc)}  {_fmt(b_icc)}  {_fmt(d_icc, 4)}  "
            f"{_fmt(l_bias, 3):>10s}  {_fmt(b_bias, 3):>9s}"
        )

    lines += [
        "  " + "-" * (W - 2),
        "  Bias = mean(Claude − DeBERTa).  Positive = Claude scores higher than DeBERTa.",
        "  Systematic bias signals where Stage 3 human alignment moved DeBERTa",
        "  away from Claude (expected where humans and Claude disagreed).",
        "",
    ]

    # ---- Section 3: Thesis framing ----
    lines += [
        "  Section 3 — Thesis Framing Guidance",
        "  " + "-" * (W - 2),
        "  Primary result:  Report ICC vs humans as the validation metric.",
        "  Secondary result: Report ICC vs Claude as an instrument comparison.",
        "  Systematic Claude-DeBERTa divergence = 'human correction signal'",
        "    (how much Stage 3 alignment moved DeBERTa away from Claude).",
        "  Recommended framing:",
        "    'DeBERTa-v3-large achieved ICC(2,1) = X.XX vs human consensus on the",
        "     held-out test set (range X.XX-X.XX across dimensions), meeting / not meeting",
        "     the 0.75 reliability threshold.  DeBERTa-v3-base achieved ICC = X.XX.",
        "     The scale difference resulted in a [negligible/modest/substantial] accuracy",
        "     gain, suggesting [base is viable / large is recommended] for deployment.",
        "     After Stage 3 human alignment, DeBERTa scores diverged from Claude by",
        "     a mean of X.XX points (MAE) on the 1-5 scale, reflecting the degree to",
        "     which human and Claude raters disagreed during annotation.'",
        "",
        "=" * W,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of DeBERTa-v3-large vs DeBERTa-v3-base."
    )
    parser.add_argument(
        "--large-human-report",
        default=str(ARTIFACTS_DIR / "human_deberta-v3-large_eval_report.json"),
        help="JSON from evaluate_deberta_vs_humans.py for DeBERTa-v3-large.",
    )
    parser.add_argument(
        "--base-human-report",
        default=str(ARTIFACTS_DIR / "human_deberta-v3-base_eval_report.json"),
        help="JSON from evaluate_deberta_vs_humans.py for DeBERTa-v3-base.",
    )
    parser.add_argument(
        "--large-claude-report",
        default=str(ARTIFACTS_DIR / "claude_deberta-v3-large_agreement_report.json"),
        help="JSON from evaluate_claude_deberta_agreement.py for DeBERTa-v3-large.",
    )
    parser.add_argument(
        "--base-claude-report",
        default=str(ARTIFACTS_DIR / "claude_deberta-v3-base_agreement_report.json"),
        help="JSON from evaluate_claude_deberta_agreement.py for DeBERTa-v3-base.",
    )
    parser.add_argument(
        "--output-dir", default=str(ARTIFACTS_DIR),
        help=f"Output directory (default: {ARTIFACTS_DIR}).",
    )
    args = parser.parse_args()

    log.info("=== Compare DeBERTa Models ===")

    large_human  = _load_json(args.large_human_report,  "large-human")
    base_human   = _load_json(args.base_human_report,   "base-human")
    large_claude = _load_json(args.large_claude_report, "large-claude")
    base_claude  = _load_json(args.base_claude_report,  "base-claude")

    if all(r is None for r in [large_human, base_human, large_claude, base_claude]):
        log.error("No evaluation reports found.  Run the evaluation scripts first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build thesis-ready CSV comparison table
    # ------------------------------------------------------------------
    rows = []
    for dim in DIMENSIONS:
        rows.append({
            "dimension":              dim,
            "large_icc_vs_human":     _get_dim_metric(large_human,  dim, "icc_2_1"),
            "base_icc_vs_human":      _get_dim_metric(base_human,   dim, "icc_2_1"),
            "large_mae_vs_human":     _get_dim_metric(large_human,  dim, "mae"),
            "base_mae_vs_human":      _get_dim_metric(base_human,   dim, "mae"),
            "large_icc_vs_claude":    _get_dim_metric(large_claude, dim, "icc_2_1"),
            "base_icc_vs_claude":     _get_dim_metric(base_claude,  dim, "icc_2_1"),
            "large_bias_vs_claude":   _get_dim_metric(large_claude, dim, "bias_claude_minus_deberta"),
            "base_bias_vs_claude":    _get_dim_metric(base_claude,  dim, "bias_claude_minus_deberta"),
        })
    df_table = pd.DataFrame(rows)
    table_path = TABLES_DIR / "model_comparison_table.csv"
    df_table.to_csv(table_path, index=False)
    log.info(f"Comparison table → {table_path}")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    report = format_report(large_human, base_human, large_claude, base_claude)

    out_dir   = Path(args.output_dir)
    txt_path  = out_dir / "deberta_model_comparison.txt"
    json_path = out_dir / "deberta_model_comparison.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    combined = {
        "large_human_eval":  large_human,
        "base_human_eval":   base_human,
        "large_claude_eval": large_claude,
        "base_claude_eval":  base_claude,
        "scale_effect":      scale_effect_interpretation(large_human, base_human),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    log.info(f"Comparison report (text) → {txt_path}")
    log.info(f"Comparison report (JSON) → {json_path}")
    print("\n" + report)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
