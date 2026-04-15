"""
src/audit_phase2_scores.py
===========================
Data quality audit for Phase 2 Claude scores.  Run this BEFORE
build_phase2_features.py to catch problems in the raw scores file.

Checks
------
  - Record counts: total / malformed (score = -1) / invalid_triple / valid
  - Per-platform and per-intervention-type breakdown of the above
  - Score distributions per dimension: percentiles, mean, SD, floor/ceiling rates
  - Confidence score distribution
  - Cross-tab: platform × intervention_type × valid count
  - Flag conditions that should pause further analysis

Input
-----
  output/phase2/phase2_claude_scores.parquet  (Claude scores, raw from label.py)
  output/phase2/phase2_sample.parquet         (for platform / intervention_type lookup)

Outputs
-------
  output/artifacts/audit_report.txt
  output/artifacts/audit_tables/
    audit_summary.csv                  overall record counts
    audit_by_platform.csv              counts by platform
    audit_by_intervention_type.csv     counts by intervention_type
    audit_platform_x_type.csv          platform × type crosstab
    score_distributions.csv            percentiles per dimension (all valid records)
    score_distributions_by_platform.csv  percentiles per dimension, split by platform
    confidence_distribution.csv        confidence percentiles + bucket counts
    floor_ceiling.csv                  % at min/max score per dimension

Usage
-----
  python src/audit_phase2_scores.py
  python src/audit_phase2_scores.py \\
      --scores output/phase2/phase2_claude_scores.parquet \\
      --sample output/phase2/phase2_sample.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
PHASE2     = ROOT / "output" / "phase2"
AUDIT_DIR  = ROOT / "output" / "artifacts" / "audit_tables"
LOG_DIR    = ROOT / "logs"

for _d in [AUDIT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "audit_phase2_scores.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def score_distribution(series: pd.Series, label: str) -> dict:
    """Percentiles, mean, SD, floor/ceiling rates for a score series."""
    clean = series.dropna()
    clean = clean[clean >= 1.0]          # exclude malformed (-1)
    n = len(clean)
    if n == 0:
        return {"label": label, "n": 0}
    row = {"label": label, "n": n, "mean": round(clean.mean(), 4), "sd": round(clean.std(), 4)}
    for p in PERCENTILES:
        row[f"p{p}"] = round(float(np.percentile(clean, p)), 4)
    row["pct_floor"] = round(100 * (clean == 1.0).mean(), 2)   # % at minimum
    row["pct_ceiling"] = round(100 * (clean == 5.0).mean(), 2)  # % at maximum
    return row


def confidence_buckets(series: pd.Series) -> pd.DataFrame:
    """Count records in confidence bands [0,0.5), [0.5,0.7), [0.7,0.9), [0.9,1.0]."""
    clean = series.dropna()
    bins   = [0.0, 0.5, 0.7, 0.9, 1.001]
    labels = ["<0.50", "0.50–0.69", "0.70–0.89", "0.90–1.00"]
    counts = pd.cut(clean, bins=bins, labels=labels, right=False).value_counts().sort_index()
    total  = len(clean)
    rows = []
    for band, n in counts.items():
        rows.append({"confidence_band": band, "n": int(n), "pct": _pct(int(n), total)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit(df_scores: pd.DataFrame, df_sample: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Run all checks.  Returns (report_text, {table_name: DataFrame}).
    """
    tables: dict[str, pd.DataFrame] = {}
    W = 78
    lines = [
        "=" * W,
        "  Phase 2 Claude Scores — Data Quality Audit",
        "=" * W,
    ]

    # ------------------------------------------------------------------
    # Merge scores with sample for platform / intervention_type
    # ------------------------------------------------------------------
    meta_cols = ["record_id", "platform", "intervention_type"]
    available = [c for c in meta_cols if c in df_sample.columns]
    df = df_scores.merge(df_sample[available], on="record_id", how="left")

    # ------------------------------------------------------------------
    # Classify each record
    # ------------------------------------------------------------------
    # Malformed: any core dimension score == -1 (label.py sentinel)
    malformed_mask = (
        df[DIMENSIONS].eq(-1).any(axis=1)
        if all(d in df.columns for d in DIMENSIONS)
        else pd.Series(False, index=df.index)
    )
    invalid_mask = df["invalid_triple"].fillna(False).astype(bool) if "invalid_triple" in df.columns else pd.Series(False, index=df.index)
    # A malformed record may also be flagged invalid — count malformed first
    valid_mask = ~malformed_mask & ~invalid_mask

    n_total    = len(df)
    n_malformed = malformed_mask.sum()
    n_invalid  = (~malformed_mask & invalid_mask).sum()   # invalid but not malformed
    n_valid    = valid_mask.sum()

    # ------------------------------------------------------------------
    # Section 1: Overall summary
    # ------------------------------------------------------------------
    lines += [
        "",
        f"  {'Total records in scores file':<35s} {n_total:>8,}",
        f"  {'Malformed (score = -1)':<35s} {n_malformed:>8,}  ({_pct(n_malformed, n_total)})",
        f"  {'Invalid triple (non-malformed)':<35s} {n_invalid:>8,}  ({_pct(n_invalid, n_total)})",
        f"  {'Valid, scoreable records':<35s} {n_valid:>8,}  ({_pct(n_valid, n_total)})",
    ]

    # Flag if malformed rate is high
    if n_total > 0 and n_malformed / n_total > 0.02:
        lines.append(f"\n  *** WARNING: malformed rate {_pct(n_malformed, n_total)} > 2% — investigate label.py logs ***")
    if n_total > 0 and n_invalid / n_total > 0.10:
        lines.append(f"\n  *** WARNING: invalid_triple rate {_pct(n_invalid, n_total)} > 10% — review prompt or data ***")

    summary_df = pd.DataFrame([
        {"category": "total",             "n": n_total,    "pct": "100%"},
        {"category": "malformed",         "n": n_malformed, "pct": _pct(n_malformed, n_total)},
        {"category": "invalid_triple",    "n": n_invalid,  "pct": _pct(n_invalid, n_total)},
        {"category": "valid",             "n": n_valid,    "pct": _pct(n_valid, n_total)},
    ])
    tables["audit_summary"] = summary_df

    # ------------------------------------------------------------------
    # Section 2: By platform
    # ------------------------------------------------------------------
    lines += ["", "  " + "-" * (W - 2), "  By Platform", "  " + "-" * (W - 2)]
    plat_rows = []
    if "platform" in df.columns:
        for plat, grp in df.groupby("platform", dropna=False):
            m = malformed_mask.loc[grp.index].sum()
            iv = (~malformed_mask.loc[grp.index] & invalid_mask.loc[grp.index]).sum()
            v  = valid_mask.loc[grp.index].sum()
            nt = len(grp)
            lines.append(
                f"  {str(plat):15s}  total={nt:>7,}  "
                f"malformed={m:>5,} ({_pct(m, nt)})  "
                f"invalid={iv:>5,} ({_pct(iv, nt)})  "
                f"valid={v:>7,} ({_pct(v, nt)})"
            )
            plat_rows.append({"platform": plat, "total": nt, "malformed": m,
                               "malformed_pct": _pct(m, nt), "invalid_triple": iv,
                               "invalid_triple_pct": _pct(iv, nt), "valid": v,
                               "valid_pct": _pct(v, nt)})
    else:
        lines.append("  platform column not available (sample not merged)")
    tables["audit_by_platform"] = pd.DataFrame(plat_rows)

    # ------------------------------------------------------------------
    # Section 3: By intervention type
    # ------------------------------------------------------------------
    lines += ["", "  " + "-" * (W - 2), "  By Intervention Type", "  " + "-" * (W - 2)]
    type_rows = []
    if "intervention_type" in df.columns:
        for itype, grp in df.groupby("intervention_type", dropna=False):
            m = malformed_mask.loc[grp.index].sum()
            iv = (~malformed_mask.loc[grp.index] & invalid_mask.loc[grp.index]).sum()
            v  = valid_mask.loc[grp.index].sum()
            nt = len(grp)
            lines.append(
                f"  {str(itype):30s}  total={nt:>7,}  "
                f"malformed={m:>5,} ({_pct(m, nt)})  "
                f"valid={v:>7,} ({_pct(v, nt)})"
            )
            type_rows.append({"intervention_type": itype, "total": nt, "malformed": m,
                               "malformed_pct": _pct(m, nt), "invalid_triple": iv,
                               "invalid_triple_pct": _pct(iv, nt), "valid": v,
                               "valid_pct": _pct(v, nt)})
    else:
        lines.append("  intervention_type column not available")
    tables["audit_by_intervention_type"] = pd.DataFrame(type_rows)

    # ------------------------------------------------------------------
    # Section 4: Platform × intervention_type crosstab (valid counts)
    # ------------------------------------------------------------------
    if "platform" in df.columns and "intervention_type" in df.columns:
        cross = (
            df[valid_mask]
            .groupby(["platform", "intervention_type"])
            .size()
            .reset_index(name="valid_n")
        )
        tables["audit_platform_x_type"] = cross
        lines += [
            "", "  " + "-" * (W - 2),
            "  Platform × Intervention Type (valid record counts)",
            "  " + "-" * (W - 2),
            cross.to_string(index=False),
        ]

    # ------------------------------------------------------------------
    # Section 5: Score distributions (valid records only)
    # ------------------------------------------------------------------
    df_valid = df[valid_mask].copy()

    lines += [
        "", "  " + "-" * (W - 2),
        "  Score Distributions — Valid Records (all platforms)",
        "  " + "-" * (W - 2),
        f"  {'Dimension':<22s}  {'n':>7s}  {'mean':>6s}  {'sd':>6s}  "
        + "  ".join(f"p{p:02d}" for p in PERCENTILES)
        + "  {'floor%':>7s}  {'ceil%':>6s}",
    ]

    dist_rows = []
    for dim in DIMENSIONS + ["prosociality_composite"] if "prosociality_composite" in df_valid.columns else DIMENSIONS:
        if dim not in df_valid.columns:
            continue
        r = score_distribution(df_valid[dim], dim)
        dist_rows.append(r)
        pct_str = "  ".join(f"{r.get(f'p{p}', float('nan')):>5.2f}" for p in PERCENTILES)
        lines.append(
            f"  {dim:<22s}  {r.get('n', 0):>7,}  {r.get('mean', float('nan')):>6.3f}  "
            f"{r.get('sd', float('nan')):>6.3f}  {pct_str}  "
            f"{r.get('pct_floor', 0):>7.1f}  {r.get('pct_ceiling', 0):>6.1f}"
        )

    tables["score_distributions"] = pd.DataFrame(dist_rows)

    # ------------------------------------------------------------------
    # Section 6: Score distributions by platform
    # ------------------------------------------------------------------
    plat_dist_rows = []
    if "platform" in df_valid.columns:
        lines += [
            "", "  " + "-" * (W - 2),
            "  Score Distributions — Prosociality Composite by Platform",
            "  " + "-" * (W - 2),
        ]
        comp_col = "prosociality_composite" if "prosociality_composite" in df_valid.columns else None
        if comp_col is None and all(d in df_valid.columns for d in DIMENSIONS):
            df_valid = df_valid.copy()
            df_valid["prosociality_composite"] = df_valid[DIMENSIONS].mean(axis=1)
            comp_col = "prosociality_composite"

        if comp_col:
            for plat, grp in df_valid.groupby("platform"):
                r = score_distribution(grp[comp_col], f"{plat}::composite")
                plat_dist_rows.append(r)
                pct_str = "  ".join(f"{r.get(f'p{p}', float('nan')):>5.2f}" for p in PERCENTILES)
                lines.append(
                    f"  {str(plat):<15s}  n={r.get('n', 0):>7,}  "
                    f"mean={r.get('mean', float('nan')):.3f}  sd={r.get('sd', float('nan')):.3f}  "
                    f"p25={r.get('p25', float('nan')):.2f}  p50={r.get('p50', float('nan')):.2f}  "
                    f"p75={r.get('p75', float('nan')):.2f}  "
                    f"floor={r.get('pct_floor', 0):.1f}%  ceil={r.get('pct_ceiling', 0):.1f}%"
                )

            for dim in DIMENSIONS:
                if dim not in df_valid.columns:
                    continue
                for plat, grp in df_valid.groupby("platform"):
                    r = score_distribution(grp[dim], f"{plat}::{dim}")
                    plat_dist_rows.append(r)

    tables["score_distributions_by_platform"] = pd.DataFrame(plat_dist_rows)

    # ------------------------------------------------------------------
    # Section 7: Confidence distribution
    # ------------------------------------------------------------------
    lines += [
        "", "  " + "-" * (W - 2),
        "  Confidence Distribution (valid records)",
        "  " + "-" * (W - 2),
    ]
    if "confidence" in df_valid.columns:
        conf = df_valid["confidence"].dropna()
        conf_pcts = {f"p{p}": round(float(np.percentile(conf, p)), 3) for p in PERCENTILES}
        lines.append(
            f"  n={len(conf):,}  mean={conf.mean():.3f}  sd={conf.std():.3f}  "
            + "  ".join(f"p{p}={v}" for p, v in conf_pcts.items())
        )
        lines.append("")
        conf_bucket_df = confidence_buckets(conf)
        tables["confidence_distribution"] = conf_bucket_df
        for _, row in conf_bucket_df.iterrows():
            lines.append(f"  {row['confidence_band']:>12s}  {row['n']:>7,}  {row['pct']}")
        lines.append("")

        low_conf = (conf < 0.5).sum()
        if low_conf / len(conf) > 0.10:
            lines.append(
                f"  *** WARNING: {_pct(low_conf, len(conf))} of records have confidence < 0.50 "
                f"— consider filtering low-confidence records in sensitivity analysis ***"
            )
    else:
        lines.append("  confidence column not found in scores file")

    # ------------------------------------------------------------------
    # Section 8: Floor / ceiling effects table
    # ------------------------------------------------------------------
    fc_rows = []
    if dist_rows:
        lines += [
            "", "  " + "-" * (W - 2),
            "  Floor / Ceiling Effects per Dimension",
            "  " + "-" * (W - 2),
            f"  {'Dimension':<25s}  {'n':>7s}  {'% at 1.0 (floor)':>18s}  {'% at 5.0 (ceil)':>16s}",
        ]
        for r in dist_rows:
            if r.get("n", 0) == 0:
                continue
            lines.append(
                f"  {r['label']:<25s}  {r['n']:>7,}  {r.get('pct_floor', 0):>18.1f}  {r.get('pct_ceiling', 0):>16.1f}"
            )
            fc_rows.append({
                "dimension": r["label"],
                "n": r["n"],
                "pct_floor": r.get("pct_floor", 0),
                "pct_ceiling": r.get("pct_ceiling", 0),
            })
    tables["floor_ceiling"] = pd.DataFrame(fc_rows)

    # ------------------------------------------------------------------
    # Final go/no-go assessment
    # ------------------------------------------------------------------
    lines += ["", "=" * W, "  Go / No-Go Assessment", "=" * W]
    issues = []
    if n_total > 0:
        if n_malformed / n_total > 0.02:
            issues.append(f"Malformed rate {_pct(n_malformed, n_total)} exceeds 2% threshold")
        if n_invalid / n_total > 0.10:
            issues.append(f"invalid_triple rate {_pct(n_invalid, n_total)} exceeds 10% threshold")
        if n_valid < 50_000:
            issues.append(f"Only {n_valid:,} valid records — unusually low for a 100K run")
        if "confidence" in df_valid.columns:
            low = (df_valid["confidence"] < 0.5).sum()
            if len(df_valid) > 0 and low / len(df_valid) > 0.10:
                issues.append(f"Low confidence rate {_pct(low, len(df_valid))} > 10%")

    if issues:
        lines.append("  STATUS: REVIEW REQUIRED before proceeding")
        for issue in issues:
            lines.append(f"    - {issue}")
    else:
        lines.append("  STATUS: OK — proceed to build_phase2_features.py")

    lines += ["", "  Next steps:", "    1. Review warnings above (if any)"]
    if not issues:
        lines += [
            "    2. python src/build_phase2_features.py",
            "    3. python src/analyze_phase2.py",
        ]
    lines += ["", "=" * W]

    return "\n".join(lines), tables


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data quality audit for Phase 2 Claude scores."
    )
    parser.add_argument(
        "--scores", default=str(PHASE2 / "phase2_claude_scores.parquet"),
        help="Claude scores parquet (default: output/phase2/phase2_claude_scores.parquet)",
    )
    parser.add_argument(
        "--sample", default=str(PHASE2 / "phase2_sample.parquet"),
        help="Phase 2 sample parquet for platform/type metadata (default: output/phase2/phase2_sample.parquet)",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "output" / "artifacts"),
        help="Directory for report and tables (default: output/artifacts/)",
    )
    args = parser.parse_args()

    log.info("=== Phase 2 Scores Audit ===")

    scores_path = Path(args.scores)
    sample_path = Path(args.sample)

    if not scores_path.exists():
        log.error(f"Scores file not found: {scores_path}")
        sys.exit(1)

    df_scores = pd.read_parquet(scores_path)
    log.info(f"Loaded scores: {len(df_scores):,} records, {len(df_scores.columns)} columns")
    log.info(f"Columns: {df_scores.columns.tolist()}")

    if sample_path.exists():
        df_sample = pd.read_parquet(sample_path)
        log.info(f"Loaded sample: {len(df_sample):,} records (for metadata join)")
    else:
        log.warning(f"Sample file not found at {sample_path} — platform/type breakdown will be skipped")
        df_sample = pd.DataFrame(columns=["record_id"])

    report_text, tables = audit(df_scores, df_sample)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir   = Path(args.output_dir)
    audit_dir = out_dir / "audit_tables"
    audit_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"\nReport saved → {report_path}")

    for name, df_table in tables.items():
        if df_table is not None and len(df_table) > 0:
            tbl_path = audit_dir / f"{name}.csv"
            df_table.to_csv(tbl_path, index=False)
            log.info(f"Table saved  → {tbl_path}")

    print("\n" + report_text)
    log.info("=== Audit complete ===")


if __name__ == "__main__":
    main()
