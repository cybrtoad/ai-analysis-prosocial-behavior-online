"""
src/sensitivity_respect.py
===========================
Sensitivity analysis: does dropping `respect` from the prosociality composite
change any substantive conclusions?

Background: `respect` has a 36.5% ceiling effect (36.5% of records scored 5.0),
making it right-censored. This script replicates the key Phase 2 analyses using
a 3-dimension composite (empathy + constructiveness + social_cohesion) and
reports whether findings hold.

Inputs
------
  output/phase2/phase2_analysis_dataset.parquet          (Reddit + SE)
  output/phase2/phase2_analysis_dataset_wikipedia.parquet

Output
------
  output/artifacts/sensitivity_respect.txt
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT      = Path(__file__).resolve().parent.parent
PHASE2    = ROOT / "output" / "phase2"
OUT_DIR   = ROOT / "output" / "artifacts"
LOG_DIR   = ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMS_4 = ["empathy", "constructiveness", "respect", "social_cohesion"]
DIMS_3 = ["empathy", "constructiveness", "social_cohesion"]
ALPHA  = 0.05


# ---------------------------------------------------------------------------
# Stat helpers (minimal, self-contained)
# ---------------------------------------------------------------------------

def kw_eta2(h, k, n):
    if n <= k:
        return float("nan")
    return (h - k + 1) / (n - k)

def effect_label(eta2):
    if np.isnan(eta2) or eta2 < 0:  return "negligible"
    if eta2 < 0.01:                  return "negligible"
    if eta2 < 0.06:                  return "small"
    if eta2 < 0.14:                  return "medium"
    return                                  "large"

def kw_test(df, group_col, value_col):
    groups = {g: grp[value_col].dropna().to_numpy()
              for g, grp in df.groupby(group_col)}
    groups = {g: v for g, v in groups.items() if len(v) >= 3}
    if len(groups) < 2:
        return None
    arrays = list(groups.values())
    n = sum(len(a) for a in arrays)
    k = len(groups)
    h, p = stats.kruskal(*arrays)
    eta2 = kw_eta2(h, k, n)
    return {"H": h, "p": p, "eta2": eta2, "effect": effect_label(eta2), "k": k, "n": n}

def spearman(df, x_col, y_col):
    mask = df[x_col].notna() & df[y_col].notna()
    x = df.loc[mask, x_col].to_numpy(dtype=float)
    y = df.loc[mask, y_col].to_numpy(dtype=float)
    if len(x) < 5:
        return None
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": len(x)}


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare(df, label, group_col, value_4, value_3):
    r4 = kw_test(df, group_col, value_4)
    r3 = kw_test(df, group_col, value_3)
    if r4 is None or r3 is None:
        return f"  {label}: insufficient groups\n"

    lines = [
        f"  {label}",
        f"    4-dim  H={r4['H']:.3f}  p={r4['p']:.4g}  eta2={r4['eta2']:.4f}  ({r4['effect']})",
        f"    3-dim  H={r3['H']:.3f}  p={r3['p']:.4g}  eta2={r3['eta2']:.4f}  ({r3['effect']})",
    ]

    # Check if effect label or significance changes
    sig_change   = (r4["p"] < ALPHA) != (r3["p"] < ALPHA)
    effect_change = r4["effect"] != r3["effect"]
    if sig_change:
        lines.append("    *** SIGNIFICANCE CHANGES ***")
    elif effect_change:
        lines.append(f"    NOTE: effect label changes ({r4['effect']} → {r3['effect']})")
    else:
        lines.append("    Conclusion unchanged.")
    return "\n".join(lines)


def compare_spearman(df, label, x_col, value_4, value_3):
    r4 = spearman(df, x_col, value_4)
    r3 = spearman(df, x_col, value_3)
    if r4 is None or r3 is None:
        return f"  {label}: insufficient data\n"
    lines = [
        f"  {label}",
        f"    4-dim  rho={r4['rho']:.4f}  p={r4['p']:.4g}  n={r4['n']:,}",
        f"    3-dim  rho={r3['rho']:.4f}  p={r3['p']:.4g}  n={r3['n']:,}",
    ]
    sig_change = (r4["p"] < ALPHA) != (r3["p"] < ALPHA)
    if sig_change:
        lines.append("    *** SIGNIFICANCE CHANGES ***")
    else:
        lines.append("    Conclusion unchanged.")
    return "\n".join(lines)


def median_table(df, group_col, value_4, value_3):
    rows = []
    for g, grp in df.groupby(group_col):
        m4 = grp[value_4].median()
        m3 = grp[value_3].median()
        rows.append(f"    {str(g):25s}  4-dim median={m4:.3f}  3-dim median={m3:.3f}  Δ={m3-m4:+.3f}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(df, name):
    W = 78
    lines = [
        "=" * W,
        f"  Sensitivity Analysis: Respect Ceiling  —  {name}",
        "=" * W,
        f"  n = {len(df):,}",
        "",
        "  Composites:",
        f"    4-dim (baseline):    mean(empathy, constructiveness, respect, social_cohesion)",
        f"    3-dim (sensitivity): mean(empathy, constructiveness, social_cohesion)",
        "",
        "  Respect ceiling effect:",
    ]

    resp = df["respect"].dropna()
    pct_ceil = (resp == 5.0).mean() * 100
    lines.append(f"    {pct_ceil:.1f}% of records at 5.0 (ceiling)")
    lines.append("")

    # Compute 3-dim composite
    df = df.copy()
    df["composite_3dim"] = df[DIMS_3].mean(axis=1).round(4)
    comp4 = "prosociality_composite"
    comp3 = "composite_3dim"

    # Overall distribution shift
    m4 = df[comp4].mean()
    m3 = df[comp3].mean()
    sd4 = df[comp4].std()
    sd3 = df[comp3].std()
    lines += [
        "  Overall composite shift:",
        f"    4-dim  mean={m4:.3f}  sd={sd4:.3f}",
        f"    3-dim  mean={m3:.3f}  sd={sd3:.3f}  Δmean={m3-m4:+.3f}",
        "",
    ]

    # RQ1 — Intervention type
    lines += ["  " + "-" * (W-2), "  RQ1 — Intervention type × composite", "  " + "-" * (W-2)]
    lines.append(compare(df, "KW test", "intervention_type", comp4, comp3))
    lines.append("")
    lines.append("  Medians by intervention type:")
    lines.append(median_table(df, "intervention_type", comp4, comp3))
    lines.append("")

    # RQ2b — Intervener role
    if "intervener_role" in df.columns:
        lines += ["  " + "-" * (W-2), "  RQ2b — Intervener role × composite", "  " + "-" * (W-2)]
        lines.append(compare(df, "KW test", "intervener_role", comp4, comp3))
        lines.append("")
        lines.append("  Medians by intervener role:")
        lines.append(median_table(df, "intervener_role", comp4, comp3))
        lines.append("")

    # RQ4 — Platform
    if df["platform"].nunique() > 1:
        lines += ["  " + "-" * (W-2), "  RQ4 — Platform × composite", "  " + "-" * (W-2)]
        lines.append(compare(df, "KW test", "platform", comp4, comp3))
        lines.append("")
        lines.append("  Medians by platform:")
        lines.append(median_table(df, "platform", comp4, comp3))
        lines.append("")

    # RQ3 — Timing
    df_rt = df[df["response_time_minutes"].notna() & (df["response_time_minutes"] > 0)]
    lines += ["  " + "-" * (W-2), "  RQ3 — Response time × composite (Spearman)", "  " + "-" * (W-2)]
    lines.append(compare_spearman(df_rt, "Spearman rho", "response_time_minutes", comp4, comp3))
    lines.append("")

    # Overall verdict
    lines += ["  " + "=" * (W-2), "  VERDICT", "  " + "=" * (W-2)]
    lines.append(
        "  Review the 'Conclusion unchanged.' / 'NOTE' / '*** SIGNIFICANCE CHANGES ***'\n"
        "  lines above. If all key comparisons read 'Conclusion unchanged.', the respect\n"
        "  ceiling does not affect the substantive findings."
    )
    lines.append("=" * W)

    return "\n".join(lines)


def main():
    log.info("=== Respect Sensitivity Analysis ===")

    main_path = PHASE2 / "phase2_analysis_dataset.parquet"
    wiki_path = PHASE2 / "phase2_analysis_dataset_wikipedia.parquet"

    sections = []

    if main_path.exists():
        df_main = pd.read_parquet(main_path)
        sections.append(run_dataset(df_main, "Reddit + StackExchange (main)"))
    else:
        log.warning(f"Not found: {main_path}")

    if wiki_path.exists():
        df_wiki = pd.read_parquet(wiki_path)
        sections.append(run_dataset(df_wiki, "Wikipedia (exploratory)"))
    else:
        log.warning(f"Not found: {wiki_path}")

    report = "\n\n".join(sections)
    print("\n" + report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sensitivity_respect.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
