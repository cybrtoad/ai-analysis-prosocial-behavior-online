"""
src/analyze_phase2.py
======================
Quantitative analysis of Phase 2 intervention patterns.

Runs all research questions answerable from the current feature set and
produces publication-ready tables, a text report, and machine-readable JSON.
All findings use associational language — no causal claims.

Input
-----
  output/phase2/phase2_analysis_dataset.parquet
  (produced by src/build_phase2_features.py)

Outputs
-------
  output/artifacts/phase2_report.txt               Full text report
  output/artifacts/phase2_report.json              Machine-readable metrics
  output/artifacts/phase2_tables/                  One CSV per table
    descriptive_by_platform.csv
    descriptive_by_intervention_type.csv
    descriptive_by_intervener_role.csv
    rq1_intervention_type_prosociality.csv
    rq1_posthoc.csv
    rq2_tenure_moderation.csv
    rq2b_intervener_role_prosociality.csv
    rq2b_posthoc.csv
    rq2b_role_x_type_crosstab.csv
    rq3_timing_analysis.csv
    rq4_platform_comparison.csv
    rq4_platform_x_type_crosstab.csv
    dimension_correlations.csv
    dimension_pca.csv

Research questions addressed
-----------------------------
  RQ1   Which communication strategies (intervention types) are most associated
        with prosocial behavior?
        → Kruskal-Wallis + Dunn post-hoc + eta-squared effect size

  RQ2a  Does the pattern vary by user history?
        → user_prior_interventions_in_sample as proxy for intervention experience
        → Spearman correlation + KW by tenure category

  RQ2b  Does intervener role moderate the association between communication
        strategy and prosocial response?
        → intervener_role × prosociality_composite: KW + Dunn post-hoc
        → Cross-tab: intervener_role × intervention_type mean composite

  RQ3   Does timing matter?
        → response_time_category × prosociality_composite
        → Kruskal-Wallis + Dunn post-hoc

  RQ4   Cross-platform comparison
        → Platform × prosociality + Platform × intervention_type interaction

  RQ5   Dimension structure
        → Pearson correlations, PCA to assess whether dimensions collapse
        → Informs thesis framing of composite vs. subscores

  Not yet computable (require additional data extraction):
        → intervention_style (only 340 human-annotated records)
        → intervention_visibility (derivable but not meaningfully variable
          in current data — all three platforms are effectively public)
        → user_retention_7day / user_retention_30day (need full activity timeline)
        → sustained_behavior_score / immediate_behavioral_change
          (need multiple post-intervention texts per user)

Statistical notes
-----------------
  Kruskal-Wallis H-test is used throughout (non-parametric, no normality
  assumption) rather than ANOVA.  Likert-scale scores are ordinal and
  distributions are visibly non-normal (heavy left skew for restriction/warning).

  Effect size: eta-squared from KW statistic:
    eta2 = (H - k + 1) / (n - k)
  where H = KW statistic, k = number of groups, n = total sample size.
  Benchmarks: small >= 0.01, medium >= 0.06, large >= 0.14 (Cohen, 1988 adapted).

  Post-hoc pairwise comparisons: Mann-Whitney U with Bonferroni correction.
  Reported as corrected p-values.

  Spearman correlation is used for continuous predictors (response_time_minutes,
  user_prior_interventions_in_sample).

Usage
-----
  python src/analyze_phase2.py
  python src/analyze_phase2.py --input output/phase2/phase2_analysis_dataset.parquet
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT          = Path(__file__).resolve().parent.parent
PHASE2_DIR    = ROOT / "output" / "phase2"
ARTIFACTS_DIR = ROOT / "output" / "artifacts"
TABLES_DIR    = ARTIFACTS_DIR / "phase2_tables"
LOG_DIR       = ROOT / "logs"

for _d in [ARTIFACTS_DIR, TABLES_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "analyze_phase2.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]
ALPHA      = 0.05


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def kruskal_eta_squared(h_stat: float, k: int, n: int) -> float:
    """Eta-squared effect size from KW H statistic."""
    if n <= k:
        return float("nan")
    return (h_stat - k + 1) / (n - k)


def _effect_label(eta2: float) -> str:
    if np.isnan(eta2) or eta2 < 0:  return "negligible"
    if eta2 < 0.01:                  return "negligible"
    if eta2 < 0.06:                  return "small"
    if eta2 < 0.14:                  return "medium"
    return                                  "large"


def kruskal_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """
    Kruskal-Wallis H-test for value_col across levels of group_col.
    Returns H, p, eta2, k, n, and group sizes.
    """
    groups = {g: grp[value_col].dropna().to_numpy() for g, grp in df.groupby(group_col)}
    groups = {g: v for g, v in groups.items() if len(v) >= 3}
    k = len(groups)
    if k < 2:
        return {"error": "fewer than 2 groups with n >= 3"}

    arrays = list(groups.values())
    n = sum(len(a) for a in arrays)
    h, p = stats.kruskal(*arrays)
    eta2 = kruskal_eta_squared(h, k, n)

    return {
        "k":       k,
        "n":       n,
        "H":       round(h, 4),
        "p":       round(p, 6),
        "eta2":    round(eta2, 4),
        "effect":  _effect_label(eta2),
        "group_n": {g: len(v) for g, v in groups.items()},
    }


def dunn_posthoc(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.
    Returns a DataFrame with columns: group_a, group_b, U, p_raw, p_bonf, significant.
    """
    groups = {g: grp[value_col].dropna().to_numpy() for g, grp in df.groupby(group_col)}
    groups = {g: v for g, v in groups.items() if len(v) >= 3}
    pairs  = list(combinations(sorted(groups.keys()), 2))
    n_comp = len(pairs)

    rows = []
    for ga, gb in pairs:
        u, p_raw = stats.mannwhitneyu(groups[ga], groups[gb], alternative="two-sided")
        p_bonf = min(p_raw * n_comp, 1.0)
        med_a = float(np.median(groups[ga]))
        med_b = float(np.median(groups[gb]))
        rows.append({
            "group_a":      ga,
            "group_b":      gb,
            "n_a":          len(groups[ga]),
            "n_b":          len(groups[gb]),
            "median_a":     round(med_a, 3),
            "median_b":     round(med_b, 3),
            "U":            round(u, 2),
            "p_raw":        round(p_raw, 6),
            "p_bonf":       round(p_bonf, 6),
            "significant":  p_bonf < ALPHA,
        })
    return pd.DataFrame(rows)


def spearman_corr(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """Spearman correlation between two columns, dropping NaN pairs."""
    mask = df[x_col].notna() & df[y_col].notna()
    x = df.loc[mask, x_col].to_numpy(dtype=float)
    y = df.loc[mask, y_col].to_numpy(dtype=float)
    if len(x) < 5:
        return {"n": len(x), "error": "insufficient data"}
    rho, p = stats.spearmanr(x, y)
    return {"n": len(x), "rho": round(float(rho), 4), "p": round(float(p), 6)}


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean ± SD for prosociality_composite and each dimension, by group."""
    rows = []
    for grp_val, grp in df.groupby(group_col):
        row = {group_col: grp_val, "n": len(grp)}
        for col in ["prosociality_composite"] + DIMENSIONS:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = round(vals.mean(), 3)
            row[f"{col}_std"]  = round(vals.std(),  3)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ1: Intervention type × prosociality
# ---------------------------------------------------------------------------

def rq1_intervention_type(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    KW test + Dunn post-hoc for intervention_type × prosociality_composite.
    Also runs KW for each individual dimension.
    """
    log.info("RQ1: Intervention type × prosociality composite")
    kw = kruskal_test(df, "intervention_type", "prosociality_composite")
    log.info(f"  KW H={kw.get('H'):.3f}  p={kw.get('p'):.4g}  eta2={kw.get('eta2'):.3f}  ({kw.get('effect')})")

    posthoc = dunn_posthoc(df, "intervention_type", "prosociality_composite")

    # Per-dimension KW
    dim_kw = {}
    for dim in DIMENSIONS:
        dim_kw[dim] = kruskal_test(df, "intervention_type", dim)

    desc = descriptive_table(df, "intervention_type")

    results = {
        "kw_composite": kw,
        "kw_per_dimension": dim_kw,
        "group_medians": df.groupby("intervention_type")["prosociality_composite"]
                           .median().round(3).to_dict(),
    }
    return results, desc, posthoc


# ---------------------------------------------------------------------------
# RQ2: User moderation history
# ---------------------------------------------------------------------------

def rq2_user_history(df: pd.DataFrame) -> dict:
    """
    Spearman correlation: user_prior_interventions_in_sample × prosociality_composite.
    Also KW by tenure estimate category.
    """
    log.info("RQ2: User moderation history × prosociality composite")
    spear = spearman_corr(df, "user_prior_interventions_in_sample", "prosociality_composite")
    log.info(f"  Spearman rho={spear.get('rho'):.3f}  p={spear.get('p'):.4g}  n={spear.get('n')}")

    kw_tenure = kruskal_test(df, "user_tenure_estimate_category", "prosociality_composite")
    posthoc_tenure = dunn_posthoc(df, "user_tenure_estimate_category", "prosociality_composite")

    # Per-dimension Spearman
    dim_spear = {
        dim: spearman_corr(df, "user_prior_interventions_in_sample", dim)
        for dim in DIMENSIONS
    }

    return {
        "spearman_prior_interventions_x_composite": spear,
        "spearman_per_dimension": dim_spear,
        "kw_tenure_category": kw_tenure,
        "tenure_posthoc": posthoc_tenure.to_dict(orient="records"),
        "tenure_group_medians": df.groupby("user_tenure_estimate_category")["prosociality_composite"]
                                  .median().round(3).to_dict(),
        "note": (
            "user_prior_interventions_in_sample is an in-sample proxy.  "
            "True user_previous_interventions requires full platform activity data."
        ),
    }


# ---------------------------------------------------------------------------
# RQ2b: Intervener role × prosociality
# ---------------------------------------------------------------------------

def rq2b_intervener_role(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    KW test + Dunn post-hoc for intervener_role × prosociality_composite.
    Also cross-tabulation: intervener_role × intervention_type mean composite,
    to show whether role or type is the stronger differentiator.
    """
    log.info("RQ2b: Intervener role × prosociality composite")

    if "intervener_role" not in df.columns:
        log.warning("  intervener_role column not found — skipping RQ2b")
        empty = pd.DataFrame()
        return {"error": "intervener_role column missing"}, empty, empty, empty

    kw = kruskal_test(df, "intervener_role", "prosociality_composite")
    log.info(f"  KW H={kw.get('H'):.3f}  p={kw.get('p'):.4g}  eta2={kw.get('eta2'):.3f}  ({kw.get('effect')})")

    posthoc = dunn_posthoc(df, "intervener_role", "prosociality_composite")

    desc = descriptive_table(df, "intervener_role")

    # Cross-tab: intervener_role × intervention_type mean composite
    cross = (
        df.groupby(["intervener_role", "intervention_type"])["prosociality_composite"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )
    cross.columns = ["intervener_role", "intervention_type", "mean_composite", "std_composite", "n"]

    # Per-dimension KW by intervener_role
    dim_kw = {dim: kruskal_test(df, "intervener_role", dim) for dim in DIMENSIONS}

    results = {
        "kw_composite":    kw,
        "kw_per_dimension": dim_kw,
        "posthoc":         posthoc.to_dict(orient="records"),
        "group_medians":   df.groupby("intervener_role")["prosociality_composite"]
                             .median().round(3).to_dict(),
    }
    return results, desc, posthoc, cross


# ---------------------------------------------------------------------------
# RQ3: Intervention timing
# ---------------------------------------------------------------------------

def rq3_timing(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    KW test + Dunn post-hoc for response_time_category × prosociality_composite.
    Also Spearman for raw response_time_minutes.
    """
    log.info("RQ3: Intervention timing × prosociality composite")

    # Filter to records with response times
    df_rt = df[df["response_time_minutes"].notna() & (df["response_time_minutes"] > 0)].copy()
    log.info(f"  Records with response time: {len(df_rt):,}")

    spear_rt = spearman_corr(df_rt, "response_time_minutes", "prosociality_composite")
    log.info(f"  Spearman (response_time × composite): rho={spear_rt.get('rho'):.3f}  p={spear_rt.get('p'):.4g}")

    kw = kruskal_test(df_rt, "response_time_category", "prosociality_composite")
    posthoc = dunn_posthoc(df_rt, "response_time_category", "prosociality_composite")

    desc = descriptive_table(df_rt, "response_time_category")

    results = {
        "spearman_response_time_minutes_x_composite": spear_rt,
        "kw_response_time_category": kw,
        "group_medians": df_rt.groupby("response_time_category")["prosociality_composite"]
                              .median().round(3).to_dict(),
        "posthoc": posthoc.to_dict(orient="records"),
    }
    return results, desc


# ---------------------------------------------------------------------------
# RQ4: Platform comparison
# ---------------------------------------------------------------------------

def rq4_platform(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    KW + Dunn post-hoc for platform × prosociality.
    Also cross-tabulation: platform × intervention_type mean composite.
    """
    log.info("RQ4: Platform comparison")
    kw = kruskal_test(df, "platform", "prosociality_composite")
    if "error" in kw:
        log.info(f"  Skipped: {kw['error']}")
    else:
        log.info(f"  KW H={kw.get('H'):.3f}  p={kw.get('p'):.4g}  eta2={kw.get('eta2'):.3f}  ({kw.get('effect')})")

    posthoc = dunn_posthoc(df, "platform", "prosociality_composite")

    desc = descriptive_table(df, "platform")

    # Cross-tab: platform × intervention_type mean composite
    cross = (
        df.groupby(["platform", "intervention_type"])["prosociality_composite"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )
    cross.columns = ["platform", "intervention_type", "mean_composite", "std_composite", "n"]

    # Per-dimension KW by platform
    dim_kw = {dim: kruskal_test(df, "platform", dim) for dim in DIMENSIONS}

    results = {
        "kw_composite": kw,
        "kw_per_dimension": dim_kw,
        "posthoc": posthoc.to_dict(orient="records"),
        "group_medians": df.groupby("platform")["prosociality_composite"]
                           .median().round(3).to_dict(),
    }
    return results, desc, cross


# ---------------------------------------------------------------------------
# RQ5: Dimension structure
# ---------------------------------------------------------------------------

def rq5_dimensions(df: pd.DataFrame) -> dict:
    """
    Pearson correlations among four dimensions + PCA to assess whether
    dimensions collapse to a single factor.
    """
    log.info("RQ5: Dimension structure (correlations + PCA)")

    # Pairwise Pearson correlations
    dim_df = df[DIMENSIONS].dropna()
    corr_matrix = dim_df.corr(method="pearson").round(4)

    # PCA
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(dim_df)
        pca = PCA()
        pca.fit(X)
        explained = pca.explained_variance_ratio_.tolist()
        loadings = pd.DataFrame(
            pca.components_.T,
            index=DIMENSIONS,
            columns=[f"PC{i+1}" for i in range(len(DIMENSIONS))],
        ).round(4)
        pca_available = True
    except ImportError:
        log.warning("scikit-learn not found — PCA skipped.  Install with: pip install scikit-learn")
        explained  = []
        loadings   = pd.DataFrame()
        pca_available = False

    # Flag if first PC explains > 70% variance (suggests single-factor structure)
    single_factor = len(explained) > 0 and explained[0] > 0.70

    log.info(f"  n records: {len(dim_df):,}")
    if pca_available:
        log.info(f"  PC1 explains {explained[0]*100:.1f}% of variance")
        if single_factor:
            log.info("  NOTE: PC1 > 70% — dimensions largely collapse to a single factor.")
            log.info("        Prosociality composite is the primary analysis variable.")

    return {
        "n":                     len(dim_df),
        "correlation_matrix":    corr_matrix.to_dict(),
        "pca_available":         pca_available,
        "pca_explained_variance": [round(v, 4) for v in explained],
        "pca_single_factor_flag": single_factor,
        "pca_loadings":          loadings.to_dict() if not loadings.empty else {},
        "note": (
            "High inter-dimension correlations (0.79-0.95) were observed in Phase 1. "
            "If PC1 > 70%, report prosociality_composite as the primary outcome "
            "and subscores as secondary descriptive findings."
        ),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def format_report(results: dict, df: pd.DataFrame) -> str:
    W = 78
    lines = [
        "=" * W,
        "  Phase 2 Pattern Analysis Report",
        "  Prosocial Corrective Communication in Online Communities",
        "=" * W,
        f"  Records analyzed:  {len(df):,}",
        f"  Platforms:         {', '.join(sorted(df['platform'].unique()))}",
        f"  Intervention types:{', '.join(sorted(df['intervention_type'].unique()))}",
        f"  Intervener roles:  {', '.join(sorted(df['intervener_role'].unique())) if 'intervener_role' in df.columns else 'n/a'}",
        f"  Date range:        {df['timestamp'].min()} — {df['timestamp'].max()}"
            if "timestamp" in df.columns else "",
        "",
        "  NOTE: All findings are associations, not causal claims.  Selection bias",
        "  and survivorship bias apply (see PROCESS.md Part 10.2 for framing guidance).",
        "",
    ]

    # ---- Descriptive by platform ----
    lines += ["  Prosociality Composite — Platform Means", "  " + "-" * (W - 2)]
    for platform, grp in df.groupby("platform"):
        vals = grp["prosociality_composite"].dropna()
        lines.append(
            f"  {platform:15s}  n={len(vals):>6,}  "
            f"mean={vals.mean():.3f}  sd={vals.std():.3f}  median={vals.median():.3f}"
        )
    lines.append("")

    # ---- RQ1 ----
    rq1 = results.get("rq1", {})
    kw = rq1.get("kw_composite", {})
    lines += [
        "  RQ1 — Intervention Type × Prosociality Composite",
        "  " + "-" * (W - 2),
        f"  Kruskal-Wallis: H({kw.get('k',0)-1}) = {kw.get('H','?'):.3f},  "
        f"p = {kw.get('p','?'):.4g} {_sig(kw.get('p', 1.0))},  "
        f"eta2 = {kw.get('eta2','?'):.3f} ({kw.get('effect','?')})",
        "",
        f"  {'Type':25s}  {'n':>6s}  {'Median':>7s}  {'Mean':>7s}  {'SD':>6s}",
        "  " + "-" * (W - 2),
    ]
    for itype, grp in df.groupby("intervention_type"):
        vals = grp["prosociality_composite"].dropna()
        lines.append(
            f"  {itype:25s}  {len(vals):>6,}  {vals.median():>7.3f}  "
            f"{vals.mean():>7.3f}  {vals.std():>6.3f}"
        )
    lines.append("")

    # ---- RQ2b ----
    rq2b = results.get("rq2b", {})
    kw2b = rq2b.get("kw_composite", {})
    lines += [
        "  RQ2b -- Intervener Role x Prosociality Composite",
        "  " + "-" * (W - 2),
    ]
    if "error" in rq2b:
        lines.append(f"  SKIPPED: {rq2b['error']}")
    else:
        lines += [
            f"  Kruskal-Wallis: H({kw2b.get('k',0)-1}) = {kw2b.get('H','?'):.3f},  "
            f"p = {kw2b.get('p','?'):.4g} {_sig(kw2b.get('p', 1.0))},  "
            f"eta2 = {kw2b.get('eta2','?'):.3f} ({kw2b.get('effect','?')})",
            "",
            f"  {'Role':20s}  {'n':>6s}  {'Median':>7s}  {'Mean':>7s}  {'SD':>6s}",
            "  " + "-" * (W - 2),
        ]
        if "intervener_role" in df.columns:
            for role, grp in df.groupby("intervener_role"):
                vals = grp["prosociality_composite"].dropna()
                lines.append(
                    f"  {role:20s}  {len(vals):>6,}  {vals.median():>7.3f}  "
                    f"{vals.mean():>7.3f}  {vals.std():>6.3f}"
                )
    lines.append("")

    # ---- RQ4 ----
    rq4 = results.get("rq4", {})
    kw4 = rq4.get("kw_composite", {})
    lines += ["  RQ4 — Platform Comparison", "  " + "-" * (W - 2)]
    if "error" in kw4:
        lines.append(f"  Skipped: {kw4['error']}")
    else:
        lines.append(
            f"  Kruskal-Wallis: H({kw4.get('k',0)-1}) = {kw4.get('H',float('nan')):.3f},  "
            f"p = {kw4.get('p',float('nan')):.4g} {_sig(kw4.get('p', 1.0))},  "
            f"eta2 = {kw4.get('eta2',float('nan')):.3f} ({kw4.get('effect','?')})"
        )
    lines += [
        "",
        f"  {'Platform':15s}  {'n':>6s}  {'Median':>7s}  {'Mean':>7s}  {'SD':>6s}",
        "  " + "-" * (W - 2),
    ]
    for platform, grp in df.groupby("platform"):
        vals = grp["prosociality_composite"].dropna()
        lines.append(
            f"  {platform:15s}  {len(vals):>6,}  {vals.median():>7.3f}  "
            f"{vals.mean():>7.3f}  {vals.std():>6.3f}"
        )
    lines.append("")

    # ---- RQ3 ----
    rq3 = results.get("rq3", {})
    spear = rq3.get("spearman_response_time_minutes_x_composite", {})
    lines += [
        "  RQ3 — Response Time × Prosociality",
        "  " + "-" * (W - 2),
        f"  Spearman rho = {spear.get('rho','?'):.3f},  "
        f"p = {spear.get('p','?'):.4g} {_sig(spear.get('p', 1.0))},  "
        f"n = {spear.get('n','?'):,}",
        "",
    ]

    # ---- RQ5 ----
    rq5 = results.get("rq5", {})
    expl = rq5.get("pca_explained_variance", [])
    lines += [
        "  RQ5 — Dimension Structure",
        "  " + "-" * (W - 2),
    ]
    if expl:
        lines.append(f"  PCA explained variance: " + "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(expl)))
        if rq5.get("pca_single_factor_flag"):
            lines.append("  WARNING: PC1 > 70% — dimensions largely collapse to a single factor.")
            lines.append("           Use prosociality_composite as primary outcome in analysis.")
    else:
        lines.append("  PCA not available (scikit-learn not installed).")

    corr = rq5.get("correlation_matrix", {})
    if corr:
        lines += ["", f"  Pearson correlations among Tier 1 dimensions:"]
        lines.append(f"  {'':20s}" + "".join(f"  {d[:8]:>8s}" for d in DIMENSIONS))
        for d1 in DIMENSIONS:
            row_str = f"  {d1:20s}"
            for d2 in DIMENSIONS:
                v = corr.get(d1, {}).get(d2, float("nan"))
                row_str += f"  {v:>8.3f}"
            lines.append(row_str)
    lines.append("")

    # ---- Pending analyses ----
    lines += [
        "  Analyses Pending Additional Data",
        "  " + "-" * (W - 2),
        "  - intervention_style × prosociality (requires extending human annotation to ~100K)",
        "  - user_retention_7day / user_retention_30day (requires full activity timeline)",
        "  - immediate_behavioral_change / sustained_behavior_score",
        "    (requires multiple post-intervention texts per user)",
        "  - user_account_age_days (requires account creation date from raw platform data)",
        "  - pre/post toxicity scores (requires detoxify or Perspective API)",
        "",
        "=" * W,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 quantitative analysis of intervention patterns."
    )
    parser.add_argument(
        "--input", default=str(PHASE2_DIR / "phase2_analysis_dataset.parquet"),
        help="Analysis dataset (default: output/phase2/phase2_analysis_dataset.parquet)",
    )
    parser.add_argument(
        "--output-dir", default=str(ARTIFACTS_DIR),
        help=f"Output directory for reports (default: {ARTIFACTS_DIR})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        log.error("Run src/build_phase2_features.py first.")
        sys.exit(1)

    log.info("=== Phase 2 Analysis ===")
    df = pd.read_parquet(input_path)
    log.info(f"Loaded {len(df):,} records")

    # Ensure timestamp is datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ------------------------------------------------------------------
    # Run analyses
    # ------------------------------------------------------------------
    results = {}

    rq1_stats, rq1_desc, rq1_posthoc = rq1_intervention_type(df)
    results["rq1"] = rq1_stats

    results["rq2"] = rq2_user_history(df)

    rq2b_stats, rq2b_desc, rq2b_posthoc, rq2b_cross = rq2b_intervener_role(df)
    results["rq2b"] = rq2b_stats

    rq3_stats, rq3_desc = rq3_timing(df)
    results["rq3"] = rq3_stats

    rq4_stats, rq4_desc, rq4_cross = rq4_platform(df)
    results["rq4"] = rq4_stats

    results["rq5"] = rq5_dimensions(df)

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "phase2_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    descriptive_table(df, "platform").to_csv(
        tables_dir / "descriptive_by_platform.csv", index=False
    )
    descriptive_table(df, "intervention_type").to_csv(
        tables_dir / "descriptive_by_intervention_type.csv", index=False
    )
    if "intervener_role" in df.columns:
        descriptive_table(df, "intervener_role").to_csv(
            tables_dir / "descriptive_by_intervener_role.csv", index=False
        )
    rq1_desc.to_csv(tables_dir / "rq1_intervention_type_prosociality.csv", index=False)
    rq1_posthoc.to_csv(tables_dir / "rq1_posthoc.csv", index=False)
    if not rq2b_desc.empty:
        rq2b_desc.to_csv(tables_dir / "rq2b_intervener_role_prosociality.csv", index=False)
        rq2b_posthoc.to_csv(tables_dir / "rq2b_posthoc.csv", index=False)
        rq2b_cross.to_csv(tables_dir / "rq2b_role_x_type_crosstab.csv", index=False)
    rq3_desc.to_csv(tables_dir / "rq3_timing_analysis.csv", index=False)
    rq4_desc.to_csv(tables_dir / "rq4_platform_comparison.csv", index=False)
    rq4_cross.to_csv(tables_dir / "rq4_platform_x_type_crosstab.csv", index=False)

    corr_df = df[DIMENSIONS].dropna().corr(method="pearson").round(4)
    corr_df.to_csv(tables_dir / "dimension_correlations.csv")

    if results["rq5"].get("pca_available") and results["rq5"].get("pca_loadings"):
        pd.DataFrame(results["rq5"]["pca_loadings"]).to_csv(
            tables_dir / "dimension_pca.csv"
        )

    log.info(f"Tables saved → {tables_dir}/")

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------
    report_txt = format_report(results, df)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path  = out_dir / "phase2_report.txt"
    json_path = out_dir / "phase2_report.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # Make results JSON-serialisable
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, pd.DataFrame):   return obj.to_dict(orient="records")
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_safe)

    log.info(f"Report (text) → {txt_path}")
    log.info(f"Report (JSON) → {json_path}")
    print("\n" + report_txt)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
