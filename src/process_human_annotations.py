"""
src/process_human_annotations.py
=================================
Process completed human annotation rating forms, compute inter-rater reliability,
flag disagreements for adjudication, and produce consensus training/validation splits.

Usage
-----
  # First pass — validate + IRR report + produce adjudication file (if needed)
  python src/process_human_annotations.py \\
      --rater1 human-validation/ratings_rater1.csv \\
      --rater2 human-validation/ratings_rater2.csv \\
      --rater3 human-validation/ratings_rater3.csv

  # Second pass — with adjudicated overrides applied
  python src/process_human_annotations.py \\
      --rater1 human-validation/ratings_rater1.csv \\
      --rater2 human-validation/ratings_rater2.csv \\
      --rater3 human-validation/ratings_rater3.csv \\
      --adjudication output/human_labels/flagged_for_adjudication.csv

Outputs
-------
  output/human_labels/human_train.parquet            ~80% stratified split
  output/human_labels/human_val.parquet              ~20% stratified split
  output/human_labels/flagged_for_adjudication.csv   records needing adjudication (if any)
  output/artifacts/human_irr_report.txt              full IRR report
  output/artifacts/human_irr_report.json             machine-readable metrics

Inter-rater reliability metrics
--------------------------------
  ICC(2,1)     Two-way random effects, single rater, absolute agreement.
               Computed per Tier 1 dimension and as overall mean.
               Target: >= 0.75 (Koo & Mae, 2016).
  Fleiss' κ    For categorical intervention_style.
               Target: >= 0.60.
  % exact      Proportion of records where all 3 raters gave identical scores.
  % within 1   Proportion where max pairwise difference <= 1.0.

Adjudication
------------
  A record is flagged when any pair of raters differs by > 1.5 on any Tier 1
  dimension.  Flagged records are written to flagged_for_adjudication.csv with
  all three raters' scores visible.  Fill in the adj_* columns and re-run with
  --adjudication to incorporate the resolved scores into the final output.

Consensus scoring
-----------------
  Non-flagged records:  mean of three raters' scores, rounded to nearest 0.5.
  Flagged + adjudicated: adjudicated scores used directly.
  Flagged + no adjudication: mean used with a warning.
  intervention_style consensus: plurality vote; ties broken alphabetically.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
HUMAN_DIR     = ROOT / "output" / "human_labels"
ARTIFACTS_DIR = ROOT / "output" / "artifacts"
LOG_DIR       = ROOT / "logs"

for _d in [HUMAN_DIR, ARTIFACTS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "process_human_annotations.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS             = ["empathy", "constructiveness", "respect", "social_cohesion"]
VALID_SCORES           = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
VALID_STYLES           = {"punitive", "educative", "suggestive", "positive", "structural"}
DISAGREEMENT_THRESHOLD = 1.5   # flag if any rater pair differs by more than this


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def icc_2_1_k_raters(scores: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects, single rater, absolute agreement.

    scores: shape (n_subjects, k_raters) — rows with any NaN are dropped.
    Formula: (MS_subjects - MS_error) /
             (MS_subjects + (k-1)*MS_error + k*(MS_raters - MS_error)/n)
    Returns NaN if computation is degenerate.
    """
    data = np.asarray(scores, dtype=float)
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    if n < 3 or k < 2:
        return float("nan")

    grand_mean = data.mean()
    row_means  = data.mean(axis=1)
    col_means  = data.mean(axis=0)

    ss_subjects = k * np.sum((row_means - grand_mean) ** 2)
    ss_raters   = n * np.sum((col_means  - grand_mean) ** 2)
    ss_total    = np.sum((data - grand_mean) ** 2)
    ss_error    = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters   = ss_raters   / (k - 1)
    ms_error    = ss_error    / ((n - 1) * (k - 1))

    denom = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if denom == 0:
        return float("nan")
    return float((ms_subjects - ms_error) / denom)


def fleiss_kappa(ratings: list[list[str]], categories: list[str]) -> float:
    """
    Fleiss' kappa for k raters assigning one of C categories to N subjects.

    ratings:    list of N lists, each of length k (one value per rater).
    categories: all valid category labels.
    Returns NaN if computation is degenerate.
    """
    N = len(ratings)
    k = len(ratings[0]) if ratings else 0
    if N == 0 or k < 2:
        return float("nan")

    cat_index = {c: i for i, c in enumerate(categories)}
    C = len(categories)

    # n_ij: N x C — count of raters assigning category j to subject i
    n_ij = np.zeros((N, C), dtype=float)
    for i, row in enumerate(ratings):
        for r in row:
            if r in cat_index:
                n_ij[i, cat_index[r]] += 1

    P_j   = n_ij.sum(axis=0) / (N * k)          # marginal proportions
    p_i   = (np.sum(n_ij ** 2, axis=1) - k) / (k * (k - 1))  # per-subject agreement
    P_bar = p_i.mean()
    P_e   = np.sum(P_j ** 2)

    if (1 - P_e) == 0:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def _round_half(x: float) -> float:
    """Round to nearest 0.5 on the 1.0–5.0 scale."""
    return round(x * 2) / 2


def _plurality_vote(values: list[str]) -> str:
    """Most common value; ties broken alphabetically."""
    counts: dict[str, int] = {}
    for v in values:
        if isinstance(v, str) and v.strip():
            key = v.strip().lower()
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    max_count = max(counts.values())
    return sorted(k for k, n in counts.items() if n == max_count)[0]


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def load_ratings(path: str | Path, rater_label: str) -> pd.DataFrame:
    """Load one completed rating CSV, validate values, return clean DataFrame."""
    path = Path(path)
    if not path.exists():
        log.error(f"Rating file not found: {path}")
        sys.exit(1)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"record_id"} | set(DIMENSIONS) | {"intervention_style"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        log.error(f"{rater_label}: missing required columns: {missing_cols}")
        sys.exit(1)

    # Normalise whitespace and replace blank/nan strings with NaN
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

    # rater_id — fill missing with the label we derived from the filename
    if "rater_id" not in df.columns:
        df["rater_id"] = rater_label
    else:
        n_blank = df["rater_id"].isna().sum()
        if n_blank > 0:
            log.warning(
                f"{rater_label}: {n_blank} rows have no rater_id — filling with '{rater_label}'"
            )
            df["rater_id"] = df["rater_id"].fillna(rater_label)

    # Numeric Tier 1 dimensions
    n_errors = 0
    for dim in DIMENSIONS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
        bad_mask = df[dim].notna() & df[dim].apply(lambda x: x not in VALID_SCORES)
        if bad_mask.any():
            log.warning(
                f"{rater_label}: {dim} — {bad_mask.sum()} out-of-range values "
                f"(valid: 1.0–5.0 in 0.5 steps):\n"
                + df.loc[bad_mask, ["record_id", dim]].to_string(index=False)
            )
            n_errors += bad_mask.sum()
        n_missing = df[dim].isna().sum()
        if n_missing:
            log.warning(f"{rater_label}: {dim} — {n_missing} missing values")

    # Categorical intervention_style
    df["intervention_style"] = df["intervention_style"].str.lower()
    bad_style = df["intervention_style"].notna() & df["intervention_style"].apply(
        lambda x: x not in VALID_STYLES
    )
    if bad_style.any():
        log.warning(
            f"{rater_label}: intervention_style — {bad_style.sum()} invalid values "
            f"(valid: {sorted(VALID_STYLES)}):\n"
            + df.loc[bad_style, ["record_id", "intervention_style"]].to_string(index=False)
        )
        n_errors += bad_style.sum()

    log.info(
        f"{rater_label}: loaded {len(df)} records  |  validation errors: {n_errors}"
    )
    df["_rater_label"] = rater_label
    return df


# ---------------------------------------------------------------------------
# Disagreement detection
# ---------------------------------------------------------------------------

def find_disagreements(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    threshold: float = DISAGREEMENT_THRESHOLD,
) -> list[str]:
    """
    Return sorted list of record_ids where any rater pair differs by > threshold
    on any Tier 1 dimension.
    """
    flagged: set[str] = set()

    # Build a wide frame aligned on record_id
    merged = (
        df1[["record_id"] + DIMENSIONS]
        .merge(df2[["record_id"] + DIMENSIONS], on="record_id", suffixes=("_r1", "_r2"))
        .merge(df3[["record_id"] + DIMENSIONS], on="record_id")
    )
    # df3 columns arrive without a suffix — rename them
    merged.rename(columns={dim: f"{dim}_r3" for dim in DIMENSIONS}, inplace=True)

    for dim in DIMENSIONS:
        c1 = merged[f"{dim}_r1"].to_numpy(dtype=float)
        c2 = merged[f"{dim}_r2"].to_numpy(dtype=float)
        c3 = merged[f"{dim}_r3"].to_numpy(dtype=float)
        for a, b in [(c1, c2), (c1, c3), (c2, c3)]:
            diff = np.abs(a - b)
            flagged.update(merged.loc[diff > threshold, "record_id"].tolist())

    return sorted(flagged)


# ---------------------------------------------------------------------------
# Adjudication export
# ---------------------------------------------------------------------------

def build_adjudication_csv(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    flagged_ids: list[str],
    path: Path,
) -> None:
    """
    Write a CSV showing all three raters' scores for flagged records, plus
    blank adj_* columns for the researcher to fill in.
    """
    cols = ["record_id"] + DIMENSIONS + ["intervention_style"]
    r1 = df1[df1["record_id"].isin(flagged_ids)][cols].copy()
    r2 = df2[df2["record_id"].isin(flagged_ids)][cols].copy()
    r3 = df3[df3["record_id"].isin(flagged_ids)][cols].copy()

    wide = r1.merge(r2, on="record_id", suffixes=("_r1", "_r2"))
    wide = wide.merge(r3, on="record_id")
    wide.rename(
        columns={
            **{dim: f"{dim}_r3" for dim in DIMENSIONS},
            "intervention_style": "intervention_style_r3",
        },
        inplace=True,
    )

    # Blank adjudication columns
    for dim in DIMENSIONS:
        wide[f"adj_{dim}"] = ""
    wide["adj_intervention_style"] = ""
    wide["adjudicator_notes"] = ""

    wide.to_csv(path, index=False)
    log.info(f"Adjudication file saved → {path}")
    log.info(f"  Fill in adj_* columns and re-run with --adjudication {path}")


# ---------------------------------------------------------------------------
# Consensus scoring
# ---------------------------------------------------------------------------

def compute_consensus(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    flagged_ids: list[str],
    df_adj: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Produce one consensus row per record_id.

    Tier 1 scores: mean of three raters, rounded to nearest 0.5.
    Flagged + adjudicated: adj_* values override the mean.
    Flagged + no adjudication: mean used with a warning per record.
    intervention_style: plurality vote; adj_intervention_style overrides if present.
    """
    # Build wide table
    cols = ["record_id"] + DIMENSIONS + ["intervention_style"]
    wide = (
        df1[cols]
        .merge(df2[cols], on="record_id", suffixes=("_r1", "_r2"))
        .merge(df3[cols], on="record_id")
    )
    wide.rename(
        columns={
            **{dim: f"{dim}_r3" for dim in DIMENSIONS},
            "intervention_style": "intervention_style_r3",
        },
        inplace=True,
    )

    flagged_set = set(flagged_ids)
    adj_index   = df_adj.set_index("record_id") if df_adj is not None else None

    rows = []
    for _, row in wide.iterrows():
        rid        = row["record_id"]
        is_flagged = rid in flagged_set

        # Tier 1 consensus
        scores: dict[str, float] = {}
        for dim in DIMENSIONS:
            vals = [row[f"{dim}_r1"], row[f"{dim}_r2"], row[f"{dim}_r3"]]
            clean = [float(v) for v in vals if pd.notna(v)]
            scores[dim] = _round_half(float(np.mean(clean))) if clean else float("nan")

        # Intervention style consensus
        style_vals = [
            row.get("intervention_style_r1"),
            row.get("intervention_style_r2"),
            row.get("intervention_style_r3"),
        ]
        style = _plurality_vote([v for v in style_vals if pd.notna(v)])

        # Adjudication overrides
        if is_flagged and adj_index is not None:
            if rid in adj_index.index:
                adj_row = adj_index.loc[rid]
                for dim in DIMENSIONS:
                    col = f"adj_{dim}"
                    if col in adj_index.columns:
                        v = pd.to_numeric(adj_row[col], errors="coerce")
                        if pd.notna(v):
                            scores[dim] = float(v)
                if "adj_intervention_style" in adj_index.columns:
                    v = adj_row.get("adj_intervention_style", "")
                    if isinstance(v, str) and v.strip().lower() in VALID_STYLES:
                        style = v.strip().lower()
            else:
                log.warning(
                    f"Flagged record {rid} not found in adjudication file — using rater mean."
                )
        elif is_flagged:
            log.warning(
                f"Flagged record {rid} has no adjudication override — using rater mean."
            )

        out = {
            "record_id":                rid,
            "flagged":                  is_flagged,
            "intervention_style_consensus": style,
        }
        out.update(scores)
        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IRR report
# ---------------------------------------------------------------------------

_ICC_BENCHMARKS   = "<0.50 poor | 0.50-0.75 moderate | 0.75-0.90 good | >0.90 excellent"
_KAPPA_BENCHMARKS = (
    "<0.20 slight | 0.21-0.40 fair | 0.41-0.60 moderate | "
    "0.61-0.80 substantial | >0.80 near-perfect"
)


def compute_irr_metrics(
    df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame
) -> dict:
    """Compute ICC per dimension, mean ICC, Fleiss' kappa, and pairwise agreement stats."""
    common_ids = sorted(set(df1["record_id"]) & set(df2["record_id"]) & set(df3["record_id"]))
    d = {
        "r1": df1.set_index("record_id"),
        "r2": df2.set_index("record_id"),
        "r3": df3.set_index("record_id"),
    }

    # ICC per dimension
    icc_per_dim: dict[str, float] = {}
    for dim in DIMENSIONS:
        matrix = np.column_stack(
            [d[r].reindex(common_ids)[dim].to_numpy(dtype=float) for r in ("r1", "r2", "r3")]
        )
        icc_per_dim[dim] = round(icc_2_1_k_raters(matrix), 4)
    mean_icc = round(float(np.nanmean(list(icc_per_dim.values()))), 4)

    # Pairwise agreement (% exact and % within 1.0, averaged across dimensions)
    pairwise: dict[str, dict] = {}
    for ra, rb in [("r1", "r2"), ("r1", "r3"), ("r2", "r3")]:
        exact_list, within1_list = [], []
        for dim in DIMENSIONS:
            a = d[ra].reindex(common_ids)[dim].to_numpy(dtype=float)
            b = d[rb].reindex(common_ids)[dim].to_numpy(dtype=float)
            mask = ~(np.isnan(a) | np.isnan(b))
            if not mask.any():
                continue
            exact_list.append((a[mask] == b[mask]).mean())
            within1_list.append((np.abs(a[mask] - b[mask]) <= 1.0).mean())
        pairwise[f"{ra}_{rb}"] = {
            "pct_exact":   round(float(np.mean(exact_list)),   4) if exact_list   else float("nan"),
            "pct_within1": round(float(np.mean(within1_list)), 4) if within1_list else float("nan"),
        }

    # Fleiss' kappa for intervention_style
    style_triples = [
        [
            str(d["r1"].loc[rid, "intervention_style"]) if rid in d["r1"].index else "",
            str(d["r2"].loc[rid, "intervention_style"]) if rid in d["r2"].index else "",
            str(d["r3"].loc[rid, "intervention_style"]) if rid in d["r3"].index else "",
        ]
        for rid in common_ids
    ]
    style_complete = [
        row for row in style_triples
        if all(v.lower() in VALID_STYLES for v in row)
    ]
    kappa = (
        round(fleiss_kappa(style_complete, sorted(VALID_STYLES)), 4)
        if len(style_complete) >= 3
        else float("nan")
    )

    return {
        "n_records":           len(common_ids),
        "icc_per_dimension":   icc_per_dim,
        "mean_icc":            mean_icc,
        "fleiss_kappa_style":  kappa,
        "n_style_complete":    len(style_complete),
        "pairwise_agreement":  pairwise,
    }


def _icc_label(v: float) -> str:
    if np.isnan(v):          return "(insufficient data)"
    if v < 0.50:             return "poor"
    if v < 0.75:             return "moderate"
    if v < 0.90:             return "good"
    return                          "excellent"


def _kappa_label(v: float) -> str:
    if np.isnan(v):          return "(insufficient data)"
    if v < 0.20:             return "slight"
    if v < 0.40:             return "fair"
    if v < 0.60:             return "moderate"
    if v < 0.80:             return "substantial"
    return                          "near-perfect"


def format_irr_report(
    metrics: dict,
    n_flagged: int,
    flagged_ids: list[str],
    role_counts: dict | None = None,
) -> str:
    W = 72
    lines = [
        "=" * W,
        "  Human Inter-Rater Reliability Report",
        "  Prosocial Corrective Communication in Online Communities",
        "=" * W,
        f"  Records rated (common to all 3 raters): {metrics['n_records']}",
        f"  Flagged for adjudication (>1.5 on any dim): "
        f"{n_flagged}  ({n_flagged / max(metrics['n_records'], 1) * 100:.1f}%)",
        "",
        "  Tier 1 — ICC(2,1)  [two-way random effects, single rater, absolute agreement]",
        "  " + "-" * (W - 2),
        f"  {'Dimension':20s}  {'ICC(2,1)':>8s}  Interpretation",
        "  " + "-" * (W - 2),
    ]
    for dim in DIMENSIONS:
        v = metrics["icc_per_dimension"][dim]
        flag = " ✓" if (not np.isnan(v) and v >= 0.75) else ""
        lines.append(f"  {dim:20s}  {v:>8.3f}  {_icc_label(v)}{flag}")

    micc = metrics["mean_icc"]
    target = "✓ TARGET MET (>= 0.75)" if micc >= 0.75 else "✗ BELOW TARGET (>= 0.75)"
    lines += [
        "  " + "-" * (W - 2),
        f"  {'Mean ICC':20s}  {micc:>8.3f}  {target}",
        "",
        f"  Benchmarks: {_ICC_BENCHMARKS}",
        "",
        "  Pairwise Agreement (mean across 4 dimensions)",
        "  " + "-" * (W - 2),
        f"  {'Pair':12s}  {'% Exact':>8s}  {'% Within 1.0':>12s}",
        "  " + "-" * (W - 2),
    ]
    for pair_key, vals in metrics["pairwise_agreement"].items():
        lines.append(
            f"  {pair_key:12s}  {vals['pct_exact'] * 100:>7.1f}%  "
            f"{vals['pct_within1'] * 100:>11.1f}%"
        )

    kappa = metrics["fleiss_kappa_style"]
    ktarget = "✓ TARGET MET (>= 0.60)" if (not np.isnan(kappa) and kappa >= 0.60) else (
        "✗ BELOW TARGET (>= 0.60)" if not np.isnan(kappa) else ""
    )
    lines += [
        "",
        "  Intervention Style — Fleiss' Kappa",
        "  " + "-" * (W - 2),
        f"  Records with complete style ratings: {metrics['n_style_complete']}",
        f"  Fleiss' kappa = {kappa:.3f}  ({_kappa_label(kappa)})  {ktarget}",
        "",
        f"  Benchmarks: {_KAPPA_BENCHMARKS}",
        "",
        "=" * W,
    ]
    if role_counts:
        total = sum(role_counts.values())
        lines += [
            "",
            "  Intervener Role Distribution",
            "  " + "-" * (W - 2),
        ]
        for role, n in sorted(role_counts.items()):
            lines.append(f"  {role:25s}  {n:4d}  ({n / total * 100:.1f}%)")

    if flagged_ids:
        lines += ["", f"  Records flagged for adjudication ({n_flagged}):"]
        for rid in flagged_ids:
            lines.append(f"    {rid}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame, val_frac: float = 0.20, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split by platform (prefix of record_id).
    Strata with < 3 records go entirely to train.
    """
    df = df.copy()
    df["_platform"] = df["record_id"].str.split("_").str[0]

    train_ids: list[str] = []
    val_ids:   list[str] = []

    for _, grp in df.groupby("_platform"):
        if len(grp) < 3:
            train_ids.extend(grp["record_id"].tolist())
            continue
        n_val      = max(1, round(len(grp) * val_frac))
        val_sample = grp.sample(n=n_val, random_state=seed)
        train_ids.extend(grp.drop(val_sample.index)["record_id"].tolist())
        val_ids.extend(val_sample["record_id"].tolist())

    train = df[df["record_id"].isin(train_ids)].drop(columns=["_platform"]).reset_index(drop=True)
    val   = df[df["record_id"].isin(val_ids)].drop(columns=["_platform"]).reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process human annotation forms, compute inter-rater reliability, "
            "and produce consensus train/val splits."
        )
    )
    parser.add_argument("--rater1", required=True,
                        help="Path to rater 1's completed rating_form.csv")
    parser.add_argument("--rater2", required=True,
                        help="Path to rater 2's completed rating_form.csv")
    parser.add_argument("--rater3", required=True,
                        help="Path to rater 3's completed rating_form.csv")
    parser.add_argument(
        "--adjudication", default=None,
        help=(
            "Optional: path to completed adjudication CSV "
            "(the flagged_for_adjudication.csv from a prior run, with adj_* columns filled in)."
        ),
    )
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="Fraction of consensus records to hold for validation (default: 0.20).")
    parser.add_argument("--threshold", type=float, default=DISAGREEMENT_THRESHOLD,
                        help=f"Pairwise disagreement threshold for adjudication (default: {DISAGREEMENT_THRESHOLD}).")
    parser.add_argument("--output-dir", default=str(HUMAN_DIR),
                        help=f"Output directory for parquet files (default: {HUMAN_DIR}).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Process Human Annotations ===")

    # ------------------------------------------------------------------
    # Load and validate rating files
    # ------------------------------------------------------------------
    df1 = load_ratings(args.rater1, "rater1")
    df2 = load_ratings(args.rater2, "rater2")
    df3 = load_ratings(args.rater3, "rater3")

    ids1, ids2, ids3 = set(df1["record_id"]), set(df2["record_id"]), set(df3["record_id"])
    common = ids1 & ids2 & ids3

    if len(common) < len(ids1 | ids2 | ids3):
        log.warning(
            f"Record ID mismatch across raters: "
            f"{len(ids1)} / {len(ids2)} / {len(ids3)} records, "
            f"{len(common)} in common.  Only common records are included."
        )
    log.info(f"Records common to all 3 raters: {len(common)}")

    # Align all three DataFrames to the same sorted order
    id_order = sorted(common)
    for df_ref, label in [(df1, "rater1"), (df2, "rater2"), (df3, "rater3")]:
        df_ref.drop(df_ref[~df_ref["record_id"].isin(common)].index, inplace=True)
    df1 = df1.set_index("record_id").loc[id_order].reset_index()
    df2 = df2.set_index("record_id").loc[id_order].reset_index()
    df3 = df3.set_index("record_id").loc[id_order].reset_index()

    # ------------------------------------------------------------------
    # Inter-rater reliability
    # ------------------------------------------------------------------
    log.info("Computing inter-rater reliability metrics...")
    metrics = compute_irr_metrics(df1, df2, df3)

    log.info(f"  Mean ICC(2,1): {metrics['mean_icc']:.3f}")
    for dim, icc in metrics["icc_per_dimension"].items():
        log.info(f"    {dim:20s}: ICC = {icc:.3f}")
    log.info(f"  Fleiss' kappa (intervention_style): {metrics['fleiss_kappa_style']:.3f}")

    if metrics["mean_icc"] < 0.75:
        log.warning(
            "Mean ICC is below the 0.75 target.  "
            "Consider revising the rubric and re-rating a subset of records before proceeding."
        )

    # ------------------------------------------------------------------
    # Disagreement flagging
    # ------------------------------------------------------------------
    log.info(f"Flagging records with any pairwise difference > {args.threshold}...")
    flagged_ids = find_disagreements(df1, df2, df3, threshold=args.threshold)
    log.info(
        f"  {len(flagged_ids)} records flagged "
        f"({len(flagged_ids) / max(len(common), 1) * 100:.1f}%)"
    )

    adj_flag_path = out_dir / "flagged_for_adjudication.csv"
    if flagged_ids:
        build_adjudication_csv(df1, df2, df3, flagged_ids, adj_flag_path)

    # ------------------------------------------------------------------
    # Adjudication overrides
    # ------------------------------------------------------------------
    df_adj: pd.DataFrame | None = None
    if args.adjudication:
        adj_path = Path(args.adjudication)
        if not adj_path.exists():
            log.error(f"Adjudication file not found: {adj_path}")
            sys.exit(1)
        df_adj = pd.read_csv(adj_path, dtype=str)
        df_adj.columns = [c.strip().lower() for c in df_adj.columns]
        df_adj.replace({"": np.nan, "nan": np.nan}, inplace=True)
        log.info(f"Loaded adjudication overrides: {len(df_adj)} rows from {adj_path.name}")

    # ------------------------------------------------------------------
    # Consensus scoring
    # ------------------------------------------------------------------
    log.info("Computing consensus scores...")
    df_consensus = compute_consensus(df1, df2, df3, flagged_ids, df_adj)
    log.info(f"  Consensus records: {len(df_consensus)}")
    for dim in DIMENSIONS:
        vals = df_consensus[dim].dropna()
        log.info(
            f"  {dim:20s}: mean={vals.mean():.2f}  std={vals.std():.2f}  "
            f"min={vals.min():.1f}  max={vals.max():.1f}"
        )

    # ------------------------------------------------------------------
    # Normalise record_ids: bl_ → reddit_, mn_ → stackexchange_, di_ → wikipedia_
    # The annotation forms used blinded prefixes to prevent raters from inferring
    # platform.  All downstream files (Claude scores, records_to_rate.csv) use
    # the canonical prefixes, so we normalise here before any join or save.
    # ------------------------------------------------------------------
    _prefix_map = {"bl_": "reddit_", "mn_": "stackexchange_", "di_": "wikipedia_"}

    def _norm_id(rid: str) -> str:
        for old, new in _prefix_map.items():
            if rid.startswith(old):
                return new + rid[len(old):]
        return rid

    df_consensus["record_id"] = df_consensus["record_id"].apply(_norm_id)
    log.info("Normalised record_ids: bl_ -> reddit_, mn_ -> stackexchange_, di_ -> wikipedia_")

    # ------------------------------------------------------------------
    # Enrich consensus with platform metadata (platform, intervention_type, intervener_role)
    # ------------------------------------------------------------------
    # Metadata join: use records_to_rate.csv (100% overlap with annotation records)
    # as primary source, fall back to labeling_sample.parquet for intervener_role.
    rtr_path   = ROOT / "human-validation" / "records_to_rate.csv"
    sample_path = ROOT / "output" / "processed" / "labeling_sample.parquet"
    role_counts: dict = {}

    if rtr_path.exists():
        rtr_meta = pd.read_csv(rtr_path, usecols=["record_id", "platform", "intervention_type"])
        df_consensus = df_consensus.merge(rtr_meta, on="record_id", how="left")
        n_matched = df_consensus["platform"].notna().sum()
        log.info(f"Joined platform/intervention_type from records_to_rate.csv ({n_matched}/{len(df_consensus)} matched)")
    else:
        log.warning(f"records_to_rate.csv not found at {rtr_path} — platform columns omitted")

    # intervener_role is only in labeling_sample (not in records_to_rate)
    if sample_path.exists():
        df_sample_full = pd.read_parquet(sample_path)
        if "intervener_role" in df_sample_full.columns:
            df_consensus = df_consensus.merge(
                df_sample_full[["record_id", "intervener_role"]], on="record_id", how="left"
            )
            if "intervener_role" in df_consensus.columns:
                role_counts = df_consensus["intervener_role"].value_counts().to_dict()
                log.info("Intervener role distribution in consensus set:")
                for role, n in sorted(role_counts.items()):
                    log.info(f"  {role}: {n}")
    else:
        log.warning(f"labeling_sample not found at {sample_path} — intervener_role omitted")

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    log.info(
        f"Splitting {len(df_consensus)} records "
        f"({1 - args.val_frac:.0%} train / {args.val_frac:.0%} val)..."
    )
    df_train, df_val = stratified_split(df_consensus, val_frac=args.val_frac)
    log.info(f"  train: {len(df_train)}   val: {len(df_val)}")

    train_path = out_dir / "human_train.parquet"
    val_path   = out_dir / "human_val.parquet"
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path,   index=False)
    log.info(f"  Saved → {train_path}")
    log.info(f"  Saved → {val_path}")

    # ------------------------------------------------------------------
    # IRR report
    # ------------------------------------------------------------------
    report_txt = format_irr_report(metrics, len(flagged_ids), flagged_ids, role_counts or None)
    report_dict = {
        **metrics,
        "n_flagged":   len(flagged_ids),
        "flagged_ids": flagged_ids,
        "n_train":     len(df_train),
        "n_val":       len(df_val),
        "threshold":   args.threshold,
    }

    txt_path  = ARTIFACTS_DIR / "human_irr_report.txt"
    json_path = ARTIFACTS_DIR / "human_irr_report.json"
    with open(txt_path,  "w", encoding="utf-8") as f:
        f.write(report_txt)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    log.info(f"IRR report (text) → {txt_path}")
    log.info(f"IRR report (JSON) → {json_path}")
    print("\n" + report_txt)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
