"""
src/analyze_tier_d.py
======================
Descriptive analysis of Tier D records — users who received an intervention
but produced no meaningful post-intervention response.

Tier D Operational Definition
------------------------------
A record is classified as Tier D if it passes the pre-intervention and
intervention-text quality filters (same thresholds as phase2_sample.py) but
fails the post-intervention text filter — i.e., post_intervention_text is
missing, a placeholder ("[deleted]", "[removed]"), or fewer than 30 characters.
This operationalizes the thesis definition: "at least one pre-intervention post
and no post-intervention posts."

Records failing the pre or intervention filter entirely (bots, placeholders,
very short texts) are excluded before classification — they represent data
quality issues, not the departure pattern of interest.

Analytical value
----------------
Tier D cannot support prosociality scoring (no response to score), but it
informs three research questions directly derivable from metadata:
  1. Departure rate: what fraction of users who received an intervention did
     not respond? Does this vary by platform and intervention type?
  2. Pre-intervention behavior: are Tier D users' pre-intervention texts
     shorter / more aggressive (caps abuse) than users who responded?
  3. Intervention distribution: does the mix of intervention types differ
     between Tier D and Tiers A-C?

Inputs
------
  output/processed/unified_interventions.parquet   full ingested pool
  output/processed/labeling_sample.parquet          Phase 1 IDs (excluded)
  output/phase2/phase2_sample.parquet               Phase 2 IDs (excluded)

Outputs
-------
  output/artifacts/tier_d_report.txt
  output/artifacts/tier_d_tables/
    tier_d_funnel.csv                 record counts at each filter stage
    tier_d_overall.csv                Tier D vs A-C counts + departure rate
    tier_d_by_platform.csv            departure rate by platform
    tier_d_by_intervention_type.csv   departure rate by intervention type
    tier_d_platform_x_type.csv        platform × type departure rate crosstab
    tier_d_pre_text_stats.csv         pre-text length/word count comparison
    tier_d_response_time.csv          response_time_minutes distribution (Tier D)

Usage
-----
  python src/analyze_tier_d.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "output" / "processed"
PHASE2    = ROOT / "output" / "phase2"
OUT_DIR   = ROOT / "output" / "artifacts" / "tier_d_tables"
LOG_DIR   = ROOT / "logs"

for _d in [OUT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "analyze_tier_d.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

# Match thresholds from phase2_sample.py exactly
MIN_PRE_LEN           = 50
MIN_INT_LEN           = 30
MIN_POST_LEN          = 30
MAX_TEXT_LEN          = 5_000
MAX_RESPONSE_MINUTES  = 365 * 24 * 60
_PLACEHOLDER_STRINGS  = {"[deleted]", "[removed]", ""}
_BOT_KEYWORDS         = ("bot", "automoderator", "spam", "moderationbot")

PERCENTILES = [10, 25, 50, 75, 90]


# ---------------------------------------------------------------------------
# Filter helpers (mirrors phase2_sample.py)
# ---------------------------------------------------------------------------

def _is_placeholder(text) -> bool:
    if not isinstance(text, str):
        return True
    return text.strip().lower() in _PLACEHOLDER_STRINGS


def _is_bot(user_id) -> bool:
    uid = str(user_id or "").lower()
    return any(k in uid for k in _BOT_KEYWORDS)


def _word_count(text) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{100 * n / total:.1f}%"


def _percentile_row(series: pd.Series, label: str) -> dict:
    clean = series.dropna()
    row = {"group": label, "n": len(clean), "mean": round(clean.mean(), 1),
           "sd": round(clean.std(), 1)}
    for p in PERCENTILES:
        row[f"p{p}"] = round(float(np.percentile(clean, p)), 1)
    return row


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(df_all: pd.DataFrame, known_ids: set) -> tuple[str, dict[str, pd.DataFrame]]:
    tables: dict[str, pd.DataFrame] = {}
    W = 78
    lines = [
        "=" * W,
        "  Tier D Analysis — Users With No Post-Intervention Response",
        "=" * W,
        "",
        "  Definition: records passing pre/intervention quality filters that",
        "  have no meaningful post-intervention text (absent, placeholder,",
        "  or < 30 characters).  These represent potential platform departures.",
        "",
    ]

    # ------------------------------------------------------------------
    # Stage 1: Exclude known-bad records (bot, response-time, text quality)
    #          These are data quality issues, not the departure pattern.
    # ------------------------------------------------------------------
    funnel_rows = [{"stage": "raw_pool", "n": len(df_all)}]

    # Bot filter
    df = df_all[~df_all["user_id"].apply(_is_bot)].copy()
    funnel_rows.append({"stage": "after_bot_filter", "n": len(df)})

    # Pre-text quality
    pre_ok = (
        ~df["pre_intervention_text"].apply(_is_placeholder)
        & (df["pre_intervention_text"].str.len() >= MIN_PRE_LEN)
        & (df["pre_intervention_text"].str.len() <= MAX_TEXT_LEN)
    )
    int_ok = (
        ~df["intervention_text"].apply(_is_placeholder)
        & (df["intervention_text"].str.len() >= MIN_INT_LEN)
        & (df["intervention_text"].str.len() <= MAX_TEXT_LEN)
    )
    df = df[pre_ok & int_ok].copy()
    funnel_rows.append({"stage": "after_pre_and_intervention_filter", "n": len(df)})

    # ------------------------------------------------------------------
    # Stage 2: Classify each remaining record as Tier D or Tiers A-C
    # ------------------------------------------------------------------
    post_text = df["post_intervention_text"]
    post_missing_mask = (
        post_text.isna()
        | post_text.apply(_is_placeholder)
        | (post_text.str.len() < MIN_POST_LEN)
    )

    df["tier"] = np.where(post_missing_mask, "tier_d", "tiers_ac")
    funnel_rows.append({"stage": "tier_d", "n": int(post_missing_mask.sum())})
    funnel_rows.append({"stage": "tiers_ac", "n": int((~post_missing_mask).sum())})

    tables["tier_d_funnel"] = pd.DataFrame(funnel_rows)

    n_total = len(df)
    n_d     = int(post_missing_mask.sum())
    n_ac    = n_total - n_d

    # ------------------------------------------------------------------
    # Note how many of the Tiers A-C are in Phase 1 / Phase 2 samples
    # ------------------------------------------------------------------
    n_known_in_ac = df.loc[df["tier"] == "tiers_ac", "record_id"].isin(known_ids).sum()

    lines += [
        "  RECORD COUNTS",
        "  " + "-" * (W - 2),
        f"  Raw pool (unified_interventions):        {len(df_all):>9,}",
        f"  After bot + pre/intervention filters:    {n_total:>9,}",
        f"  Tier D (no meaningful post-response):    {n_d:>9,}  ({_pct(n_d, n_total)})",
        f"  Tiers A-C (meaningful post-response):    {n_ac:>9,}  ({_pct(n_ac, n_total)})",
        f"    of which in Phase 1 or Phase 2 sample: {n_known_in_ac:>9,}  ({_pct(n_known_in_ac, n_ac)})",
        "",
    ]

    overall_df = pd.DataFrame([
        {"tier": "tier_d",    "n": n_d,    "pct_of_filtered_pool": _pct(n_d, n_total)},
        {"tier": "tiers_ac",  "n": n_ac,   "pct_of_filtered_pool": _pct(n_ac, n_total)},
        {"tier": "total",     "n": n_total, "pct_of_filtered_pool": "100%"},
    ])
    tables["tier_d_overall"] = overall_df

    df_d  = df[df["tier"] == "tier_d"]
    df_ac = df[df["tier"] == "tiers_ac"]

    # ------------------------------------------------------------------
    # Departure rate by platform
    # ------------------------------------------------------------------
    lines += [
        "  DEPARTURE RATE BY PLATFORM",
        "  " + "-" * (W - 2),
        f"  {'Platform':<15s}  {'Total':>8s}  {'Tier D':>8s}  {'Tiers A-C':>10s}  {'Depart%':>9s}",
    ]
    plat_rows = []
    for plat, grp in df.groupby("platform"):
        nd = (grp["tier"] == "tier_d").sum()
        nac = (grp["tier"] == "tiers_ac").sum()
        nt = len(grp)
        lines.append(
            f"  {str(plat):<15s}  {nt:>8,}  {nd:>8,}  {nac:>10,}  {_pct(nd, nt):>9s}"
        )
        plat_rows.append({"platform": plat, "total": nt, "tier_d": nd,
                          "tiers_ac": nac, "departure_rate": _pct(nd, nt)})
    lines.append("")
    tables["tier_d_by_platform"] = pd.DataFrame(plat_rows)

    # ------------------------------------------------------------------
    # Departure rate by intervention type
    # ------------------------------------------------------------------
    lines += [
        "  DEPARTURE RATE BY INTERVENTION TYPE",
        "  " + "-" * (W - 2),
        f"  {'Intervention Type':<30s}  {'Total':>8s}  {'Tier D':>8s}  {'Depart%':>9s}",
    ]
    type_rows = []
    for itype, grp in df.groupby("intervention_type"):
        nd = (grp["tier"] == "tier_d").sum()
        nt = len(grp)
        lines.append(f"  {str(itype):<30s}  {nt:>8,}  {nd:>8,}  {_pct(nd, nt):>9s}")
        type_rows.append({"intervention_type": itype, "total": nt, "tier_d": nd,
                          "tiers_ac": nt - nd, "departure_rate": _pct(nd, nt)})
    lines.append("")
    tables["tier_d_by_intervention_type"] = pd.DataFrame(type_rows)

    # ------------------------------------------------------------------
    # Platform × intervention type departure rate crosstab
    # ------------------------------------------------------------------
    cross_rows = []
    for (plat, itype), grp in df.groupby(["platform", "intervention_type"]):
        nd = (grp["tier"] == "tier_d").sum()
        nt = len(grp)
        cross_rows.append({
            "platform": plat, "intervention_type": itype,
            "total": nt, "tier_d": nd, "tiers_ac": nt - nd,
            "departure_rate_pct": round(100 * nd / nt, 1) if nt > 0 else None,
        })
    cross_df = pd.DataFrame(cross_rows)
    tables["tier_d_platform_x_type"] = cross_df

    lines += [
        "  PLATFORM × INTERVENTION TYPE DEPARTURE RATES",
        "  " + "-" * (W - 2),
        cross_df[["platform", "intervention_type", "total", "tier_d",
                  "departure_rate_pct"]].to_string(index=False),
        "",
    ]

    # ------------------------------------------------------------------
    # Pre-intervention text characteristics: Tier D vs. Tiers A-C
    # ------------------------------------------------------------------
    lines += [
        "  PRE-INTERVENTION TEXT CHARACTERISTICS",
        "  " + "-" * (W - 2),
        "  Compares the pre-intervention text of users who did not respond",
        "  (Tier D) vs. those who did (Tiers A-C).",
        "",
    ]

    pre_rows = []
    for tier_label, tier_df in [("tier_d", df_d), ("tiers_ac", df_ac)]:
        lengths = tier_df["pre_intervention_text"].dropna().str.len()
        words   = tier_df["pre_intervention_text"].dropna().apply(_word_count)
        caps    = tier_df["pre_intervention_text"].dropna().apply(
            lambda t: sum(1 for w in t.split() if len(w) >= 2 and w.isupper()) / max(len(t.split()), 1)
        )
        pre_rows.append({
            "tier": tier_label, "n": len(tier_df),
            "char_length_mean": round(lengths.mean(), 1),
            "char_length_median": round(lengths.median(), 1),
            "word_count_mean": round(words.mean(), 1),
            "word_count_median": round(words.median(), 1),
            "caps_abuse_rate_pct": round(100 * (caps > 0.30).mean(), 1),
        })

    pre_df = pd.DataFrame(pre_rows)
    tables["tier_d_pre_text_stats"] = pre_df

    for _, row in pre_df.iterrows():
        lines.append(
            f"  {row['tier']:<12s}  n={row['n']:>8,}  "
            f"char_len: mean={row['char_length_mean']:.0f} median={row['char_length_median']:.0f}  "
            f"words: mean={row['word_count_mean']:.0f} median={row['word_count_median']:.0f}  "
            f"caps_abuse={row['caps_abuse_rate_pct']:.1f}%"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Response time distribution for Tier D (where available)
    # ------------------------------------------------------------------
    rt_d = df_d["response_time_minutes"].dropna()
    rt_d = rt_d[(rt_d > 0) & (rt_d <= MAX_RESPONSE_MINUTES)]

    rt_rows = []
    lines += [
        "  RESPONSE TIME DISTRIBUTION — TIER D",
        "  " + "-" * (W - 2),
        f"  Tier D records with a response_time_minutes value: {len(rt_d):,}",
        f"  ({_pct(len(rt_d), len(df_d))} of all Tier D records)",
        "  Note: response_time_minutes in this context is the time from",
        "  intervention to the NEXT activity captured — not necessarily a",
        "  direct response.  Many Tier D records will have no value at all.",
        "",
    ]
    if len(rt_d) >= 10:
        for label, grp_rt in [("tier_d_with_rt", rt_d),
                               ("tiers_ac", df_ac["response_time_minutes"].dropna())]:
            grp_clean = grp_rt[(grp_rt > 0) & (grp_rt <= MAX_RESPONSE_MINUTES)]
            rt_rows.append(_percentile_row(grp_clean, label))
            lines.append(
                f"  {label:<20s}  n={len(grp_clean):>7,}  "
                + "  ".join(f"p{p}={_percentile_row(grp_clean, '')[ f'p{p}']:.0f}min"
                             for p in PERCENTILES)
            )
    else:
        lines.append("  Insufficient Tier D records with response_time to report.")
    lines.append("")
    if rt_rows:
        tables["tier_d_response_time"] = pd.DataFrame(rt_rows)

    # ------------------------------------------------------------------
    # Summary interpretation notes
    # ------------------------------------------------------------------
    lines += [
        "=" * W,
        "  INTERPRETATION NOTES",
        "=" * W,
        "",
        "  These findings are descriptive associations only.  Selection bias",
        "  applies in both directions: users who received and responded to",
        "  interventions are a different population from those who did not.",
        "",
        "  Key questions to address in the thesis:",
        "  1. Is the departure rate consistent across platforms, or does it",
        "     vary?  Platform-specific norms may drive different response rates.",
        "  2. Are departure rates higher for punitive intervention types",
        "     (warning, restriction) than prosocial ones?  This is the primary",
        "     analytical claim Tier D supports.",
        "  3. Do Tier D users show different pre-intervention behavior",
        "     (shorter texts, higher caps abuse)?  If so, departure may reflect",
        "     user characteristics, not intervention effectiveness.",
        "",
        "  Framing guidance (from thesis_reframing.md):",
        "  'The absence of post-intervention activity may itself indicate that",
        "  the user left the platform following the intervention.'",
        "  Report as: '…was associated with a departure rate of X%…'",
        "  Do NOT report as: '…caused the user to leave…'",
        "",
        "=" * W,
    ]

    return "\n".join(lines), tables


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Tier D (no post-intervention response) descriptive analysis."
    )
    parser.add_argument(
        "--unified", default=str(PROCESSED / "unified_interventions.parquet"),
        help="Unified interventions parquet (default: output/processed/unified_interventions.parquet)",
    )
    parser.add_argument(
        "--phase1-sample", default=str(PROCESSED / "labeling_sample.parquet"),
        help="Phase 1 labeling sample (for context counts)",
    )
    parser.add_argument(
        "--phase2-sample", default=str(PHASE2 / "phase2_sample.parquet"),
        help="Phase 2 sample (for context counts)",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "output" / "artifacts"),
        help="Output directory (default: output/artifacts/)",
    )
    args = parser.parse_args()

    log.info("=== Tier D Analysis ===")

    unified_path = Path(args.unified)
    if not unified_path.exists():
        log.error(f"Unified interventions not found: {unified_path}")
        sys.exit(1)

    df_all = pd.read_parquet(unified_path)
    log.info(f"Loaded {len(df_all):,} records from unified_interventions.parquet")

    # Collect known Phase 1 + Phase 2 record IDs for context counts
    known_ids: set = set()
    for path_str, label in [(args.phase1_sample, "Phase 1"), (args.phase2_sample, "Phase 2")]:
        p = Path(path_str)
        if p.exists():
            ids = set(pd.read_parquet(p)["record_id"].tolist())
            known_ids |= ids
            log.info(f"  {label}: {len(ids):,} known record IDs loaded")
        else:
            log.warning(f"  {label} sample not found at {path_str} — skipping")

    report_text, tables = analyze(df_all, known_ids)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir   = Path(args.output_dir)
    table_dir = out_dir / "tier_d_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "tier_d_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"\nReport saved → {report_path}")

    for name, df_table in tables.items():
        if df_table is not None and len(df_table) > 0:
            tbl_path = table_dir / f"{name}.csv"
            df_table.to_csv(tbl_path, index=False)
            log.info(f"Table saved  → {tbl_path}")

    print("\n" + report_text)
    log.info("=== Tier D Analysis complete ===")


if __name__ == "__main__":
    main()
