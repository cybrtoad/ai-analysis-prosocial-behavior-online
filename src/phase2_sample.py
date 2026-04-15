"""
Phase 2 Sample Generation
==========================
Draws ~100,000 records from the filtered intervention pool for Phase 2
Claude scoring. Excludes all records already in the Phase 1 labeling sample
to prevent overlap.

Sampling is proportional to each stratum's (platform × intervention_type)
pool size, so the Phase 2 dataset reflects the natural distribution of the
data. StackExchange dominates the pool (~90%) and will dominate here too —
this is the actual data distribution and is appropriate for Phase 2 analysis.

Input:
  output/processed/unified_interventions.parquet
  output/processed/labeling_sample.parquet  (Phase 1 — excluded from Phase 2)

Output:
  output/phase2/phase2_sample.parquet

Usage:
  python src/phase2_sample.py
  python src/phase2_sample.py --total 100000
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "phase2_sample.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Same quality filters as sample.py
# ---------------------------------------------------------------------------
RANDOM_SEED    = 42
MIN_PRE_LEN    = 50
MIN_INT_LEN    = 30
MIN_POST_LEN   = 30
MAX_TEXT_LEN   = 5_000
MAX_RESPONSE_MINUTES = 365 * 24 * 60

_PLACEHOLDER_STRINGS = {"[deleted]", "[removed]", ""}
_BOT_USERNAME_KEYWORDS = ("bot", "automoderator", "spam", "moderationbot")


def _is_placeholder(text: str) -> bool:
    return text.strip().lower() in _PLACEHOLDER_STRINGS


def _is_bot(user_id: str) -> bool:
    uid_lower = str(user_id or "").lower()
    return any(k in uid_lower for k in _BOT_USERNAME_KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 2 labeling sample")
    parser.add_argument(
        "--total", type=int, default=100_000, metavar="N",
        help="Target total records (default: 100,000)",
    )
    args = parser.parse_args()

    log.info("=== Phase 2 Sample Generation ===")
    log.info(f"Target total: {args.total:,}")

    # ------------------------------------------------------------------
    # Load unified pool
    # ------------------------------------------------------------------
    unified_path = ROOT / "output" / "processed" / "unified_interventions.parquet"
    df = pd.read_parquet(unified_path)
    log.info(f"Loaded {len(df):,} records from unified_interventions.parquet")

    # ------------------------------------------------------------------
    # Apply same quality filters as sample.py
    # ------------------------------------------------------------------
    before = len(df)
    df = df[
        df["response_time_minutes"].notna()
        & (df["response_time_minutes"] > 0)
        & (df["response_time_minutes"] <= MAX_RESPONSE_MINUTES)
    ]
    log.info(f"After response-time filter: {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    df = df[
        ~df["pre_intervention_text"].apply(_is_placeholder)
        & ~df["intervention_text"].apply(_is_placeholder)
        & ~df["post_intervention_text"].apply(_is_placeholder)
    ]
    log.info(f"After placeholder filter:   {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    df = df[~df["user_id"].apply(_is_bot)]
    log.info(f"After bot-user filter:      {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    df = df[
        (df["pre_intervention_text"].str.len()    >= MIN_PRE_LEN)
        & (df["intervention_text"].str.len()      >= MIN_INT_LEN)
        & (df["post_intervention_text"].str.len() >= MIN_POST_LEN)
    ]
    log.info(f"After min-length filter:    {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    df = df[
        (df["pre_intervention_text"].str.len()    <= MAX_TEXT_LEN)
        & (df["intervention_text"].str.len()      <= MAX_TEXT_LEN)
        & (df["post_intervention_text"].str.len() <= MAX_TEXT_LEN)
    ]
    log.info(f"After max-length filter:    {len(df):,} (removed {before - len(df):,})")

    # ------------------------------------------------------------------
    # Exclude Phase 1 records
    # ------------------------------------------------------------------
    phase1_path = ROOT / "output" / "processed" / "labeling_sample.parquet"
    if phase1_path.exists():
        phase1_ids = set(pd.read_parquet(phase1_path)["record_id"].tolist())
        before = len(df)
        df = df[~df["record_id"].isin(phase1_ids)]
        log.info(f"After excluding Phase 1:    {len(df):,} (removed {before - len(df):,})")
    else:
        log.warning("Phase 1 labeling_sample.parquet not found — no exclusion applied")

    log.info(f"\nPool available for Phase 2: {len(df):,}")

    # ------------------------------------------------------------------
    # Proportional stratified sampling
    # ------------------------------------------------------------------
    pool_total = len(df)
    strata = df.groupby(["platform", "intervention_type"]).size().reset_index(name="pool_n")
    strata["target"] = (strata["pool_n"] / pool_total * args.total).round().astype(int)

    # Adjust for rounding so targets sum exactly to args.total
    diff = args.total - strata["target"].sum()
    if diff != 0:
        strata.loc[strata["pool_n"].idxmax(), "target"] += diff

    log.info("\nSampling targets:")
    log.info(strata.to_string(index=False))

    sampled_parts = []
    for _, row in strata.iterrows():
        platform = row["platform"]
        itype    = row["intervention_type"]
        target   = row["target"]
        pool     = df[(df["platform"] == platform) & (df["intervention_type"] == itype)]
        n        = min(target, len(pool))
        if n < target:
            log.warning(f"  [{platform}/{itype}] only {len(pool):,} available; using all")
        sampled_parts.append(pool.sample(n=n, random_state=RANDOM_SEED))
        log.info(f"  [{platform}/{itype}]  {n:,} records")

    sample_df = pd.concat(sampled_parts, ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = ROOT / "output" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_sample.parquet"
    sample_df.to_parquet(out_path, index=False)

    log.info(f"\nTotal Phase 2 sample: {len(sample_df):,} records")
    log.info(f"Saved -> {out_path}")
    log.info("\nNext step:")
    log.info("  python src/label.py \\")
    log.info("      --input output/phase2/phase2_sample.parquet \\")
    log.info("      --output output/phase2/phase2_claude_scores.parquet")


if __name__ == "__main__":
    main()
