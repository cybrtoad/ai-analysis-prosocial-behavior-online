"""
Step 4: Filter and Stratified Sample
=====================================
Loads the unified intervention parquet, applies quality filters, then draws a
stratified sample of ~3,000 records across platforms and intervention types.

Filters applied:
  - response_time_minutes in (0, 525_600]  (up to 1 year; must be positive)
  - No [deleted] / [removed] placeholder text
  - No known bot usernames (AutoModerator, *Bot, *bot)
  - pre_intervention_text   >= MIN_PRE_LEN   characters
  - intervention_text       >= MIN_INT_LEN   characters
  - post_intervention_text  >= MIN_POST_LEN  characters
  - All three texts         <= MAX_TEXT_LEN  characters

Sampling strategy (stratified by platform × intervention_type):
  reddit        positive_reinforcement  :  500
  reddit        reframe                 :  500
  stackexchange reframe                 : 1000
  wikipedia     warning                 :  250
  wikipedia     restriction             :  250
  wikipedia     positive_reinforcement  :  150
  wikipedia     reframe                 :  350
  ─────────────────────────────────────────────
  TOTAL TARGET                          : 3,000

Within each stratum, records where both the pre-intervention and post-intervention
texts are longer (>= PREFERRED_MIN_PRE / PREFERRED_MIN_POST chars) are sampled
first, to prefer richer context (Tier A/B equivalent). Remaining quota is filled
from the full stratum pool.

If a stratum has fewer records than the target after filtering, all available
records in that stratum are used (with a logged warning).

Input:
  output/processed/unified_interventions.parquet

Output:
  output/processed/labeling_sample.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "output" / "processed"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "sample.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

MIN_PRE_LEN  = 50    # chars
MIN_INT_LEN  = 30    # chars
MIN_POST_LEN = 30    # chars
MAX_TEXT_LEN = 5_000 # chars — keeps prompts manageable for the labeling API call

MAX_RESPONSE_MINUTES = 365 * 24 * 60   # 1 year

# Stratified sampling targets: (platform, intervention_type) -> n
TARGETS: dict[tuple[str, str], int] = {
    ("reddit",        "positive_reinforcement"):  500,
    ("reddit",        "reframe"):                 500,
    ("stackexchange", "reframe"):                1000,
    ("wikipedia",     "warning"):                 250,
    ("wikipedia",     "restriction"):             250,
    ("wikipedia",     "positive_reinforcement"):  150,
    ("wikipedia",     "reframe"):                 350,
}

# Richness preference thresholds (proxy for Tier A/B).
# Records meeting both thresholds are sampled first within each stratum.
PREFERRED_MIN_PRE  = 200   # chars for pre_intervention_text
PREFERRED_MIN_POST = 100   # chars for post_intervention_text

# ---------------------------------------------------------------------------
# Bot / placeholder detection helpers
# ---------------------------------------------------------------------------
_PLACEHOLDER_STRINGS = {"[deleted]", "[removed]", ""}

_BOT_USERNAME_KEYWORDS = ("bot", "automoderator", "spam", "moderationbot")


def _is_placeholder(text: str) -> bool:
    return text.strip().lower() in _PLACEHOLDER_STRINGS


def _is_bot(user_id: str) -> bool:
    uid_lower = str(user_id or "").lower()
    return any(k in uid_lower for k in _BOT_USERNAME_KEYWORDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 4: Filter and Stratified Sample ===")

    in_path = PROCESSED_DIR / "unified_interventions.parquet"
    df = pd.read_parquet(in_path)
    log.info(f"Loaded {len(df):,} records from {in_path.name}")

    # -----------------------------------------------------------------------
    # Quality filters
    # -----------------------------------------------------------------------

    before = len(df)

    # 1. Response time
    df = df[
        df["response_time_minutes"].notna()
        & (df["response_time_minutes"] > 0)
        & (df["response_time_minutes"] <= MAX_RESPONSE_MINUTES)
    ]
    log.info(f"After response-time filter: {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    # 2. No placeholder texts in any of the three fields
    df = df[
        ~df["pre_intervention_text"].apply(_is_placeholder)
        & ~df["intervention_text"].apply(_is_placeholder)
        & ~df["post_intervention_text"].apply(_is_placeholder)
    ]
    log.info(f"After placeholder filter:   {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    # 3. Bot usernames
    df = df[~df["user_id"].apply(_is_bot)]
    log.info(f"After bot-user filter:      {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    # 4. Minimum text lengths
    df = df[
        (df["pre_intervention_text"].str.len()  >= MIN_PRE_LEN)
        & (df["intervention_text"].str.len()    >= MIN_INT_LEN)
        & (df["post_intervention_text"].str.len() >= MIN_POST_LEN)
    ]
    log.info(f"After min-length filter:    {len(df):,} (removed {before - len(df):,})")
    before = len(df)

    # 5. Maximum text lengths (any of the three fields must fit)
    df = df[
        (df["pre_intervention_text"].str.len()   <= MAX_TEXT_LEN)
        & (df["intervention_text"].str.len()     <= MAX_TEXT_LEN)
        & (df["post_intervention_text"].str.len() <= MAX_TEXT_LEN)
    ]
    log.info(f"After max-length filter:    {len(df):,} (removed {before - len(df):,})")

    log.info(f"\nPost-filter distribution:")
    dist = df.groupby(["platform", "intervention_type"]).size().reset_index(name="available")
    log.info("\n" + dist.to_string(index=False))

    # -----------------------------------------------------------------------
    # Stratified sampling
    # -----------------------------------------------------------------------
    log.info("\nSampling...")
    sampled_parts: list[pd.DataFrame] = []

    for (platform, itype), target in TARGETS.items():
        pool = df[(df["platform"] == platform) & (df["intervention_type"] == itype)]
        n_available = len(pool)

        if n_available == 0:
            log.warning(f"  [{platform} / {itype}] NO records available after filtering — skipping")
            continue

        n_sample = min(target, n_available)
        if n_sample < target:
            log.warning(
                f"  [{platform} / {itype}] only {n_available:,} available; "
                f"using all (target was {target})"
            )

        # Prefer records with richer pre/post context (Tier A/B equivalent).
        # Fill from the "preferred" pool first; top up from the remainder.
        rich_mask = (
            (pool["pre_intervention_text"].str.len()  >= PREFERRED_MIN_PRE)
            & (pool["post_intervention_text"].str.len() >= PREFERRED_MIN_POST)
        )
        preferred = pool[rich_mask]
        fallback  = pool[~rich_mask]

        n_from_preferred = min(n_sample, len(preferred))
        n_from_fallback  = n_sample - n_from_preferred

        parts = [preferred.sample(n=n_from_preferred, random_state=RANDOM_SEED)]
        if n_from_fallback > 0:
            parts.append(fallback.sample(n=min(n_from_fallback, len(fallback)), random_state=RANDOM_SEED))

        sampled = pd.concat(parts, ignore_index=True)
        sampled_parts.append(sampled)
        log.info(
            f"  [{platform} / {itype}]  {len(sampled):,} records "
            f"({n_from_preferred:,} preferred / {len(sampled) - n_from_preferred:,} fallback)"
        )

    sample_df = pd.concat(sampled_parts, ignore_index=True)
    # Shuffle so platform order is mixed
    sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = PROCESSED_DIR / "labeling_sample.parquet"
    sample_df.to_parquet(out_path, index=False)

    log.info(f"\nTotal sample: {len(sample_df):,} records")
    log.info(f"Saved -> {out_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("\nFinal distribution:")
    final_dist = (
        sample_df.groupby(["platform", "intervention_type"])
        .size()
        .reset_index(name="count")
    )
    log.info("\n" + final_dist.to_string(index=False))

    log.info("\nResponse time stats (minutes):")
    log.info(sample_df["response_time_minutes"].describe().to_string())

    log.info("\nText length stats (pre_intervention_text):")
    log.info(sample_df["pre_intervention_text"].str.len().describe().to_string())

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
