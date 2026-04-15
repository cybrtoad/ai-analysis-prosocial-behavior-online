"""
src/build_phase2_features.py
=============================
Merge Phase 2 Claude scores with the Phase 2 sample, compute all features
that are derivable from the current data, and save a single flat analysis
dataset ready for analyze_phase2.py.

Input files
-----------
  output/phase2/phase2_claude_scores.parquet   Claude Tier 1 scores for ~100K records
  output/phase2/phase2_sample.parquet          Unified-schema records (text + metadata)

Output
------
  output/phase2/phase2_analysis_dataset.parquet          Reddit + StackExchange only (main results)
  output/phase2/phase2_analysis_dataset_wikipedia.parquet  Wikipedia only (boundary condition)

Features computed here
----------------------
  Prosociality
    prosociality_composite      mean of four Tier 1 scores (1.0-5.0)

  Text features (all from raw text fields)
    pre_text_length             pre_intervention_text character count
    pre_word_count              word count
    pre_has_caps_abuse          >30% uppercase words (crude aggression signal)
    intervention_length         intervention_text word count
    intervention_has_question   "?" present in intervention text
    post_text_length            post_intervention_text character count (nullable)
    post_word_count             word count (nullable)
    scoring_mode                text_scored / outcome_only

  Intervention features (derived from type or text)
    intervener_role             peer / domain_expert / formal_authority / peer_editor
                                  (derived from platform + intervention_type; passed through
                                  from phase2_sample if already present, otherwise derived here)
    intervention_severity       1=low / 2=medium / 3=high (derived from intervention_type)
    intervention_visibility     public / semi_public (derived from platform)
    intervention_timing         immediate / delayed (response_time_minutes <= 60 threshold)
    response_time_category      immediate (<=60min) / same_day (<=1440) / delayed (>1440)

  User context features (approximated from in-sample activity)
    user_n_interventions_in_sample   total times this user appears in the 100K sample
    user_prior_interventions_in_sample  count of records for same user_id with earlier timestamp
    user_tenure_estimate_category   newcomer / regular / veteran based on first-last appearance
                                    in sample (proxy for account age — not true account age)

  Platform metadata (extracted from platform_metadata JSON blob)
    reddit_delta_awarded        binary (Reddit only)
    se_post_score               int (StackExchange only)
    se_accepted_answer          binary (StackExchange only)

Features NOT yet computable (require additional data extraction)
---------------------------------------------------------------
  user_account_age_days         needs raw platform account creation date lookup
  user_previous_contributions   needs full platform activity timeline (not just sample)
  user_retention_7day           needs full activity timeline beyond intervention date
  user_retention_30day          same
  user_outcome_type             departure/ban detection needs full activity scan
  days_until_next_activity      same
  immediate_behavioral_change   needs multiple post-intervention texts per user
  sustained_behavior_score      same
  pre_toxicity_score            needs detoxify / Perspective API
  post_toxicity_score           same
  intervention_style            requires human annotation (have only for 340 records)
  conversation_depth            needs full conversation thread reconstruction

Usage
-----
  python src/build_phase2_features.py
  python src/build_phase2_features.py \\
      --scores output/phase2/phase2_claude_scores.parquet \\
      --sample output/phase2/phase2_sample.parquet \\
      --output output/phase2/phase2_analysis_dataset.parquet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
PHASE2    = ROOT / "output" / "phase2"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "build_phase2_features.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]

# Intervener role derivation (mirrors src/standardize.py)
INTERVENER_ROLE_MAP = {
    ("reddit",        "reframe"):                "peer",
    ("reddit",        "positive_reinforcement"):  "peer",
    ("reddit",        "warning"):                 "peer",
    ("reddit",        "restriction"):             "peer",
    ("stackexchange", "reframe"):                "domain_expert",
    ("stackexchange", "positive_reinforcement"):  "domain_expert",
    ("stackexchange", "warning"):                "domain_expert",
    ("stackexchange", "restriction"):            "domain_expert",
    ("wikipedia",     "warning"):                "formal_authority",
    ("wikipedia",     "restriction"):            "formal_authority",
    ("wikipedia",     "reframe"):                "peer_editor",
    ("wikipedia",     "positive_reinforcement"):  "peer_editor",
}


def _derive_intervener_role(platform: str, intervention_type: str) -> str:
    return INTERVENER_ROLE_MAP.get((platform, intervention_type), "unknown")


# Derived severity from intervention_type
SEVERITY_MAP = {
    "warning":               3,
    "restriction":           3,
    "reframe":               2,
    "positive_reinforcement": 1,
}

# Response time thresholds in minutes
IMMEDIATE_MINUTES = 60       # <= 60 min = immediate
SAME_DAY_MINUTES  = 1440     # <= 24 h = same_day; > 24 h = delayed


# ---------------------------------------------------------------------------
# Text feature helpers
# ---------------------------------------------------------------------------

def _word_count(text: str | None) -> int | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return len(text.split())


def _char_count(text: str | None) -> int | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return len(text)


def _has_caps_abuse(text: str | None, threshold: float = 0.30) -> bool:
    """True if more than threshold fraction of words are fully uppercase (len >= 2)."""
    if not isinstance(text, str) or not text.strip():
        return False
    words = [w for w in text.split() if len(w) >= 2]
    if not words:
        return False
    return sum(1 for w in words if w.isupper()) / len(words) > threshold


def _has_question(text: str | None) -> bool:
    return isinstance(text, str) and "?" in text


# ---------------------------------------------------------------------------
# Platform metadata extraction
# ---------------------------------------------------------------------------

def _parse_metadata(meta_str: str | None) -> dict:
    if not isinstance(meta_str, str) or not meta_str.strip():
        return {}
    try:
        return json.loads(meta_str)
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the platform_metadata JSON column and extract platform-specific fields
    into flat columns.  Unknown fields are silently ignored.
    """
    metas = df["platform_metadata"].apply(_parse_metadata)

    # Reddit
    df["reddit_delta_awarded"] = metas.apply(
        lambda m: bool(m.get("delta_awarded", False))
        if isinstance(m, dict) else False
    )

    # StackExchange
    df["se_post_score"] = metas.apply(
        lambda m: int(m["Score"]) if isinstance(m, dict) and "Score" in m else np.nan
    )
    df["se_accepted_answer"] = metas.apply(
        lambda m: bool(m.get("AcceptedAnswerId")) if isinstance(m, dict) else False
    )

    return df


# ---------------------------------------------------------------------------
# User in-sample activity features
# ---------------------------------------------------------------------------

def compute_user_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user in-sample intervention counts and a proxy tenure category.

    user_n_interventions_in_sample:
        Total records for this user_id in the 100K sample.

    user_prior_interventions_in_sample:
        Records for this user_id with timestamp strictly before this record's
        timestamp.  Approximates prior moderation experience.

    user_tenure_estimate_category:
        Derived from the range of timestamps for this user_id in the sample.
        newcomer:  span < 30 days (or only one appearance)
        regular:   30-365 days
        veteran:   > 365 days
        NOTE: This is a very rough proxy.  True tenure requires account
        creation dates from raw platform data.
    """
    log.info("Computing user in-sample activity features...")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Total appearances per user
    counts = df.groupby("user_id").size().rename("user_n_interventions_in_sample")
    df = df.merge(counts, on="user_id", how="left")

    # Prior interventions per user (records with earlier timestamp)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["user_prior_interventions_in_sample"] = (
        df.groupby("user_id").cumcount()   # 0-indexed rank within user, sorted by time
    )

    # Tenure estimate from timestamp span within sample
    user_ts = df.groupby("user_id")["timestamp"].agg(["min", "max"])
    user_ts["span_days"] = (user_ts["max"] - user_ts["min"]).dt.total_seconds() / 86400
    user_ts["user_tenure_estimate_category"] = pd.cut(
        user_ts["span_days"],
        bins=[-1, 30, 365, float("inf")],
        labels=["newcomer", "regular", "veteran"],
    ).astype(str)
    df = df.merge(
        user_ts[["user_tenure_estimate_category"]],
        on="user_id", how="left",
    )

    return df


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def build_features(df_scores: pd.DataFrame, df_sample: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Scores: {len(df_scores):,}   Sample: {len(df_sample):,}")

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    df = df_sample.merge(df_scores[["record_id"] + DIMENSIONS + ["confidence", "invalid_triple"]],
                         on="record_id", how="inner")
    log.info(f"After merge: {len(df):,} records")

    # Drop invalid triples
    if "invalid_triple" in df.columns:
        n_before = len(df)
        df = df[~df["invalid_triple"].fillna(False).astype(bool)].copy()
        log.info(f"Dropped {n_before - len(df):,} invalid triples → {len(df):,} records")

    # ------------------------------------------------------------------
    # Prosociality composite
    # ------------------------------------------------------------------
    df["prosociality_composite"] = df[DIMENSIONS].mean(axis=1).round(4)

    # ------------------------------------------------------------------
    # Text features
    # ------------------------------------------------------------------
    log.info("Computing text features...")
    df["pre_text_length"]  = df["pre_intervention_text"].apply(_char_count)
    df["pre_word_count"]   = df["pre_intervention_text"].apply(_word_count)
    df["pre_has_caps_abuse"] = df["pre_intervention_text"].apply(_has_caps_abuse)

    df["intervention_length"]       = df["intervention_text"].apply(_word_count)
    df["intervention_has_question"] = df["intervention_text"].apply(_has_question)

    df["post_text_length"] = df["post_intervention_text"].apply(_char_count)
    df["post_word_count"]  = df["post_intervention_text"].apply(_word_count)

    df["scoring_mode"] = df["post_intervention_text"].apply(
        lambda t: "text_scored" if isinstance(t, str) and t.strip() else "outcome_only"
    )

    # ------------------------------------------------------------------
    # Intervention features
    # ------------------------------------------------------------------
    log.info("Computing intervention features...")

    # Intervener role: pass through from sample if already present, otherwise derive
    if "intervener_role" not in df.columns:
        df["intervener_role"] = df.apply(
            lambda r: _derive_intervener_role(r["platform"], r["intervention_type"]), axis=1
        )
        log.info("  intervener_role: derived from platform + intervention_type")
    else:
        log.info("  intervener_role: passed through from sample")

    df["intervention_severity"] = (
        df["intervention_type"].map(SEVERITY_MAP).fillna(2).astype(int)
    )

    # Visibility: Wikipedia user talk page messages are addressed to a specific
    # user but technically publicly readable — coded semi_public.  Reddit and SE
    # comments are fully public.
    df["intervention_visibility"] = df["platform"].map({
        "reddit":         "public",
        "stackexchange":  "public",
        "wikipedia":      "semi_public",
    }).fillna("public")

    # Timing: derived from response_time_minutes
    def _timing_category(rt: float | None) -> str:
        if pd.isna(rt):
            return "unknown"
        if rt <= IMMEDIATE_MINUTES:
            return "immediate"
        if rt <= SAME_DAY_MINUTES:
            return "same_day"
        return "delayed"

    df["intervention_timing"]      = df["response_time_minutes"].apply(_timing_category)
    df["response_time_category"]   = df["intervention_timing"]   # alias for clarity

    # ------------------------------------------------------------------
    # Platform metadata
    # ------------------------------------------------------------------
    log.info("Extracting platform metadata features...")
    df = extract_platform_features(df)

    # ------------------------------------------------------------------
    # User in-sample activity
    # ------------------------------------------------------------------
    df = compute_user_activity(df)

    # ------------------------------------------------------------------
    # Sort and clean up
    # ------------------------------------------------------------------
    df = df.sort_values("record_id").reset_index(drop=True)

    # Drop raw text columns to keep the analysis dataset compact.
    # The texts are still in phase2_sample.parquet if needed.
    df = df.drop(columns=[
        "pre_intervention_text", "intervention_text", "post_intervention_text",
        "platform_metadata",
    ], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 analysis dataset by merging Claude scores with sample features."
    )
    parser.add_argument("--scores", default=str(PHASE2 / "phase2_claude_scores.parquet"),
                        help="Claude scores parquet (default: output/phase2/phase2_claude_scores.parquet)")
    parser.add_argument("--sample", default=str(PHASE2 / "phase2_sample.parquet"),
                        help="Phase 2 sample parquet (default: output/phase2/phase2_sample.parquet)")
    parser.add_argument("--output", default=str(PHASE2 / "phase2_analysis_dataset.parquet"),
                        help="Output path for Reddit+SE dataset (default: output/phase2/phase2_analysis_dataset.parquet)")
    parser.add_argument("--output-wikipedia", default=str(PHASE2 / "phase2_analysis_dataset_wikipedia.parquet"),
                        help="Output path for Wikipedia-only dataset (default: output/phase2/phase2_analysis_dataset_wikipedia.parquet)")
    args = parser.parse_args()

    log.info("=== Build Phase 2 Analysis Dataset ===")

    for path_str, label in [(args.scores, "scores"), (args.sample, "sample")]:
        if not Path(path_str).exists():
            log.error(f"{label} file not found: {path_str}")
            sys.exit(1)

    df_scores = pd.read_parquet(args.scores)
    df_sample = pd.read_parquet(args.sample)
    log.info(f"Loaded scores: {len(df_scores):,}  sample: {len(df_sample):,}")

    df = build_features(df_scores, df_sample)

    # ------------------------------------------------------------------
    # Split into main (Reddit + SE) and Wikipedia-only datasets
    # ------------------------------------------------------------------
    df_main = df[df["platform"] != "wikipedia"].copy().reset_index(drop=True)
    df_wiki = df[df["platform"] == "wikipedia"].copy().reset_index(drop=True)
    log.info(f"\nPlatform split:")
    log.info(f"  Main (Reddit + SE): {len(df_main):,} records")
    log.info(f"  Wikipedia only:     {len(df_wiki):,} records")

    # Save main dataset
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_main.to_parquet(out_path, index=False)
    log.info(f"\nSaved main dataset (Reddit + SE) → {out_path}")

    # Save Wikipedia dataset
    wiki_path = Path(args.output_wikipedia)
    wiki_path.parent.mkdir(parents=True, exist_ok=True)
    df_wiki.to_parquet(wiki_path, index=False)
    log.info(f"Saved Wikipedia dataset          → {wiki_path}")

    # ------------------------------------------------------------------
    # Quick summary for main dataset
    # ------------------------------------------------------------------
    log.info("\n--- Main dataset summary (Reddit + SE) ---")
    log.info(f"  Rows: {len(df_main):,}   Columns: {len(df_main.columns)}")
    log.info("\nPlatform × intervention_type × intervener_role counts:")
    log.info(
        df_main.groupby(["platform", "intervention_type", "intervener_role"]).size()
        .rename("n").reset_index().to_string(index=False)
    )
    log.info("\nProsociality composite stats (main):")
    log.info(df_main["prosociality_composite"].describe().round(3).to_string())

    if len(df_wiki) > 0:
        log.info("\n--- Wikipedia dataset summary ---")
        log.info(f"  Rows: {len(df_wiki):,}")
        log.info("\nProsociality composite stats (Wikipedia):")
        log.info(df_wiki["prosociality_composite"].describe().round(3).to_string())

    log.info("\nFeatures NOT YET computed (require additional data):")
    for feat in [
        "user_account_age_days", "user_previous_contributions",
        "user_retention_7day", "user_retention_30day", "user_outcome_type",
        "days_until_next_activity", "immediate_behavioral_change",
        "sustained_behavior_score", "pre_toxicity_score", "post_toxicity_score",
        "intervention_style (human annotation — 340 records only)",
        "conversation_depth",
    ]:
        log.info(f"  - {feat}")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
