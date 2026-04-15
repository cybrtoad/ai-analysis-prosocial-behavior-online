"""
Reddit CMV Ingestion Script
===========================
Reads the Conversations Gone Awry CMV Corpus (ConvoKit format) directly from
the raw JSONL/JSON files — no ConvoKit library required.

Extracts all utterances and joins conversation + speaker metadata.

Input:
  data/reddit/conversations-gone-awry-cmv-corpus/utterances.jsonl
  data/reddit/conversations-gone-awry-cmv-corpus/conversations.json
  data/reddit/conversations-gone-awry-cmv-corpus/speakers.json

Output:
  output/raw/reddit_raw.parquet
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "reddit" / "conversations-gone-awry-cmv-corpus"
OUT_DIR = ROOT / "output" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingest_reddit.log", encoding="utf-8"),
    ],
)
# Force stdout to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> dict:
    log.info(f"Loading conversations from {path}")
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    log.info(f"  {len(convs):,} conversations loaded")
    return convs


def load_speakers(path: Path) -> dict:
    log.info(f"Loading speakers from {path}")
    with open(path, encoding="utf-8") as f:
        speakers = json.load(f)
    log.info(f"  {len(speakers):,} speakers loaded")
    return speakers


def count_lines(path: Path) -> int:
    """Fast line count for progress bar."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def load_utterances(path: Path, conversations: dict, speakers: dict) -> pd.DataFrame:
    log.info(f"Loading utterances from {path}")
    n_lines = count_lines(path)
    log.info(f"  Counting lines: {n_lines:,}")

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=n_lines, desc="utterances", unit="utt"):
            u = json.loads(line)

            utt_id = u.get("id")
            conv_id = u.get("conversation_id")
            speaker = u.get("speaker")
            text = u.get("text", "")
            reply_to = u.get("reply-to")
            timestamp = u.get("timestamp")

            meta = u.get("meta", {})
            score = meta.get("score")
            top_level_comment = meta.get("top_level_comment")
            retrieved_on = meta.get("retrieved_on")
            gilded = meta.get("gilded", 0)
            subreddit = meta.get("subreddit", "changemyview")
            stickied = meta.get("stickied", False)
            permalink = meta.get("permalink", "")
            author_flair = meta.get("author_flair_text", "")

            # Join conversation metadata
            conv = conversations.get(conv_id, {})
            conv_meta = conv.get("meta", {})
            pair_id = conv_meta.get("pair_id")
            has_removed_comment = conv_meta.get("has_removed_comment")
            split = conv_meta.get("split")

            # Join speaker metadata
            spk = speakers.get(speaker, {})
            spk_meta = spk.get("meta", {})
            speaker_num_posts = spk_meta.get("num_posts")
            speaker_num_comments = spk_meta.get("num_comments")

            rows.append({
                "utterance_id": utt_id,
                "conversation_id": conv_id,
                "speaker": speaker,
                "text": text,
                "reply_to": reply_to,
                "timestamp": timestamp,
                "score": score,
                "top_level_comment": top_level_comment,
                "retrieved_on": retrieved_on,
                "gilded": gilded,
                "subreddit": subreddit,
                "stickied": stickied,
                "permalink": permalink,
                "author_flair": author_flair,
                # Conversation-level metadata
                "conv_pair_id": pair_id,
                "conv_has_removed_comment": has_removed_comment,
                "conv_split": split,
                # Speaker-level metadata
                "speaker_num_posts": speaker_num_posts,
                "speaker_num_comments": speaker_num_comments,
            })

    df = pd.DataFrame(rows)
    log.info(f"  {len(df):,} utterances loaded")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Reddit CMV Ingestion ===")

    conversations = load_conversations(DATA_DIR / "conversations.json")
    speakers = load_speakers(DATA_DIR / "speakers.json")
    df = load_utterances(DATA_DIR / "utterances.jsonl", conversations, speakers)

    # Basic stats
    log.info(f"Conversations: {df['conversation_id'].nunique():,}")
    log.info(f"Speakers: {df['speaker'].nunique():,}")
    log.info(f"Utterances with text: {df['text'].notna().sum():,}")
    log.info(f"Conv splits: {df['conv_split'].value_counts().to_dict()}")

    out_path = OUT_DIR / "reddit_raw.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df):,} rows -> {out_path}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
