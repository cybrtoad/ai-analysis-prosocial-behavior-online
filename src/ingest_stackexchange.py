"""
StackExchange Ask Ubuntu Ingestion Script
==========================================
Parses three XML files from the Ask Ubuntu data dump using iterparse (low
memory, suitable for multi-GB files).

Files parsed:
  Posts.xml        — questions and answers
  Comments.xml     — comments on posts (primary intervention source)
  PostHistory.xml  — edit history, closures, deletions (moderation events)

HTML body text is extracted from the Body field using BeautifulSoup.

Input directory:
  data/stackexchange/askubuntu.com/

Output:
  output/raw/stackexchange_posts.parquet
  output/raw/stackexchange_comments.parquet
  output/raw/stackexchange_history.parquet  (moderation events only)
"""

import logging
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SE_DIR = ROOT / "data" / "stackexchange" / "askubuntu.com"
OUT_DIR = ROOT / "output" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingest_stackexchange.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

CHUNK_SIZE = 50_000

# PostHistoryTypeId values that represent moderation/intervention events
# 10=Closed, 11=Reopened, 12=Deleted, 13=Undeleted, 14=Locked, 15=Unlocked,
# 19=Protected, 20=Unprotected
MODERATION_HISTORY_TYPES = {10, 11, 12, 13, 14, 15, 19, 20}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def html_to_text(html: str | None) -> str:
    """Strip HTML tags and return plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


def parse_tags(tags_str: str | None) -> str:
    """Convert '<python><pandas>' to 'python pandas'."""
    if not tags_str:
        return ""
    return tags_str.replace("><", " ").strip("<>")


def iterparse_rows(xml_path: Path):
    """Yield each <row> element's attributes as a dict."""
    context = ET.iterparse(xml_path, events=("end",))
    for event, elem in context:
        if elem.tag == "row":
            yield dict(elem.attrib)
            elem.clear()


def write_parquet_chunks(rows_iter, out_path: Path, transform_fn=None, desc="rows"):
    """Stream rows from an iterator, transform, and write to parquet in chunks."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    chunk = []
    total = 0

    for raw in tqdm(rows_iter, desc=desc, unit="row"):
        row = transform_fn(raw) if transform_fn else raw
        if row is None:
            continue
        chunk.append(row)
        total += 1

        if len(chunk) >= CHUNK_SIZE:
            df = pd.DataFrame(chunk)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), pa.Schema.from_pandas(df))
            writer.write_table(pa.Table.from_pandas(df))
            chunk.clear()

    if chunk:
        df = pd.DataFrame(chunk)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), pa.Schema.from_pandas(df))
        writer.write_table(pa.Table.from_pandas(df))

    if writer:
        writer.close()

    return total


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------

def transform_post(raw: dict) -> dict | None:
    post_type = int(raw.get("PostTypeId", 0))
    # Keep only questions (1) and answers (2)
    if post_type not in (1, 2):
        return None

    body_html = raw.get("Body", "")
    return {
        "post_id": raw.get("Id"),
        "post_type_id": post_type,
        "parent_id": raw.get("ParentId"),          # answers only
        "accepted_answer_id": raw.get("AcceptedAnswerId"),  # questions only
        "creation_date": raw.get("CreationDate"),
        "score": raw.get("Score"),
        "view_count": raw.get("ViewCount"),
        "body_html": body_html,
        "body_text": html_to_text(body_html),
        "owner_user_id": raw.get("OwnerUserId"),
        "last_editor_user_id": raw.get("LastEditorUserId"),
        "last_edit_date": raw.get("LastEditDate"),
        "last_activity_date": raw.get("LastActivityDate"),
        "title": raw.get("Title", ""),             # questions only
        "tags": parse_tags(raw.get("Tags")),       # questions only
        "answer_count": raw.get("AnswerCount"),
        "comment_count": raw.get("CommentCount"),
        "closed_date": raw.get("ClosedDate"),
        "content_license": raw.get("ContentLicense"),
    }


def ingest_posts():
    path = SE_DIR / "Posts.xml"
    out_path = OUT_DIR / "stackexchange_posts.parquet"
    log.info(f"Parsing Posts.xml ({path.stat().st_size / 1e6:.0f} MB)…")
    n = write_parquet_chunks(
        iterparse_rows(path),
        out_path,
        transform_fn=transform_post,
        desc="posts",
    )
    log.info(f"  Saved {n:,} posts -> {out_path}")


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

def transform_comment(raw: dict) -> dict:
    return {
        "comment_id": raw.get("Id"),
        "post_id": raw.get("PostId"),
        "score": raw.get("Score"),
        "text": raw.get("Text", ""),
        "creation_date": raw.get("CreationDate"),
        "user_id": raw.get("UserId"),
        "user_display_name": raw.get("UserDisplayName", ""),
        "content_license": raw.get("ContentLicense"),
    }


def ingest_comments():
    path = SE_DIR / "Comments.xml"
    out_path = OUT_DIR / "stackexchange_comments.parquet"
    log.info(f"Parsing Comments.xml ({path.stat().st_size / 1e6:.0f} MB)…")
    n = write_parquet_chunks(
        iterparse_rows(path),
        out_path,
        transform_fn=transform_comment,
        desc="comments",
    )
    log.info(f"  Saved {n:,} comments -> {out_path}")


# ---------------------------------------------------------------------------
# PostHistory (moderation events only)
# ---------------------------------------------------------------------------

def transform_history(raw: dict) -> dict | None:
    try:
        type_id = int(raw.get("PostHistoryTypeId", 0))
    except ValueError:
        return None
    # Keep only moderation events
    if type_id not in MODERATION_HISTORY_TYPES:
        return None
    return {
        "history_id": raw.get("Id"),
        "post_id": raw.get("PostId"),
        "history_type_id": type_id,
        "creation_date": raw.get("CreationDate"),
        "user_id": raw.get("UserId"),
        "user_display_name": raw.get("UserDisplayName", ""),
        "comment": raw.get("Comment", ""),   # Edit summary / close reason
        "text": raw.get("Text", ""),
        "revision_guid": raw.get("RevisionGUID"),
        "content_license": raw.get("ContentLicense"),
    }


def ingest_post_history():
    path = SE_DIR / "PostHistory.xml"
    out_path = OUT_DIR / "stackexchange_history.parquet"
    log.info(f"Parsing PostHistory.xml ({path.stat().st_size / 1e6:.0f} MB)…")
    n = write_parquet_chunks(
        iterparse_rows(path),
        out_path,
        transform_fn=transform_history,
        desc="history",
    )
    log.info(f"  Saved {n:,} moderation events -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== StackExchange Ask Ubuntu Ingestion ===")
    ingest_posts()
    ingest_comments()
    ingest_post_history()
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
