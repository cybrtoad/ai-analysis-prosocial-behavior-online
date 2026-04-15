"""
Wikipedia NS0 Article Edits Ingestion Script
=============================================
Targeted second streaming pass of the 43 GB Wikipedia XML dump.
Extracts only NS 0 (main article) revisions authored by users who appear
as targets in intervention events on NS 3 (User talk) pages.

This provides the pre- and post-intervention article edit texts needed
for cross-namespace triple construction (see standardize.py).

Phase A:
  Load wikipedia_raw.parquet (NS 3 User talk pages already extracted).
  Identify intervention events: revisions where a non-owner edits a user's
  talk page with intervention keywords/templates.
  Collect the set of target usernames (the users being intervened upon).

Phase B:
  Stream the full XML again with iterparse.
  Yield only NS 0 revisions whose contributor is in the target set.
  Cap text_content at TEXT_CAP characters (article texts can be multi-MB).

Input:
  output/raw/wikipedia_raw.parquet          (NS 3 talk pages — already extracted)
  data/simplewiki/simplewiki-latest-pages-meta-history.xml  (43 GB)

Output:
  output/raw/wikipedia_user_article_edits.parquet
"""

import logging
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "simplewiki" / "simplewiki-latest-pages-meta-history.xml"
RAW_DIR  = ROOT / "output" / "raw"
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingest_wikipedia_ns0.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MW_NS      = "{http://www.mediawiki.org/xml/export-0.11/}"
CHUNK_SIZE = 50_000
TEXT_CAP   = 5_000   # chars — article texts can be very large

# Keywords in edit_comment that signal an intervention
COMMENT_KEYWORDS = (
    "warn", "vandal", "block", "spam", "uw-",
    "welcome", "thanks", "revert", " rv ", "undid", "undo", "civil",
)

# Templates in text_content that signal an intervention
TEXT_TEMPLATES = (
    "{{uw-", "{{welcome", "{{warning", "{{bansummary",
    "{{indef-block", "{{blocked", "wp:civil",
)


# ---------------------------------------------------------------------------
# Phase A — identify target usernames from NS 3
# ---------------------------------------------------------------------------

def _extract_target_username(page_title: str) -> str | None:
    """
    Extract the target username from a User talk page title.
    'User talk:Angela'    -> 'Angela'
    'User talk:127.0.0.1' -> '127.0.0.1'
    Returns None if the title does not match the expected prefix.
    """
    prefix = "User talk:"
    if page_title and page_title.startswith(prefix):
        return page_title[len(prefix):]
    return None


def _is_intervention(edit_comment: str | None, text_content: str | None) -> bool:
    """Return True if this NS 3 revision looks like a moderation intervention."""
    comment = (edit_comment or "").lower()
    text    = (text_content or "").lower()
    return (
        any(k in comment for k in COMMENT_KEYWORDS)
        or any(t in text for t in TEXT_TEMPLATES)
    )


def build_target_set(talk_parquet: Path) -> set[str]:
    """
    Load the existing NS 3 parquet and return the set of target usernames
    who appear as the subject of at least one intervention event.
    Uses fully vectorized pandas operations — no iterrows().
    """
    log.info("=== Phase A: identifying target users from NS 3 talk pages ===")
    df = pd.read_parquet(talk_parquet)
    log.info(f"  Loaded {len(df):,} revisions")

    # Keep only User talk pages (NS 3)
    df = df[df["namespace"] == 3].copy()
    log.info(f"  {len(df):,} revisions in User talk (NS 3)")

    # Extract target username from page_title vectorized:
    # "User talk:Angela" -> "Angela", others -> NaN
    PREFIX = "User talk:"
    mask_prefix = df["page_title"].str.startswith(PREFIX, na=False)
    df = df[mask_prefix].copy()
    df["target_user"] = df["page_title"].str[len(PREFIX):]
    log.info(f"  {len(df):,} revisions on User talk pages")

    # Best contributor identifier per row (vectorized)
    df["contributor"] = df["contributor_username"].where(
        df["contributor_username"].notna(),
        df["contributor_ip"]
    )
    df = df[df["contributor"].notna()]

    # Drop self-edits (user editing their own talk page)
    df = df[df["contributor"] != df["target_user"]]
    log.info(f"  {len(df):,} revisions by non-owner contributors")

    # Vectorized intervention keyword matching
    comment_lower = df["edit_comment"].fillna("").str.lower()
    text_lower    = df["text_content"].fillna("").str.lower()

    kw_mask = comment_lower.str.contains(
        "|".join(COMMENT_KEYWORDS), regex=True, na=False
    )
    tmpl_mask = text_lower.str.contains(
        "|".join(re.escape(t) for t in TEXT_TEMPLATES), regex=True, na=False
    )
    df = df[kw_mask | tmpl_mask]
    log.info(f"  {len(df):,} intervention events (keyword/template match)")

    target_set = set(df["target_user"].dropna().unique())
    log.info(f"  Unique target users: {len(target_set):,}")
    return target_set


# ---------------------------------------------------------------------------
# Phase B — stream XML for NS 0 revisions by target users
# ---------------------------------------------------------------------------

def strip_ns(tag: str) -> str:
    return tag.replace(MW_NS, "")


def stream_ns0_revisions(xml_path: Path, target_set: set[str]):
    """
    Generator: yield one dict per NS 0 revision whose contributor is in target_set.
    Uses iterparse — the 43 GB file is never fully loaded into memory.
    """
    context = ET.iterparse(xml_path, events=("start", "end"))

    current_page_id    = None
    current_page_title = None
    current_ns         = None
    in_target_ns       = False

    rev          = {}
    in_revision  = False
    in_contributor = False

    for event, elem in context:
        tag = strip_ns(elem.tag)

        # ---- PAGE-LEVEL ----
        if event == "start" and tag == "page":
            current_page_id    = None
            current_page_title = None
            current_ns         = None
            in_target_ns       = False

        elif event == "end" and tag == "ns" and not in_revision:
            try:
                current_ns = int(elem.text) if elem.text else -1
            except ValueError:
                current_ns = -1
            in_target_ns = (current_ns == 0)   # NS 0 = Main articles

        elif event == "end" and tag == "title" and not in_revision:
            current_page_title = elem.text

        elif event == "end" and tag == "id" and not in_revision and not in_contributor:
            current_page_id = elem.text

        # ---- REVISION-LEVEL ----
        elif event == "start" and tag == "revision":
            if in_target_ns:
                in_revision = True
                rev = {
                    "page_id":              current_page_id,
                    "page_title":           current_page_title,
                    "revision_id":          None,
                    "parent_revision_id":   None,
                    "timestamp":            None,
                    "contributor_username": None,
                    "contributor_id":       None,
                    "contributor_ip":       None,
                    "edit_comment":         None,
                    "text_content":         None,
                    "text_bytes":           None,
                    "sha1":                 None,
                }

        elif event == "end" and tag == "revision":
            if in_target_ns and in_revision:
                # Only yield if the contributor is one of our targets
                contributor = rev.get("contributor_username") or rev.get("contributor_ip")
                if contributor and contributor in target_set:
                    yield rev
            in_revision = False
            rev = {}
            elem.clear()

        elif in_revision:
            if event == "start" and tag == "contributor":
                in_contributor = True
            elif event == "end" and tag == "contributor":
                in_contributor = False
            elif event == "end":
                if tag == "id":
                    if in_contributor:
                        rev["contributor_id"] = elem.text
                    else:
                        rev["revision_id"] = elem.text
                elif tag == "parentid":
                    rev["parent_revision_id"] = elem.text
                elif tag == "timestamp":
                    rev["timestamp"] = elem.text
                elif tag == "username":
                    rev["contributor_username"] = elem.text
                elif tag == "ip":
                    rev["contributor_ip"] = elem.text
                elif tag == "comment":
                    rev["edit_comment"] = elem.text
                elif tag == "text":
                    raw = elem.text or ""
                    rev["text_content"] = raw[:TEXT_CAP]
                    rev["text_bytes"]   = elem.get("bytes")
                elif tag == "sha1":
                    rev["sha1"] = elem.text

        elif event == "end" and tag == "page":
            elem.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Wikipedia NS0 Article Edits Ingestion ===")
    log.info(f"Source: {XML_PATH}")

    talk_parquet = RAW_DIR / "wikipedia_raw.parquet"
    if not talk_parquet.exists():
        log.error(f"wikipedia_raw.parquet not found at {talk_parquet}")
        log.error("Run src/ingest_wikipedia.py first.")
        sys.exit(1)

    # Phase A
    target_set = build_target_set(talk_parquet)
    if not target_set:
        log.error("No target users found — check intervention keyword/template logic.")
        sys.exit(1)

    # Phase B
    log.info(f"\n=== Phase B: streaming NS 0 revisions for {len(target_set):,} target users ===")
    out_path = RAW_DIR / "wikipedia_user_article_edits.parquet"

    chunk: list[dict] = []
    total = 0
    chunk_num = 0
    writer = None

    for rev in tqdm(
        stream_ns0_revisions(XML_PATH, target_set),
        desc="NS0 revisions matched",
        unit="rev",
    ):
        chunk.append(rev)
        total += 1

        if len(chunk) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(chunk)
            if writer is None:
                import pyarrow as pa
                import pyarrow.parquet as pq
                schema = pa.Schema.from_pandas(df_chunk)
                writer = pq.ParquetWriter(str(out_path), schema)
            import pyarrow as pa
            writer.write_table(pa.Table.from_pandas(df_chunk))
            chunk_num += 1
            log.info(f"  Wrote chunk {chunk_num} ({total:,} revisions so far)")
            chunk.clear()

    # Final partial chunk
    if chunk:
        df_chunk = pd.DataFrame(chunk)
        if writer is None:
            import pyarrow as pa
            import pyarrow.parquet as pq
            schema = pa.Schema.from_pandas(df_chunk)
            writer = pq.ParquetWriter(str(out_path), schema)
        import pyarrow as pa
        writer.write_table(pa.Table.from_pandas(df_chunk))

    if writer:
        writer.close()

    log.info(f"\nTotal NS0 revisions written: {total:,}")
    log.info(f"Saved -> {out_path}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
