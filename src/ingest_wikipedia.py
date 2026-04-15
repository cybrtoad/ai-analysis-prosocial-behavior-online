"""
Simple English Wikipedia Ingestion Script
==========================================
Streams the 43 GB pages-meta-history XML dump using iterparse (low memory).
Extracts revisions from talk namespaces only:
  NS 1  = Article Talk
  NS 3  = User Talk   (primary intervention space)
  NS 5  = Wikipedia Talk

For each revision, records page info, contributor, edit summary, and raw
wikitext content.  Intervention detection (template parsing) happens in
Step 3 (standardization).

Input:
  data/simplewiki/simplewiki-latest-pages-meta-history.xml  (43 GB)

Output:
  output/raw/wikipedia_raw.parquet
"""

import logging
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "simplewiki" / "simplewiki-latest-pages-meta-history.xml"
OUT_DIR = ROOT / "output" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingest_wikipedia.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

# Talk namespaces we care about
TARGET_NAMESPACES = {1, 3, 5}
NS_NAMES = {1: "Talk", 3: "User_talk", 5: "Wikipedia_talk"}

# MediaWiki XML namespace prefix
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# Write to parquet in chunks to keep memory manageable (43 GB file)
CHUNK_SIZE = 50_000


def strip_ns(tag: str) -> str:
    """Remove MediaWiki XML namespace prefix from a tag."""
    return tag.replace(MW_NS, "")


def stream_talk_revisions(xml_path: Path):
    """
    Generator: yields one dict per revision from talk-namespace pages.
    Uses iterparse so the full 43 GB file is never loaded into memory.
    """
    context = ET.iterparse(xml_path, events=("start", "end"))

    # State for the current page
    current_page_id = None
    current_page_title = None
    current_ns = None
    in_target_ns = False

    # State for the current revision
    rev = {}
    in_revision = False
    in_contributor = False

    for event, elem in context:
        tag = strip_ns(elem.tag)

        # ---- PAGE-LEVEL ELEMENTS ----
        if event == "start" and tag == "page":
            current_page_id = None
            current_page_title = None
            current_ns = None
            in_target_ns = False

        elif event == "end" and tag == "ns" and not in_revision:
            try:
                current_ns = int(elem.text) if elem.text else 0
            except ValueError:
                current_ns = 0
            in_target_ns = current_ns in TARGET_NAMESPACES

        elif event == "end" and tag == "title" and not in_revision:
            current_page_title = elem.text

        elif event == "end" and tag == "id" and not in_revision and not in_contributor:
            current_page_id = elem.text

        # ---- REVISION-LEVEL ELEMENTS ----
        elif event == "start" and tag == "revision":
            if in_target_ns:
                in_revision = True
                rev = {
                    "page_id": current_page_id,
                    "page_title": current_page_title,
                    "namespace": current_ns,
                    "namespace_name": NS_NAMES.get(current_ns, str(current_ns)),
                    "revision_id": None,
                    "parent_revision_id": None,
                    "timestamp": None,
                    "contributor_username": None,
                    "contributor_id": None,
                    "contributor_ip": None,
                    "edit_comment": None,
                    "text_content": None,
                    "text_bytes": None,
                    "sha1": None,
                }

        elif event == "end" and tag == "revision":
            if in_target_ns and in_revision:
                yield rev
            in_revision = False
            rev = {}
            # Free memory for processed elements
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
                    rev["text_content"] = elem.text
                    rev["text_bytes"] = elem.get("bytes")
                elif tag == "sha1":
                    rev["sha1"] = elem.text

        # Free memory for page-level elements once done
        elif event == "end" and tag == "page":
            elem.clear()


def main():
    log.info("=== Simple English Wikipedia Ingestion ===")
    log.info(f"Source: {XML_PATH} ({XML_PATH.stat().st_size / 1e9:.1f} GB)")
    log.info(f"Target namespaces: {TARGET_NAMESPACES}")

    out_path = OUT_DIR / "wikipedia_raw.parquet"
    chunk: list[dict] = []
    total_revisions = 0
    chunk_num = 0
    writer = None

    for rev in tqdm(stream_talk_revisions(XML_PATH), desc="revisions", unit="rev"):
        chunk.append(rev)
        total_revisions += 1

        if len(chunk) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(chunk)
            if writer is None:
                import pyarrow as pa
                import pyarrow.parquet as pq
                schema = pa.Schema.from_pandas(df_chunk)
                writer = pq.ParquetWriter(str(out_path), schema)
            writer.write_table(pa.Table.from_pandas(df_chunk))
            chunk_num += 1
            log.info(f"  Wrote chunk {chunk_num} ({total_revisions:,} revisions so far)")
            chunk.clear()

    # Write final partial chunk
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

    log.info(f"Total revisions written: {total_revisions:,}")
    log.info(f"Saved -> {out_path}")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
