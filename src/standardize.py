"""
Step 3: Standardize to Unified Feature Schema
==============================================
Reads each platform's raw Parquet, constructs intervention triples (pre / intervention /
post), and maps them to a single unified schema.

Unified schema columns:
  record_id              str       - globally unique  e.g. reddit_000001
  platform               str       - reddit | wikipedia | stackexchange
  user_id                str       - the person being intervened upon
  pre_intervention_text  str       - their text that prompted the intervention
  intervention_text      str       - what was said/done to them
  intervention_type      str       - warning | reframe | positive_reinforcement | restriction
  post_intervention_text str       - their response after the intervention
  timestamp              datetime  - when the intervention occurred (UTC)
  response_time_minutes  float     - minutes from intervention to post-intervention
  platform_metadata      str       - JSON blob of platform-specific fields

Input:
  output/raw/reddit_raw.parquet
  output/raw/stackexchange_posts.parquet
  output/raw/stackexchange_comments.parquet
  output/raw/wikipedia_raw.parquet  (optional -- skipped if not yet complete)

Output:
  output/processed/unified_interventions.parquet
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
RAW_DIR = ROOT / "output" / "raw"
OUT_DIR = ROOT / "output" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "standardize.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

MIN_TEXT_LEN = 20   # characters
MAX_TRIPLES_PER_POST = 3   # cap SE triples per post


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text) -> str:
    """Strip and collapse excessive whitespace."""
    if text is None or (isinstance(text, float)):
        return ""
    return " ".join(str(text).split())


def _valid(pre: str, intervention: str, post: str) -> bool:
    return (
        len(pre) >= MIN_TEXT_LEN
        and len(intervention) >= MIN_TEXT_LEN
        and len(post) >= MIN_TEXT_LEN
    )


def _derive_intervener_role(platform: str, intervention_type: str) -> str:
    """
    Derive intervener role from platform and intervention type.
      reddit                                      -> peer
      stackexchange                               -> domain_expert
      wikipedia + warning | restriction           -> formal_authority
      wikipedia + reframe | positive_reinforcement -> peer_editor
    """
    if platform == "reddit":
        return "peer"
    if platform == "stackexchange":
        return "domain_expert"
    if platform == "wikipedia":
        if intervention_type in ("warning", "restriction"):
            return "formal_authority"
        return "peer_editor"
    return "unknown"


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

def build_reddit_records(raw_path: Path) -> list[dict]:
    log.info("=== Reddit ===")
    df = pd.read_parquet(raw_path)
    df = df.drop_duplicates("utterance_id")
    log.info(f"  Loaded {len(df):,} utterances across {df['conversation_id'].nunique():,} conversations")

    # Build fast lookup: utterance_id -> Series
    utt_by_id: dict = df.set_index("utterance_id").to_dict("index")

    records: list[dict] = []

    for conv_id, conv_df in tqdm(
        df.groupby("conversation_id"), desc="Reddit convs", unit="conv"
    ):
        conv_df = conv_df.sort_values("timestamp").reset_index(drop=True)

        # Map speaker -> list of utterance dicts (sorted by time)
        speaker_utts: dict[str, list] = {}
        for row in conv_df.to_dict("records"):
            speaker_utts.setdefault(row["speaker"], []).append(row)

        for row in conv_df.to_dict("records"):
            parent_id = row.get("reply_to")
            if not parent_id or not isinstance(parent_id, str):
                continue
            if parent_id not in utt_by_id:
                continue

            pre_row = utt_by_id[parent_id]
            pre_speaker = pre_row["speaker"]

            # Must be different speakers
            if pre_speaker == row["speaker"]:
                continue

            int_ts = row["timestamp"]

            # Find pre_speaker's earliest utterance after intervention
            post_candidates = [
                u for u in speaker_utts.get(pre_speaker, [])
                if u["timestamp"] is not None
                and int_ts is not None
                and int(u["timestamp"]) > int(int_ts)
                and u["utterance_id"] != parent_id
            ]
            if not post_candidates:
                continue

            post_row = post_candidates[0]

            pre_text = _clean(pre_row["text"])
            int_text = _clean(row["text"])
            post_text = _clean(post_row["text"])

            if not _valid(pre_text, int_text, post_text):
                continue

            # Intervention type
            flair = str(row.get("author_flair") or "")
            if row.get("stickied") is True:
                itype = "warning"
            elif "\u2206" in flair or "\u0394" in flair:   # ∆ or Δ
                itype = "positive_reinforcement"
            else:
                itype = "reframe"

            # Timestamp & response time
            try:
                int_dt = pd.Timestamp(int(int_ts), unit="s", tz="UTC")
                post_ts = post_row["timestamp"]
                response_mins = (int(post_ts) - int(int_ts)) / 60.0
            except (TypeError, ValueError):
                int_dt = pd.NaT
                response_mins = None

            records.append(
                {
                    "platform": "reddit",
                    "user_id": str(pre_speaker),
                    "pre_intervention_text": pre_text,
                    "intervention_text": int_text,
                    "intervention_type": itype,
                    "post_intervention_text": post_text,
                    "timestamp": int_dt,
                    "response_time_minutes": response_mins,
                    "platform_metadata": json.dumps(
                        {
                            "conversation_id": conv_id,
                            "pre_utterance_id": parent_id,
                            "intervention_utterance_id": row["utterance_id"],
                            "post_utterance_id": post_row["utterance_id"],
                            "conv_split": row.get("conv_split"),
                            "conv_has_removed_comment": bool(
                                row.get("conv_has_removed_comment")
                            ),
                            "intervention_score": row.get("score"),
                        },
                        default=str,
                    ),
                }
            )

    log.info(f"  -> {len(records):,} Reddit records")
    return records


# ---------------------------------------------------------------------------
# StackExchange
# ---------------------------------------------------------------------------

def build_stackexchange_records(
    posts_path: Path, comments_path: Path
) -> list[dict]:
    log.info("=== StackExchange ===")

    posts = pd.read_parquet(posts_path)
    comments = pd.read_parquet(comments_path)
    log.info(f"  Loaded {len(posts):,} posts, {len(comments):,} comments")

    # Parse dates
    posts["creation_date"] = pd.to_datetime(
        posts["creation_date"], utc=True, errors="coerce"
    )
    comments["creation_date"] = pd.to_datetime(
        comments["creation_date"], utc=True, errors="coerce"
    )

    # Drop posts with missing owner or body
    posts = posts.dropna(subset=["owner_user_id", "body_text"])
    posts = posts[posts["body_text"].str.strip().str.len() >= MIN_TEXT_LEN]
    log.info(f"  {len(posts):,} posts after owner/body filter")

    # Build post lookup: post_id -> post row dict
    posts_dict: dict = posts.set_index("post_id").to_dict("index")

    # Drop comments with missing user_id or text
    comments = comments.dropna(subset=["user_id", "text"])
    comments = comments[comments["text"].str.strip().str.len() >= MIN_TEXT_LEN]
    log.info(f"  {len(comments):,} comments after user/text filter")

    records: list[dict] = []

    for post_id, post_comments in tqdm(
        comments.groupby("post_id"), desc="SE posts", unit="post"
    ):
        if post_id not in posts_dict:
            continue

        post = posts_dict[post_id]
        owner_id = str(post["owner_user_id"])
        pre_text = _clean(post["body_text"])

        if len(pre_text) < MIN_TEXT_LEN:
            continue

        post_comments = post_comments.sort_values("creation_date").reset_index(drop=True)
        comments_list = post_comments.to_dict("records")

        triples_this_post = 0

        for i, intervention_comment in enumerate(comments_list):
            if triples_this_post >= MAX_TRIPLES_PER_POST:
                break

            commenter_id = str(intervention_comment.get("user_id") or "")
            if not commenter_id or commenter_id == owner_id:
                continue  # skip anonymous or owner's own comments

            int_text = _clean(intervention_comment["text"])
            if len(int_text) < MIN_TEXT_LEN:
                continue

            int_date = intervention_comment["creation_date"]

            # Find owner's next comment after this intervention
            owner_response = None
            for later in comments_list[i + 1:]:
                if str(later.get("user_id") or "") == owner_id:
                    owner_response = later
                    break

            if owner_response is None:
                continue

            post_text = _clean(owner_response["text"])
            if len(post_text) < MIN_TEXT_LEN:
                continue

            resp_date = owner_response["creation_date"]
            try:
                response_mins = (resp_date - int_date).total_seconds() / 60.0
            except Exception:
                response_mins = None

            records.append(
                {
                    "platform": "stackexchange",
                    "user_id": owner_id,
                    "pre_intervention_text": pre_text,
                    "intervention_text": int_text,
                    "intervention_type": "reframe",
                    "post_intervention_text": post_text,
                    "timestamp": int_date,
                    "response_time_minutes": response_mins,
                    "platform_metadata": json.dumps(
                        {
                            "post_id": post_id,
                            "post_type_id": post.get("post_type_id"),
                            "post_score": post.get("score"),
                            "post_title": post.get("title"),
                            "intervention_comment_id": intervention_comment["comment_id"],
                            "response_comment_id": owner_response["comment_id"],
                        },
                        default=str,
                    ),
                }
            )
            triples_this_post += 1

    log.info(f"  -> {len(records):,} StackExchange records")
    return records


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def _canonical_contributor(row: dict) -> str | None:
    """Return the best available contributor identifier."""
    return (
        row.get("contributor_username")
        or row.get("contributor_id")
        or row.get("contributor_ip")
    )


def _extract_added_content(text_before: str, text_after: str, cap: int = 3000) -> str:
    """
    Approximate the content added by an edit.
    For talk pages, new content is usually appended, so we return the tail
    that extends beyond the previous text.  Falls back to the full new text
    (capped) when the page shrank or the lengths are similar.
    """
    if not text_after:
        return ""
    if not text_before:
        return text_after[:cap]
    len_a = len(text_before)
    len_b = len(text_after)
    if len_b > len_a + MIN_TEXT_LEN:
        return text_after[len_a:].strip()[:cap]
    # Rewrite / revert: return the new full text (capped)
    return text_after[:cap]


def _format_article_edit(rev: dict) -> str:
    """
    Format an NS0 article edit as a single text for prosociality scoring.
    Combines the edit summary with an excerpt of the article text.
    """
    comment = _clean(str(rev.get("edit_comment") or ""))
    text    = _clean(str(rev.get("text_content") or ""))[:1500]
    if comment and text:
        return f"Edit summary: {comment}\n\nArticle excerpt:\n{text}"
    elif comment:
        return f"Edit summary: {comment}"
    else:
        return text


# Keyword sets shared by both Wikipedia functions
_WARNING_KWS     = ("warn", "vandal", "block", "spam", "uw-")
_RESTRICTION_KWS = ("revert", " rv ", "undid", "undo")
_POSITIVE_KWS    = ("welcome", "thanks")
_INTERVENTION_KWS = _WARNING_KWS + _RESTRICTION_KWS + _POSITIVE_KWS + ("civil",)
_INTERVENTION_TEMPLATES = (
    "{{uw-", "{{welcome", "{{warning", "{{bansummary",
    "{{indef-block", "{{blocked", "wp:civil",
)

PRE_WINDOW_HOURS = 48   # how far back to search for a pre-intervention article edit


def _wiki_intervention_type(comment: str) -> str:
    c = comment.lower()
    if any(k in c for k in _WARNING_KWS):
        return "warning"
    if any(k in c for k in _RESTRICTION_KWS):
        return "restriction"
    if any(k in c for k in _POSITIVE_KWS):
        return "positive_reinforcement"
    return "reframe"


def _extract_target_username(page_title: str) -> str | None:
    """'User talk:Angela' -> 'Angela';  returns None if pattern doesn't match."""
    prefix = "User talk:"
    if page_title and page_title.startswith(prefix):
        return page_title[len(prefix):]
    return None


def build_wikipedia_records_xns(talk_path: Path, ns0_path: Path) -> list[dict]:
    """
    Cross-namespace Wikipedia triple construction.

    Source 1 (talk_path): NS3 User talk revisions  -> intervention events
    Source 2 (ns0_path):  NS0 article edit revisions -> pre/post texts

    Triple:
      pre_intervention_text  = edit_comment + article excerpt of user's most
                               recent NS0 edit within PRE_WINDOW_HOURS before
                               the intervention timestamp
      intervention_text      = content appended to user's NS3 talk page
      post_intervention_text = edit_comment + article excerpt of user's first
                               NS0 edit after the intervention (empty string
                               when user never edits again -- filtered out by
                               sample.py min-length check)
    """
    from datetime import timedelta

    log.info("=== Wikipedia (cross-namespace) ===")

    import pyarrow.parquet as pq

    # ------------------------------------------------------------------ #
    # 1. Load NS3 talk data in batches to avoid OOM                      #
    # text_content is capped per row so large pages don't blow up RAM.   #
    # ------------------------------------------------------------------ #
    _TALK_COLS   = ["page_id", "page_title", "namespace", "revision_id",
                    "timestamp", "contributor_username", "contributor_ip",
                    "edit_comment", "text_content"]
    _TALK_TEXTCAP = 10_000  # chars — enough for diff; saves ~60 % RAM

    pf = pq.ParquetFile(talk_path)
    talk_by_page: dict[str, list[dict]] = {}
    total_revs = 0

    for _batch in pf.iter_batches(batch_size=50_000, columns=_TALK_COLS):
        _df = _batch.to_pandas()
        _df = _df[_df["namespace"] == 3]
        if _df.empty:
            continue
        _df["timestamp"] = pd.to_datetime(_df["timestamp"], utc=True, errors="coerce")
        _df = _df.dropna(subset=["timestamp", "text_content"])
        _df["text_content"] = _df["text_content"].str[:_TALK_TEXTCAP]
        for _pid, _grp in _df.groupby("page_id"):
            talk_by_page.setdefault(_pid, []).extend(_grp.to_dict("records"))
        total_revs += len(_df)

    log.info(f"  {total_revs:,} NS3 User talk revisions with timestamp and text")
    log.info(f"  {len(talk_by_page):,} unique NS3 pages")

    interventions: list[dict] = []

    for page_id, revs_unsorted in tqdm(
        talk_by_page.items(), desc="NS3 pages", unit="page"
    ):
        revs = sorted(revs_unsorted, key=lambda r: r["timestamp"])
        n = len(revs)
        if n < 2:
            continue

        page_title  = revs[0].get("page_title", "")
        target_user = _extract_target_username(page_title)
        if not target_user:
            continue

        for i in range(n - 1):
            rev_prev = revs[i]
            rev_curr = revs[i + 1]

            editor = _canonical_contributor(rev_curr)
            if not editor or editor == target_user:
                continue   # skip self-edits on own talk page

            comment = str(rev_curr.get("edit_comment") or "")
            text    = str(rev_curr.get("text_content") or "")
            is_kw   = any(k in comment.lower() for k in _INTERVENTION_KWS)
            is_tmpl = any(t in text.lower()    for t in _INTERVENTION_TEMPLATES)
            if not is_kw and not is_tmpl:
                continue

            int_text = _extract_added_content(
                str(rev_prev.get("text_content") or ""), text
            )
            if len(int_text) < MIN_TEXT_LEN:
                continue

            interventions.append({
                "target_user":         target_user,
                "intervention_ts":     rev_curr["timestamp"],
                "intervention_text":   int_text,
                "intervention_type":   _wiki_intervention_type(comment),
                "talk_page_title":     page_title,
                "intervention_rev_id": rev_curr.get("revision_id"),
                "edit_comment":        comment,
            })

    log.info(f"  Found {len(interventions):,} intervention events")
    if not interventions:
        log.warning("  No intervention events found -- returning empty list")
        return []

    # ------------------------------------------------------------------ #
    # 2. Load NS0 article edits filtered to target users only            #
    # ------------------------------------------------------------------ #
    # Collect the set of target usernames we actually need — this lets   #
    # pyarrow push the filter down to the parquet reader so we never     #
    # load the full 1.9 GB file into RAM.                                #
    target_users = {ev["target_user"] for ev in interventions}
    log.info(f"  Filtering NS0 parquet to {len(target_users):,} target users")

    table = pq.read_table(
        ns0_path,
        filters=[("contributor_username", "in", list(target_users))],
    )
    ns0_df = table.to_pandas()
    ns0_df["timestamp"] = pd.to_datetime(ns0_df["timestamp"], utc=True, errors="coerce")
    ns0_df = ns0_df.dropna(subset=["timestamp"])
    log.info(f"  {len(ns0_df):,} NS0 article revisions for target users loaded")

    ns0_by_user: dict[str, list[dict]] = {}
    for row in ns0_df.to_dict("records"):
        key = row.get("contributor_username") or row.get("contributor_ip")
        if key:
            ns0_by_user.setdefault(key, []).append(row)
    for key in ns0_by_user:
        ns0_by_user[key].sort(key=lambda r: r["timestamp"])
    log.info(f"  {len(ns0_by_user):,} unique contributors in NS0 data")

    # ------------------------------------------------------------------ #
    # 3. Assemble triples                                                  #
    # ------------------------------------------------------------------ #
    pre_window = timedelta(hours=PRE_WINDOW_HOURS)
    records: list[dict] = []

    for event in tqdm(interventions, desc="assembling triples", unit="event"):
        target     = event["target_user"]
        int_ts     = event["intervention_ts"]
        user_edits = ns0_by_user.get(target, [])
        if not user_edits:
            continue

        # Pre: most recent article edit within PRE_WINDOW_HOURS before intervention
        pre_candidates = [
            e for e in user_edits
            if e["timestamp"] < int_ts
            and (int_ts - e["timestamp"]) <= pre_window
        ]
        if not pre_candidates:
            continue
        pre_rev = pre_candidates[-1]

        # Post: first article edit after intervention (nullable)
        post_candidates = [e for e in user_edits if e["timestamp"] > int_ts]
        post_rev = post_candidates[0] if post_candidates else None

        pre_text  = _format_article_edit(pre_rev)
        int_text  = event["intervention_text"]
        post_text = _format_article_edit(post_rev) if post_rev else ""

        if not _valid(pre_text, int_text, pre_text):   # check pre and int lengths
            continue

        try:
            response_mins = (
                (post_rev["timestamp"] - int_ts).total_seconds() / 60.0
                if post_rev else None
            )
        except Exception:
            response_mins = None

        is_ip = bool(
            not pre_rev.get("contributor_username")
            and pre_rev.get("contributor_ip")
        )

        records.append(
            {
                "platform":               "wikipedia",
                "user_id":                str(target),
                "pre_intervention_text":  pre_text,
                "intervention_text":      int_text,
                "intervention_type":      event["intervention_type"],
                "post_intervention_text": post_text,
                "timestamp":              int_ts,
                "response_time_minutes":  response_mins,
                "platform_metadata": json.dumps(
                    {
                        "talk_page_title":          event["talk_page_title"],
                        "pre_article_title":        pre_rev.get("page_title"),
                        "post_article_title":       post_rev.get("page_title") if post_rev else None,
                        "intervention_revision_id": event["intervention_rev_id"],
                        "pre_article_revision_id":  pre_rev.get("revision_id"),
                        "post_article_revision_id": post_rev.get("revision_id") if post_rev else None,
                        "edit_comment":             event["edit_comment"],
                        "wiki_user_is_ip":          is_ip,
                    },
                    default=str,
                ),
            }
        )

    log.info(f"  -> {len(records):,} Wikipedia cross-namespace records")
    return records


def _build_wikipedia_records_legacy(raw_path: Path) -> list[dict]:
    """
    Original NS3-only Wikipedia triple construction.
    Kept as a fallback for when wikipedia_user_article_edits.parquet is not
    yet available. NOTE: This approach produced ~80% invalid_triple flags
    and should not be used for the final dataset.
    """
    log.info("=== Wikipedia (LEGACY NS3-only -- not recommended) ===")
    if not raw_path.exists():
        log.warning("  wikipedia_raw.parquet not found -- skipping")
        return []

    df = pd.read_parquet(raw_path)
    log.info(f"  Loaded {len(df):,} revisions")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["text_content"].notna()]
    log.info(f"  {len(df):,} revisions with timestamp and text")

    records: list[dict] = []

    for page_id, page_df in tqdm(
        df.groupby("page_id"), desc="Wiki pages (legacy)", unit="page"
    ):
        page_df = page_df.sort_values("timestamp").reset_index(drop=True)
        revs = page_df.to_dict("records")
        n = len(revs)
        if n < 3:
            continue

        contrib_revs: dict[str, list[int]] = {}
        for idx, rev in enumerate(revs):
            c = _canonical_contributor(rev)
            if c:
                contrib_revs.setdefault(c, []).append(idx)

        for i in range(n - 1):
            rev_a  = revs[i]
            rev_b  = revs[i + 1]
            a_user = _canonical_contributor(rev_a)
            b_user = _canonical_contributor(rev_b)
            if not a_user or not b_user or a_user == b_user:
                continue
            a_indices    = contrib_revs.get(a_user, [])
            post_indices = [idx for idx in a_indices if idx > i + 1]
            if not post_indices:
                continue
            rev_c = revs[post_indices[0]]
            text_a    = str(rev_a.get("text_content") or "")
            text_b    = str(rev_b.get("text_content") or "")
            text_c    = str(rev_c.get("text_content") or "")
            pre_text  = text_a[:2000].strip()
            int_text  = _extract_added_content(text_a, text_b)
            post_text = text_c[:2000].strip()
            if not _valid(pre_text, int_text, post_text):
                continue
            comment = str(rev_b.get("edit_comment") or "")
            itype   = _wiki_intervention_type(comment)
            int_ts  = rev_b["timestamp"]
            post_ts = rev_c["timestamp"]
            try:
                response_mins = (post_ts - int_ts).total_seconds() / 60.0
            except Exception:
                response_mins = None
            records.append(
                {
                    "platform":               "wikipedia",
                    "user_id":                str(a_user),
                    "pre_intervention_text":  pre_text,
                    "intervention_text":      int_text,
                    "intervention_type":      itype,
                    "post_intervention_text": post_text,
                    "timestamp":              int_ts,
                    "response_time_minutes":  response_mins,
                    "platform_metadata": json.dumps(
                        {
                            "page_id":                  str(page_id),
                            "page_title":               rev_a.get("page_title"),
                            "namespace":                rev_a.get("namespace"),
                            "pre_revision_id":          rev_a.get("revision_id"),
                            "intervention_revision_id": rev_b.get("revision_id"),
                            "post_revision_id":         rev_c.get("revision_id"),
                            "edit_comment":             rev_b.get("edit_comment"),
                        },
                        default=str,
                    ),
                }
            )

    log.info(f"  -> {len(records):,} Wikipedia records (legacy)")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Step 3: Standardize to Unified Feature Schema ===")

    all_records: list[dict] = []

    all_records.extend(
        build_reddit_records(RAW_DIR / "reddit_raw.parquet")
    )
    all_records.extend(
        build_stackexchange_records(
            RAW_DIR / "stackexchange_posts.parquet",
            RAW_DIR / "stackexchange_comments.parquet",
        )
    )
    ns0_path  = RAW_DIR / "wikipedia_user_article_edits.parquet"
    talk_path = RAW_DIR / "wikipedia_raw.parquet"
    if ns0_path.exists() and talk_path.exists():
        all_records.extend(build_wikipedia_records_xns(talk_path, ns0_path))
    elif talk_path.exists():
        log.warning(
            "wikipedia_user_article_edits.parquet not found -- "
            "falling back to legacy NS3-only construction (high invalid_triple rate).\n"
            "Run src/ingest_wikipedia_ns0.py first for correct cross-namespace triples."
        )
        all_records.extend(_build_wikipedia_records_legacy(talk_path))
    else:
        log.warning("No Wikipedia parquet found -- skipping Wikipedia.")

    log.info(f"Total records before dedup: {len(all_records):,}")

    if not all_records:
        log.error("No records produced -- aborting.")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    # Assign sequential record IDs
    df.insert(0, "record_id", [
        f"{row['platform']}_{i:06d}"
        for i, row in enumerate(all_records)
    ])

    # Derive intervener role from platform + intervention_type
    df["intervener_role"] = df.apply(
        lambda r: _derive_intervener_role(r["platform"], r["intervention_type"]), axis=1
    )

    out_path = OUT_DIR / "unified_interventions.parquet"
    df.to_parquet(out_path, index=False)

    log.info(f"Saved {len(df):,} records -> {out_path}")
    log.info("\nDistribution by platform x intervention_type x intervener_role:")
    dist = df.groupby(["platform", "intervention_type", "intervener_role"]).size().reset_index(name="count")
    log.info("\n" + dist.to_string(index=False))

    log.info("\nResponse time stats (minutes):")
    log.info(df["response_time_minutes"].describe().to_string())

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
