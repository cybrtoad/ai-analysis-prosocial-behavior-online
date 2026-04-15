# Data Sources

This study uses three publicly available datasets. Raw data is **not included** in this repository — it must be downloaded separately. Pre-computed outputs from all pipeline steps are in `output/artifacts/`.

---

## Reddit — Conversations Gone Awry (ConvoKit)

**Source:** Cornell ConvoKit  
**License:** Reddit Terms of Service permit non-commercial academic research use. Findings must anonymize user information, and a copy of published results must be provided to reddit.research@reddit.com before publication.

The Conversations Gone Awry corpus contains Reddit conversations that were either removed by moderators or allowed to continue, making it well-suited for studying corrective communication and its effects.

**Download:**

```python
import convokit
corpus = convokit.Corpus(filename=convokit.download("conversations-gone-awry-corpus"))
```

Or via the ConvoKit documentation: https://convokit.cornell.edu/documentation/awry.html

**Ingestion script:** `src/ingest_reddit.py`  
**Output:** `output/raw/reddit_raw.parquet`

---

## Stack Exchange — Ask Ubuntu

**Source:** Stack Exchange Data Dump via Internet Archive  
**License:** Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0). Analytical reuse is permitted.

This study uses the Ask Ubuntu data dump archived on the Internet Archive (September 2025 snapshot, content through September 2024), prior to Stack Exchange's mid-2024 transition away from official Archive.org hosting.

**Download:** https://archive.org/download/stackexchange  
Look for `askubuntu.com.7z` in the file list.

Extract and place the XML files in a local directory, then run:

```bash
python src/ingest_stackexchange.py --input /path/to/askubuntu/
```

**Ingestion script:** `src/ingest_stackexchange.py`  
**Output:** `output/raw/stackexchange_posts.parquet`, `output/raw/stackexchange_comments.parquet`

---

## Simple English Wikipedia

**Source:** Wikimedia database dumps  
**License:** Creative Commons Attribution-ShareAlike (CC BY-SA). Wikimedia provides public dumps expressly for research and educational use.

This study uses two namespaces:
- **NS3 (User talk pages):** Source of intervention events — messages left on a user's talk page
- **NS0 (Article edits):** Source of pre- and post-intervention behavior — the user's article edits surrounding the intervention event

**Download:** https://dumps.wikimedia.org/simplewiki/

Download the most recent dump:
- `simplewiki-YYYYMMDD-pages-meta-history.xml.bz2` (full revision history — large file)

Then run:

```bash
python src/ingest_wikipedia.py --input simplewiki-YYYYMMDD-pages-meta-history.xml.bz2
python src/ingest_wikipedia_ns0.py --input simplewiki-YYYYMMDD-pages-meta-history.xml.bz2
```

**Ingestion scripts:** `src/ingest_wikipedia.py`, `src/ingest_wikipedia_ns0.py`  
**Output:** `output/raw/wikipedia_raw.parquet`, `output/raw/wikipedia_user_article_edits.parquet`

---

## Ethical Considerations

**No raw post or comment text is published in this repository.** Although the source data is technically public, reproducing verbatim text in the context of a study that scores individual posts for norm-violating behavior (low respect, low empathy, community departure) could cause reputational harm to users who had no knowledge of their inclusion. Verbatim text can be searched and traced back to its author even when usernames are withheld.

All repository contents — tables, figures, reports — are aggregate statistical results. Individual-level outputs (scored records, intermediate labeled files) are not distributed.

**Usernames** appear in raw data as structural identifiers for constructing intervention triples but are not present in any analytical output, report, or figure in this repository.

This research is observational and retrospective. No bots, automated accounts, or planted interventions were introduced into any platform.
