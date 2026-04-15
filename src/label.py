"""
Step 5 & 6: Claude API Labeling
=================================
Scores each post-intervention response on four Tier 1 prosocial dimensions and
two Tier 3 platform-specific dimensions using Claude.

Tier 1 dimensions (1.0–5.0 scale):
  - empathy              : Acknowledgment of others' feelings/perspectives
  - constructiveness     : Adds substance, moves toward resolution
  - respect              : Maintains polite, non-hostile interaction norms
  - social_cohesion      : Attempts to repair/unify/maintain relational harmony

Tier 3 dimensions (1.0–5.0 scale):
  - deliberative_quality : Open-mindedness, argumentation quality, receptiveness
  - task_norm_alignment  : Platform-specific norm adherence (Wikipedia: collaborative
                           editing intent; StackExchange: addresses the technical
                           question; Reddit: aligns with subreddit norms)

Input:
  output/processed/labeling_sample.parquet

Output:
  output/labels/gold_labels.parquet

Usage:
  # Test on 10 records first (prints to stdout, does NOT write output)
  python src/label.py --test 10

  # Full run (auto-resumes if output already exists)
  python src/label.py

  # Limit to first N records (useful for incremental runs)
  python src/label.py --limit 100

  # Force re-label everything even if output exists
  python src/label.py --force

  # Use a different model
  python src/label.py --model claude-sonnet-4-6

Features:
  - Checkpointing: saves every CHECKPOINT_EVERY records; auto-resumes from prior run
  - Retry: exponential backoff on API errors (up to MAX_RETRIES attempts)
  - Rate limiting: up to MAX_RPS requests per second
  - Cost tracking: prints estimated token usage and cost at end
  - Bad response flagging: malformed JSON stored with score=-1 for manual review
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_SAMPLE_PATH = ROOT / "output" / "processed" / "labeling_sample.parquet"
DEFAULT_LABELS_PATH = ROOT / "output" / "labels" / "gold_labels.parquet"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "label.log", encoding="utf-8"),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Labeling parameters
# ---------------------------------------------------------------------------
DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
MAX_RETRIES      = 4         # total attempts per record
BASE_BACKOFF_S   = 2.0       # seconds (doubles each retry)
MIN_DELAY_S      = 0.3       # minimum pause between requests (~3 RPS max)
CHECKPOINT_EVERY = 50        # save parquet every N records
MAX_TOKENS_OUT   = 1024      # ceiling for JSON response (Tier 3 adds ~150 tokens)

DIMENSIONS       = ["empathy", "constructiveness", "respect", "social_cohesion"]
TIER3_DIMENSIONS = ["deliberative_quality", "task_norm_alignment"]
ALL_DIMENSIONS   = DIMENSIONS + TIER3_DIMENSIONS

# Approximate token cost per MTok (as of early 2025)
TOKEN_COST = {
    "claude-haiku-4-5-20251001":  {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":          {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":            {"input": 15.00, "output": 75.00},
}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert social scientist analyzing online interactions for prosocial behavior.
You will be given an interaction triple from a social media platform:
  1. PRE-INTERVENTION TEXT: what the target user wrote before receiving a corrective communication
  2. CORRECTIVE COMMUNICATION: what another community member said or did — this may be a peer
     challenge (Reddit), a domain expert's guidance (StackExchange), or a formal moderator's
     warning or reframe (Wikipedia), depending on the platform and intervener role indicated
  3. POST-INTERVENTION RESPONSE: how the target user responded afterward

Your task is to score the POST-INTERVENTION RESPONSE on six dimensions (four general,
two platform-specific). Score each dimension from 1.0 to 5.0 in increments of 0.5.

IMPORTANT: You are scoring ONLY the post-intervention response — not the corrective communication itself.

──────────────────────────────────────────────────────
TIER 1 RUBRICS (all platforms)
──────────────────────────────────────────────────────

EMPATHY — Does the response acknowledge the other person's feelings, perspective, or intent?
  1.0 Completely ignores or dismisses the intervention; no acknowledgment of the other's
      viewpoint; purely defensive, hostile, or self-centered.
  3.0 Partial acknowledgment; the responder shows mild awareness of the other's point but
      doesn't fully engage with it; mixed or ambivalent tone.
  5.0 Clear, genuine acknowledgment of the other's perspective or feelings; explicitly or
      implicitly validates the intervention; shows real understanding.

CONSTRUCTIVENESS — Does the response add meaningful substance and move toward resolution?
  1.0 Adds nothing useful; purely off-topic, evasive, or oppositional without new substance;
      the conversation cannot move forward from this response.
  3.0 Contains some useful information or partial engagement; moves the conversation slightly
      forward but misses clear opportunities to resolve the issue.
  5.0 Clearly advances the conversation; provides new information, accepts valid points,
      or proposes a path forward; meaningfully resolves or reduces the tension.

RESPECT — Does the response maintain polite, non-hostile interaction norms?
  1.0 Contains insults, slurs, sarcasm used as a weapon, name-calling, or overtly hostile
      language; treats the other person with contempt.
  3.0 Generally neutral but may include mildly dismissive phrasing, slight condescension,
      or passive aggression; no outright hostility.
  5.0 Consistently polite and professional; treats the other party with dignity even in
      disagreement; tone is calm and considerate.

SOCIAL_COHESION — Does the response try to repair, unify, or maintain relational harmony?
  1.0 Actively escalates the conflict; increases hostility or entrenches the divide; makes
      a constructive relationship less likely.
  3.0 Neither escalates nor de-escalates; neutral response that leaves the relationship
      unchanged; no repair attempt but no further damage either.
  5.0 Actively builds bridges; finds common ground; de-escalates tension; expresses
      willingness to continue dialogue constructively; makes reconciliation more likely.

──────────────────────────────────────────────────────
TIER 3 RUBRICS (platform-specific)
──────────────────────────────────────────────────────

DELIBERATIVE_QUALITY — Is the response open-minded, well-reasoned, and receptive to
counterpoints? (Apply to all platforms)
  1.0 Closed-minded; dismisses counterpoints without engagement; no logical reasoning;
      asserts position without support.
  3.0 Partially open; acknowledges some counterpoints but selectively; reasoning is
      present but weak or inconsistent.
  5.0 Genuinely open-minded; engages thoughtfully with opposing views; provides clear
      reasoning; receptive to updating their position.

TASK_NORM_ALIGNMENT — Does the response align with the platform's core purpose and norms?
  Apply the rubric that matches the PLATFORM indicated in the user message:

  [WIKIPEDIA] Does the response show collaborative editing intent?
    1.0 No collaborative intent; self-interested, disruptive, or ignores editorial
        feedback; edits do not improve the article.
    3.0 Mixed signals; some attempt at collaboration but also self-interested elements;
        partially responsive to editorial feedback.
    5.0 Clear collaborative intent; focuses on improving the article; responsive to
        editorial feedback; aligns with Wikipedia's neutral-point-of-view norms.

  [STACKEXCHANGE] Does the response directly address the technical question?
    1.0 Completely off-topic; ignores the technical question; defensive or argumentative
        rather than helpful.
    3.0 Partially addresses the question; tangential or incomplete; some relevant
        technical content but misses the core issue.
    5.0 Directly and helpfully addresses the technical question; provides accurate,
        relevant information; moves toward solving the problem.

  [REDDIT] Does the response align with the subreddit's discussion norms?
    1.0 Clearly violates discussion norms; off-topic, inflammatory, or in bad faith
        for the community context.
    3.0 Partially aligned; generally on-topic but may violate norms in minor ways or
        miss the community's expectations.
    5.0 Fully aligned with community norms; engages in good faith with the subreddit's
        purpose; appropriate tone and content for the community.

──────────────────────────────────────────────────────
SPECIAL CASES
──────────────────────────────────────────────────────
- WIKIPEDIA TRIPLES — CRITICAL: For Wikipedia records, the post-intervention response is
  the user's next article edit after receiving a talk-page message. It will almost always
  be about a different article and will NOT address the intervention — this is expected
  and by design. DO NOT attempt to measure whether the editor "responded to" or
  "acknowledged" the talk-page message. DO NOT flag as invalid because the edit is
  topically unrelated to the intervention.
  Your task is to score the QUALITY of the post-intervention article edit itself,
  independently of whether it relates to the intervention topic. Apply each dimension to
  the edit's content and style:
    Empathy           → does the edit respect existing content and other editors' work?
    Constructiveness  → does the edit improve the article (accurate content, sources,
                        clear structure)?
    Respect           → is the edit conducted professionally (appropriate edit summary,
                        no hostile removals, follows policies)?
    Social Cohesion   → does the edit operate within collaborative norms (NPOV,
                        attribution, does not damage the editing environment)?
    Deliberative Qual → does the edit show balanced, thoughtful treatment of the subject?
    Task/Norm Align   → does the edit follow Wikipedia standards (sourcing, formatting,
                        neutral POV, verifiability)?
  NEVER set invalid_triple to true for a Wikipedia record as long as the
  post-intervention field contains actual article content. Score the edit on its own
  merits — you do not need to compare it to the pre-intervention edit or to the
  intervention topic.
- If the texts contain wiki markup ({{templates}}, [[links]], etc.), score based on the
  readable prose content, ignoring the markup syntax.
- If the post-intervention response is very short (under ~50 words), score conservatively:
  short responses may be polite without being constructive, etc.
- If the triple does not appear to be a genuine prosocial corrective exchange (e.g., the three
  texts are nearly identical, or there is no meaningful response), set
  "invalid_triple": true in your response and assign score 1.0 to all dimensions.

──────────────────────────────────────────────────────
OUTPUT FORMAT (strict JSON, no markdown fencing)
──────────────────────────────────────────────────────
{
  "empathy": <float 1.0–5.0, step 0.5>,
  "constructiveness": <float 1.0–5.0, step 0.5>,
  "respect": <float 1.0–5.0, step 0.5>,
  "social_cohesion": <float 1.0–5.0, step 0.5>,
  "deliberative_quality": <float 1.0–5.0, step 0.5>,
  "task_norm_alignment": <float 1.0–5.0, step 0.5>,
  "rationale": {
    "empathy": "<1–2 sentence explanation>",
    "constructiveness": "<1–2 sentence explanation>",
    "respect": "<1–2 sentence explanation>",
    "social_cohesion": "<1–2 sentence explanation>",
    "deliberative_quality": "<1–2 sentence explanation>",
    "task_norm_alignment": "<1–2 sentence explanation>"
  },
  "confidence": <float 0.0–1.0>,
  "invalid_triple": <true | false>
}
"""


def _build_user_message(row: dict) -> str:
    """Format the three-part interaction for the labeling prompt."""
    platform  = row.get("platform", "unknown").upper()
    itype     = row.get("intervention_type", "unknown")
    role      = row.get("intervener_role", "unknown")
    pre       = (row.get("pre_intervention_text") or "").strip()
    interv    = (row.get("intervention_text") or "").strip()
    post      = (row.get("post_intervention_text") or "").strip()

    return (
        f"PLATFORM: {platform} | INTERVENTION TYPE: {itype} | INTERVENER ROLE: {role}\n"
        f"\n"
        f"━━ PRE-INTERVENTION TEXT ━━\n{pre}\n"
        f"\n"
        f"━━ CORRECTIVE COMMUNICATION ━━\n{interv}\n"
        f"\n"
        f"━━ POST-INTERVENTION RESPONSE (score this) ━━\n{post}\n"
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict | None:
    """
    Extract and validate JSON from the model's response.
    Returns a validated dict, or None if parsing/validation fails.
    """
    # Strip any accidental markdown fencing
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    m = _JSON_RE.search(text)
    if not m:
        return None

    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    # Validate required fields (Tier 1 + Tier 3)
    valid_scores = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
    for dim in ALL_DIMENSIONS:
        val = obj.get(dim)
        if val is None:
            return None
        # Coerce to float and round to nearest 0.5
        try:
            fval = round(float(val) * 2) / 2
        except (TypeError, ValueError):
            return None
        if fval not in valid_scores:
            return None
        obj[dim] = fval

    if "confidence" in obj:
        try:
            obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
        except (TypeError, ValueError):
            obj["confidence"] = None

    obj.setdefault("invalid_triple", False)
    obj.setdefault("rationale", {})
    return obj


# ---------------------------------------------------------------------------
# Single-record labeler
# ---------------------------------------------------------------------------

def label_record(
    client: anthropic.Anthropic,
    row: dict,
    model: str,
    dry_run: bool = False,
) -> dict:
    """
    Call Claude to score one record.
    Returns a flat dict with dimension scores + metadata.
    Always returns a dict (uses sentinel values on failure).
    """
    user_msg = _build_user_message(row)
    record_id = row.get("record_id", "?")

    if dry_run:
        log.info(f"[DRY RUN] {record_id}")
        log.info("USER MESSAGE:\n" + user_msg)
        return {}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS_OUT,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text  = response.content[0].text
            in_tokens = response.usage.input_tokens
            out_tokens = response.usage.output_tokens

            parsed = _parse_response(raw_text)
            if parsed is None:
                log.warning(
                    f"  [{record_id}] attempt {attempt}: malformed JSON "
                    f"— raw: {raw_text[:120]!r}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_BACKOFF_S * (2 ** (attempt - 1)))
                    continue
                # Give up — store raw text for manual review
                return {
                    "record_id": record_id,
                    **{d: -1.0 for d in ALL_DIMENSIONS},
                    "confidence": None,
                    "invalid_triple": None,
                    "rationale": "{}",
                    "raw_response": raw_text[:1000],
                    "label_error": "malformed_json",
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                }

            return {
                "record_id": record_id,
                **{d: parsed[d] for d in ALL_DIMENSIONS},
                "confidence": parsed.get("confidence"),
                "invalid_triple": bool(parsed.get("invalid_triple", False)),
                "rationale": json.dumps(parsed.get("rationale", {})),
                "raw_response": None,
                "label_error": None,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            }

        except anthropic.RateLimitError as e:
            wait = BASE_BACKOFF_S * (2 ** attempt)
            log.warning(f"  [{record_id}] rate limit (attempt {attempt}); sleeping {wait:.0f}s")
            time.sleep(wait)
            last_error = e
        except anthropic.APIStatusError as e:
            wait = BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                f"  [{record_id}] API error {e.status_code} (attempt {attempt}); "
                f"sleeping {wait:.0f}s"
            )
            time.sleep(wait)
            last_error = e
        except Exception as e:
            wait = BASE_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(f"  [{record_id}] unexpected error (attempt {attempt}): {e}")
            time.sleep(wait)
            last_error = e

    return {
        "record_id": record_id,
        **{d: -1.0 for d in ALL_DIMENSIONS},
        "confidence": None,
        "invalid_triple": None,
        "rationale": "{}",
        "raw_response": str(last_error)[:500] if last_error else "max retries exceeded",
        "label_error": "api_error",
        "input_tokens": 0,
        "output_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Label intervention records with Claude")
    parser.add_argument(
        "--test", type=int, metavar="N", default=0,
        help="Print prompts + responses for first N records without saving (default: off)",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N", default=0,
        help="Process at most N records in total (0 = all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore existing gold_labels.parquet and re-label everything",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_SAMPLE_PATH), metavar="PATH",
        help="Input parquet of records to label (default: output/processed/labeling_sample.parquet)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_LABELS_PATH), metavar="PATH",
        help="Output parquet for labels (default: output/labels/gold_labels.parquet)",
    )
    args = parser.parse_args()

    sample_path = Path(args.input)
    labels_path = Path(args.output)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== Step 5/6: Claude API Labeling ===")
    log.info(f"Model  : {args.model}")
    log.info(f"Input  : {sample_path}")
    log.info(f"Output : {labels_path}")

    # ------------------------------------------------------------------
    # Load sample
    # ------------------------------------------------------------------
    sample_df = pd.read_parquet(sample_path)
    log.info(f"Loaded {len(sample_df):,} records from {sample_path.name}")

    if args.limit:
        sample_df = sample_df.head(args.limit)
        log.info(f"Limiting to first {args.limit} records")

    # ------------------------------------------------------------------
    # Test mode — print prompts & responses, do not save
    # ------------------------------------------------------------------
    if args.test:
        client = anthropic.Anthropic()
        n_test = min(args.test, len(sample_df))
        log.info(f"TEST MODE: labeling {n_test} diverse records\n")

        # Pick diverse records across platforms and types
        parts = []
        for _, grp in sample_df.groupby(["platform", "intervention_type"]):
            parts.append(grp.sample(min(2, len(grp)), random_state=42))
        test_rows = pd.concat(parts, ignore_index=True).head(n_test)

        for _, row in test_rows.iterrows():
            row_dict = row.to_dict()
            log.info(f"\n{'='*60}")
            log.info(f"record_id: {row_dict['record_id']}  [{row_dict['platform']} / {row_dict['intervention_type']}]")
            result = label_record(client, row_dict, model=args.model, dry_run=False)
            for k, v in result.items():
                if k not in ("raw_response", "input_tokens", "output_tokens"):
                    log.info(f"  {k}: {v}")
            log.info(f"  tokens: {result.get('input_tokens',0)}in / {result.get('output_tokens',0)}out")
            time.sleep(MIN_DELAY_S)

        log.info("\n=== Test complete — no output written ===")
        return

    # ------------------------------------------------------------------
    # Full labeling run
    # ------------------------------------------------------------------
    client = anthropic.Anthropic()

    # Checkpointing: load existing labels if present
    done_ids: set[str] = set()
    existing_rows: list[dict] = []

    if labels_path.exists() and not args.force:
        existing_df = pd.read_parquet(labels_path)
        done_ids = set(existing_df["record_id"].tolist())
        existing_rows = existing_df.to_dict("records")
        log.info(f"Resuming: {len(done_ids):,} records already labeled")

    todo_df = sample_df[~sample_df["record_id"].isin(done_ids)]
    log.info(f"Records to label: {len(todo_df):,}")

    if todo_df.empty:
        log.info("Nothing to do — all records already labeled.")
        return

    # ------------------------------------------------------------------
    # Label loop
    # ------------------------------------------------------------------
    results = list(existing_rows)
    total_in_tokens  = 0
    total_out_tokens = 0
    n_errors         = 0
    n_labeled        = 0
    t_start          = time.time()

    for i, (_, row) in enumerate(todo_df.iterrows(), start=1):
        row_dict = row.to_dict()
        record_id = row_dict.get("record_id", f"row_{i}")
        platform  = row_dict.get("platform", "?")
        itype     = row_dict.get("intervention_type", "?")

        log.info(f"[{i:4d}/{len(todo_df)}] {record_id}  [{platform}/{itype}]")

        result = label_record(client, row_dict, model=args.model)
        results.append(result)

        total_in_tokens  += result.get("input_tokens",  0)
        total_out_tokens += result.get("output_tokens", 0)
        if result.get("label_error"):
            n_errors += 1
            log.warning(f"  ERROR: {result['label_error']}")
        else:
            scores = {d: result[d] for d in ALL_DIMENSIONS}
            conf   = result.get("confidence", "?")
            log.info(f"  scores={scores}  confidence={conf}")
            n_labeled += 1

        # Rate limiting
        time.sleep(MIN_DELAY_S)

        # Checkpoint
        if i % CHECKPOINT_EVERY == 0:
            _save(results, labels_path)
            elapsed = time.time() - t_start
            rps = i / elapsed
            remaining = len(todo_df) - i
            eta_min = remaining / rps / 60 if rps > 0 else 0
            log.info(
                f"  [checkpoint] saved {len(results):,} rows | "
                f"{rps:.2f} rec/s | ETA ~{eta_min:.0f} min"
            )

    # Final save
    _save(results, labels_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_min = (time.time() - t_start) / 60
    log.info(f"\n=== Done in {elapsed_min:.1f} min ===")
    log.info(f"Records labeled this run: {n_labeled:,}  |  Errors: {n_errors:,}")
    log.info(f"Total token usage: {total_in_tokens:,} input / {total_out_tokens:,} output")

    costs = TOKEN_COST.get(args.model, {"input": 0, "output": 0})
    cost_usd = (
        total_in_tokens  / 1_000_000 * costs["input"]
        + total_out_tokens / 1_000_000 * costs["output"]
    )
    log.info(f"Estimated API cost: ${cost_usd:.4f}")

    # Distribution of scores
    labels_df = pd.read_parquet(labels_path)
    valid_df  = labels_df[labels_df["label_error"].isna()]
    log.info(f"\nScore statistics (n={len(valid_df):,} valid records):")
    for dim in ALL_DIMENSIONS:
        s = valid_df[dim].describe()
        log.info(f"  {dim:18s}: mean={s['mean']:.2f}  std={s['std']:.2f}  "
                 f"min={s['min']:.1f}  max={s['max']:.1f}")

    invalid_pct = valid_df["invalid_triple"].mean() * 100
    log.info(f"\nFlagged as invalid triple: {invalid_pct:.1f}%")
    log.info(f"Saved -> {labels_path}")


def _save(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
