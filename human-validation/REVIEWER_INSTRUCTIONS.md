# Reviewer Instructions

## What This Study Is

This research investigates patterns in online moderation interventions across Reddit, StackExchange (Ask Ubuntu), and Wikipedia. Specifically, it examines what types of moderator responses are associated with prosocial behavior from the users who receive them.

You are helping build a gold-standard labeled dataset that will be used to:
1. Validate an AI labeling system (Claude)
2. Train a machine learning model (DeBERTa) to score prosocial behavior at scale

Your careful, independent judgments are the foundation the entire system is validated against.

---

## Before You Begin

1. Read `RUBRIC.md` in full. Pay particular attention to the divergence examples — they show cases where dimensions score differently from each other.
2. Read `INTERVENTION_STYLE_GUIDE.md` in full, including the decision rules for ambiguous cases.
3. Open `records_to_rate.csv` in a spreadsheet application (Excel, Google Sheets, LibreOffice Calc).
4. Open `rating_form.csv` as a separate file — this is where you record your scores.
5. Do **not** discuss specific records with the other reviewers until you have submitted your completed form.

---

## The Records

`records_to_rate.csv` contains 300 conversation records from two platforms:
- **Reddit** (Change My View subreddit) — discussion-style conversations
- **StackExchange** (Ask Ubuntu) — technical Q&A discussions

Each row contains:

| Column | What it is |
|---|---|
| `record_id` | Unique identifier — use this in your rating form |
| `platform` | `reddit` or `stackexchange` |
| `intervention_type` | Pipeline-detected type (for context only — do not use for coding style) |
| `pre_intervention_text` | What the user wrote before moderation |
| `intervention_text` | What the moderator said or did |
| `post_intervention_text` | What the user wrote next — **this is what you are scoring** |

---

## The Rating Form

`rating_form.csv` has one row per record. Fill in:

| Column | What to enter |
|---|---|
| `record_id` | Already filled in — do not change |
| `rater_id` | Your initials or assigned ID |
| `empathy` | Score 1.0–5.0 in 0.5 steps |
| `constructiveness` | Score 1.0–5.0 in 0.5 steps |
| `respect` | Score 1.0–5.0 in 0.5 steps |
| `social_cohesion` | Score 1.0–5.0 in 0.5 steps |
| `intervention_style` | One of: punitive / educative / suggestive / positive / structural |
| `confidence_tier1` | Your confidence in the four Tier 1 scores: low / medium / high |
| `confidence_style` | Your confidence in the intervention style code: low / medium / high |
| `notes` | Optional — flag anything unclear, ambiguous, or unusual |

---

## Step-by-Step Rating Process

### For each record:

**Step 1**: Read all three texts in order: pre → intervention → post.

**Step 2**: Read the **post-intervention text** again carefully. This is the text you are scoring for Tier 1.

**Step 3**: Score **Empathy** (1.0–5.0). Ask: does this user acknowledge others' feelings or perspectives? Check RUBRIC.md if unsure.

**Step 4**: Score **Constructiveness** (1.0–5.0). Ask: does this response add substance or move toward resolution?

**Step 5**: Score **Respect** (1.0–5.0). Ask: does the user maintain polite, non-hostile norms?

**Step 6**: Score **Social Cohesion** (1.0–5.0). Ask: does the user try to repair or maintain relational harmony?

**Step 7**: Read the **intervention text**. Code its **intervention style** (punitive / educative / suggestive / positive / structural). Check INTERVENTION_STYLE_GUIDE.md if unsure.

**Step 8**: Enter your confidence ratings and any notes.

**Step 9**: Move to the next record.

---

## Handling Difficult Cases

### Very short post-intervention texts
Some responses are brief ("okay," "thanks," "noted"). Score what is there. A brief acknowledgment typically scores around 2.5–3.5 on Respect (polite but minimal) and low on Constructiveness (nothing substantive added). Do not assume positive intent not demonstrated in the text.

### The user seems to be ignoring the intervention
Score the text as written. If the post-intervention response addresses a different topic entirely and makes no reference to the intervention, score it on its own merits. Low empathy, low social cohesion, and moderate respect/constructiveness is a reasonable profile for this pattern.

### The user seems very upset or defensive
Score the content, not your sympathy. A defensive but substantive response may still score reasonably on Constructiveness even if it scores low on Empathy and Social Cohesion.

### Technical Wikipedia edits
For Wikipedia records, the post-intervention text is an article edit. It may look like: [edit comment] + [diff showing text added/removed]. Score the edit's prosociality: does the edit acknowledge the intervention implicitly (empathy)? Does it add something meaningful to the article (constructiveness)? Is the edit comment respectful (respect)? Does it seem oriented toward collaborative improvement (social cohesion)?

### Records that seem like they don't belong
A small number of records may appear to not be a genuine intervention sequence — for example, the "intervention" is actually just a regular conversation reply, not a moderation action. Note this in the `notes` column. Do not leave the row blank; score it as best you can and flag it.

### Ambiguous intervention style
If you genuinely cannot decide between two styles, write both in the notes column (e.g., "educative/suggestive") and use your confidence rating to flag it as low. These will be discussed in the group adjudication meeting.

---

## Things to Avoid

- **Do not look up the original threads** or research the conversation context beyond what is provided. You should score based solely on the text in the record.
- **Do not discuss specific records** with the other reviewers until after all ratings are submitted.
- **Do not anchor to platform expectations**. A StackExchange response is not automatically high on constructiveness because the platform rewards technical answers. Score the actual text.
- **Do not penalize brevity** on dimensions where it is not relevant. A short response can score 5.0 on Respect if it is genuinely warm and courteous.
- **Do not reward length**. A long response that restates the same position scores low on Constructiveness even if it is elaborate.

---

## Submitting Your Ratings

Save your completed `rating_form.csv` with your rater ID in the filename (e.g., `rating_form_TL.csv`) and return it to Tamara by the agreed deadline.

If you have questions about a specific record or the rubric, contact Tamara — do not ask the other reviewers.

---

## After All Ratings Are Submitted

Once all three reviewers have submitted their ratings, the research team will:

1. Compute inter-rater reliability (ICC for Tier 1 scores; Fleiss' Kappa for intervention style)
2. Identify records where reviewers disagreed substantially
3. Hold a short adjudication meeting where disagreements are discussed and consensus scores are recorded

Your individual ratings are preserved; the consensus score is recorded separately. Both are used in the analysis.

---

## Thank You

This annotation is essential to the research. The human consensus labels you produce are the gold standard that the entire AI labeling system is evaluated against. Your careful, independent judgment directly shapes the quality of the findings.
