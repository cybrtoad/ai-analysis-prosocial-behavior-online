# Human Validation Kit — Prosocial Interventions in Online Moderation

**Project**: Tamara Linse, EECS, University of Wyoming

---

## Purpose

This kit contains everything needed for the human annotation phase of the research. Three trained reviewers will independently rate 300 conversation records to:

1. Establish inter-rater reliability for the Tier 1 prosociality scoring rubric
2. Validate that Claude's AI-generated labels are well-calibrated
3. Produce gold-standard human-consensus labels for the final stage of DeBERTa training

---

## Files in This Directory

| File | What it is |
|---|---|
| `README.md` | This file — start here |
| `REVIEWER_INSTRUCTIONS.md` | Step-by-step instructions for reviewers; read before rating |
| `RUBRIC.md` | Detailed rubric for the four Tier 1 behavioral dimensions |
| `INTERVENTION_STYLE_GUIDE.md` | Coding guide for the five intervention style categories |
| `records_to_rate.csv` | The 300 conversation records to be rated |
| `rating_form.csv` | Blank form — one copy per reviewer; fill in and return |

---

## What Reviewers Rate

Each record contains three texts:

- **Pre-intervention text** — what the user wrote before being moderated (the problematic content)
- **Intervention text** — what the moderator said or did
- **Post-intervention text** — what the user wrote next (this is what you are scoring)

**You are rating the post-intervention text only.**

You are NOT evaluating whether the moderator did a good job. You are evaluating how the user responded.

---

## Scoring Overview

### Tier 1 — Four Behavioral Dimensions (1.0–5.0 scale)

Rate the post-intervention text on each of these four dimensions independently:

| Dimension | One-line definition |
|---|---|
| Empathy | Does the user acknowledge others' feelings or perspectives? |
| Constructiveness | Does the response add substance or move toward resolution? |
| Respect | Does the user maintain polite, non-hostile norms? |
| Social Cohesion | Does the user try to repair or maintain relational harmony? |

Scale: 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0

See `RUBRIC.md` for detailed anchors and examples.

### Intervention Style — One Categorical Label

Also code what kind of intervention the moderator used (not the user's response — this time you are evaluating the intervention text):

| Code | Meaning |
|---|---|
| punitive | Focuses on punishment, censure, or blame |
| educative | Explains rules, norms, or the reasoning behind them |
| suggestive | Offers an alternative approach or reframes the situation |
| positive | Affirms and encourages prosocial behavior |
| structural | Platform-level action without direct communication |

See `INTERVENTION_STYLE_GUIDE.md` for definitions and examples.

---

## Timeline

- Records available: see `records_to_rate.csv`
- Return completed `rating_form.csv` to Tamara by the agreed deadline
- Do NOT discuss specific records with other reviewers until all three ratings are submitted

---

## Questions

Contact Tamara Linse with any questions about the rubric, records, or process.
