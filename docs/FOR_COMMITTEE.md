# Thesis Committee Guide

This document maps the code and outputs in this repository to the methodology and findings described in the thesis. It is intended for thesis committee members reviewing the implementation.

**Thesis:** Linse, T. T. (2026). *Who Corrects and How: Large-Scale AI-Assisted Analysis of Communication Style and Prosocial Behavioral Response Across Peer, Expert, and Authority Contexts.* M.S. thesis, Department of Electrical Engineering and Computer Science, University of Wyoming.

---

## A Note on Naming

Scripts, directories, and output files use the internal name `phase2` (e.g., `src/analyze_phase2.py`, `output/phase2/`, `phase2_tables/`) to refer to what the thesis calls the large-scale analysis. The naming reflects the development history and has been left unchanged to avoid introducing errors.

---

## Overview of the Pipeline

The full pipeline runs in eight stages. All intermediate and final outputs are pre-computed and included in `output/artifacts/` — the pipeline does not need to be re-run to examine results.

```
Raw platform data
    │
    ├── src/ingest_reddit.py
    ├── src/ingest_stackexchange.py
    ├── src/ingest_wikipedia.py          (NS3 User talk pages)
    └── src/ingest_wikipedia_ns0.py      (NS0 article edits)
    │
    ▼
src/standardize.py          → output/processed/unified_interventions.parquet
    │                           Unified schema: record_id, platform, user_id,
    │                           pre/intervention/post text, intervention_type,
    │                           intervener_role, timestamp, response_time_minutes
    ▼
src/sample.py               → output/processed/labeling_sample.parquet
src/split.py                → output/splits/
    │
    ▼
src/label.py                → output/labels/gold_labels.parquet
    │                           Claude API scores each record on 6 dimensions
    ▼
src/phase2_sample.py        → output/phase2/phase2_sample.parquet
src/build_phase2_features.py
    │
    ▼
src/analyze_phase2.py       → output/artifacts/phase2_report.txt + phase2_tables/
src/analyze_tier_d.py       → output/artifacts/tier_d_report.txt + tier_d_tables/
src/analyze_confidence.py   → output/artifacts/confidence_report.txt
src/audit_phase2_scores.py  → output/artifacts/audit_report.txt
src/sensitivity_respect.py  → output/artifacts/sensitivity_respect.txt
    │
    ▼
src/process_human_annotations.py
src/compare_claude_vs_humans.py → output/artifacts/claude_human_validation_report.txt
```

---

## Intervention Triple Construction

**Script:** `src/standardize.py`

Each record is an *intervention triple*: three consecutive texts from the same interaction thread involving the same target user.

| Field | Description |
|---|---|
| `pre_intervention_text` | What the target user wrote before receiving a correction |
| `intervention_text` | The corrective communication directed at the target user |
| `post_intervention_text` | The target user's response after the intervention |

**Platform-specific construction:**

- **Reddit:** Triples are built from conversation reply chains. The pre text is the parent utterance, the intervention is the reply from a different speaker, and the post is the original speaker's next utterance in the same conversation.
- **Stack Exchange:** The pre text is the original post body. The intervention is a comment from a different user. The post is the post owner's next comment after the intervention.
- **Wikipedia (cross-namespace design):** The pre text is the user's most recent NS0 article edit within 48 hours before the intervention. The intervention is content added to the user's NS3 User talk page by another editor. The post text is the user's first NS0 article edit after the intervention. This cross-namespace design avoids the ~80% invalid_triple rate of the earlier NS3-only approach (legacy function retained in `standardize.py` for reference).

**Intervener role** is derived deterministically from platform and intervention type (`src/standardize.py`, `_derive_intervener_role()`):
- Reddit → `peer`
- Stack Exchange → `domain_expert`
- Wikipedia + warning/restriction → `formal_authority`
- Wikipedia + reframe/positive_reinforcement → `peer_editor`

---

## Prosociality Scoring (Claude API)

**Script:** `src/label.py`

Each post-intervention response is scored by the Claude API on six dimensions using a structured rubric. The complete system prompt and per-dimension rubrics are in `src/label.py` (lines 110–255).

**Four Tier 1 dimensions (scored 1.0–5.0 in 0.5 increments):**

| Dimension | What it measures |
|---|---|
| `empathy` | Acknowledgment of others' feelings or perspective |
| `constructiveness` | Adds substance and moves toward resolution |
| `respect` | Maintains polite, non-hostile interaction norms |
| `social_cohesion` | Attempts to repair or maintain relational harmony |

**Two Tier 3 dimensions (platform-specific):**

| Dimension | What it measures |
|---|---|
| `deliberative_quality` | Open-mindedness and quality of argumentation |
| `task_norm_alignment` | Platform-specific norm adherence (collaborative editing intent / technical helpfulness / subreddit norms) |

The API also returns a `confidence` score (0.0–1.0) and an `invalid_triple` flag for records that do not represent genuine prosocial corrective exchanges.

**Model used:** `claude-haiku-4-5-20251001` (cost-efficient at ~100K record scale; see `DEFAULT_MODEL` in `label.py`)

**Checkpointing:** The labeler saves a checkpoint parquet every 50 records and automatically resumes from the last checkpoint if interrupted.

---

## Tiered Analysis Design

**Script:** `src/analyze_phase2.py`, `src/analyze_tier_d.py`

Rather than applying a hard filter at ingestion, all records are retained and assigned an analysis tier at query time based on how rich their pre- and post-intervention contexts are. Tiers are **inclusive** — a Tier A record also qualifies for Tiers B, C, and D. Pre- and post-intervention post counts are stored as covariates, not filters, so every record contributes to all analyses for which it has sufficient data.

| Tier | Approx. N | Criteria | Analytical scope |
|---|---|---|---|
| A | ~8,000 | 5+ pre-intervention posts; 10+ post-intervention posts | Sustained behavior analysis; full outcome battery |
| B | ~15,000 | 2+ pre-intervention posts; 3+ post-intervention posts | Immediate behavioral change; short-term trajectory |
| C | ~40,000 | 1+ pre-intervention post; 1+ post-intervention post | Single-point Tier 1 prosociality scoring (core analysis) |
| D | ~100,000 | 1+ pre-intervention post; no post-intervention posts | Outcome and retention analysis only |

Tier D records cannot support behavioral change measurement but carry analytical value: the absence of post-intervention activity may indicate the user left the platform following the intervention. `src/analyze_tier_d.py` analyzes departure rates by platform, intervention type, and intervener role.

A hard filter requiring 5+ pre and 10+ post posts would retain only ~8,000 records and would systematically over-represent established, high-engagement users. The tiered design avoids this attrition bias.

---

## Validation Against Human Raters

**Scripts:** `src/process_human_annotations.py`, `src/compare_claude_vs_humans.py`  
**Data:** `human-validation/`  
**Output:** `output/artifacts/claude_human_validation_report.txt` and `.json`

Three human raters independently scored the same set of records (`human-validation/records_to_rate.csv`) using the rubric in `human-validation/RUBRIC.md` and the instructions in `human-validation/REVIEWER_INSTRUCTIONS.md`.

**Agreement is measured with:**
- Intraclass Correlation Coefficient (ICC) for inter-rater reliability among human raters
- Pearson and Spearman correlation between Claude scores and human consensus
- Bland-Altman analysis for systematic bias (Figure 9 in `figures/`)

Claude's ratings of the same records are in `human-validation/claude_scores_EMBARGOED.csv`. Individual rater files: `rating_form_gd_9942.csv`, `rating_form_tl_3029.csv`, `rating_form_vt_2755.xlsx`.

---

## Key Output Files and Thesis Table Mapping

| Output file | Thesis content |
|---|---|
| `output/artifacts/phase2_report.txt` | Main Phase 2 findings narrative |
| `output/artifacts/phase2_tables/rq1_intervention_type_prosociality.csv` | RQ1: intervention type → prosociality |
| `output/artifacts/phase2_tables/rq2b_intervener_role_prosociality.csv` | RQ2: intervener role → prosociality |
| `output/artifacts/phase2_tables/rq4_platform_comparison.csv` | RQ3/RQ4: platform comparison |
| `output/artifacts/phase2_tables/dimension_pca.csv` | PCA of four prosociality dimensions |
| `output/artifacts/phase2_tables/dimension_correlations.csv` | Dimension correlation matrix |
| `output/artifacts/tier_d_report.txt` + `tier_d_tables/` | Tier D departure analysis |
| `output/artifacts/claude_human_validation_report.txt` | Validation: ICC, correlation, Bland-Altman |
| `output/artifacts/confidence_report.txt` | Annotation confidence analysis |
| `output/artifacts/sensitivity_respect.txt` | Respect dimension sensitivity analysis |
| `output/artifacts/wikipedia/` | Wikipedia-only replication of main analysis |
| `figures/figure4_platform_violin.*` | Figure 4: prosociality by platform |
| `figures/figure5_reddit_intervention_bar.*` | Figure 5: intervention type effects (Reddit) |
| `figures/figure6_tier_d_departure_rates.*` | Figure 6: Tier D departure rates |
| `figures/figure7_correlation_heatmap.*` | Figure 7: dimension correlation heatmap |
| `figures/figure8_pca_scree.*` | Figure 8: PCA scree plot |
| `figures/figure9_bland_altman.*` | Figure 9: Bland-Altman validation plot |
| `figures/figure10_wikipedia_gradient.*` | Figure 10: Wikipedia gradient analysis |

---

## Sensitivity Analysis

**Script:** `src/sensitivity_respect.py`  
**Output:** `output/artifacts/sensitivity_respect.txt`

The `respect` dimension showed the highest systematic divergence between Claude scores and human rater consensus. The sensitivity analysis re-runs the main Phase 2 analyses excluding `respect` to confirm that the central finding (platform dominance) is robust to this dimension's measurement uncertainty.

---

## Annotation Confidence and Boundary Case Analysis

**Script:** `src/analyze_confidence.py`  
**Output:** `output/artifacts/confidence_report.txt`, `confidence_tables/`

The Claude API returns a `confidence` score (0.0–1.0) alongside each set of dimension scores. The confidence analysis examines:
- Whether low-confidence records show higher score variance
- Distribution of confidence by platform and intervention type
- Relationship between confidence and deviation from human rater consensus

Boundary cases (records with confidence < 0.5 or high inter-rater disagreement) were reviewed manually to diagnose annotation uncertainty and inform rubric refinement.

---

## Reproducing the Analysis Without Re-Running Labeling

All pre-computed labeled data needed for the statistical analyses is in `output/artifacts/`. To re-run just the statistical analysis and figure generation without re-calling the Claude API:

```bash
python src/analyze_phase2.py
python src/analyze_tier_d.py
python src/analyze_confidence.py
python src/audit_phase2_scores.py
python src/sensitivity_respect.py
python src/process_human_annotations.py
python src/compare_claude_vs_humans.py
python scripts/create_thesis_figures.py
```

To re-run the full labeling pipeline from scratch, you will need an Anthropic API key and the raw platform data (see [DATA.md](DATA.md)). At ~100K records using `claude-haiku-4-5-20251001`, expect approximately $20–40 in API costs depending on record length.

---

## A Note on DeBERTa Training

The initial project plan included the training of two versions of an LLM (DeBERTa-large and DeBERTa-base) as artifacts for use by future researchers. This aspect of the project was removed during development. Some scripts (/src/split.py) retain this reference, however.  
