# Who Corrects and How

**Large-Scale AI-Assisted Analysis of Communication Style and Prosocial Behavioral Response Across Peer, Expert, and Authority Contexts**

Tamara T. Linse — M.S. Computer Science, University of Wyoming, 2026

---

## About

Online communities spend enormous resources managing norm-violating behavior, yet there is limited empirical evidence on which corrective communication strategies are most effective, whether the social role of the intervener matters, and how these patterns vary across platforms. Prior work is mostly small-scale and single-platform and focused on toxicity detection rather than response effectiveness.

This repository contains the full analysis pipeline, scoring outputs, and validation data for a study of prosocial corrective communication across three platforms—Reddit (Change My View Conversations Gone Awry corpus), Stack Exchange (Ask Ubuntu), and Simple English Wikipedia—using a large-scale AI-assisted annotation pipeline. Approximately 100,000 interaction triples (pre-intervention behavior, the intervention itself, and post-intervention response) were extracted and scored on four prosociality dimensions using the Claude API and then validated against independent human annotation from three raters.

**Thesis:** Linse, T. T. (2026). *Who Corrects and How: Large-Scale AI-Assisted Analysis of Communication Style and Prosocial Behavioral Response Across Peer, Expert, and Authority Contexts.* M.S. thesis, Department of Electrical Engineering and Computer Science, University of Wyoming. [Forthcoming in UWyo institutional repository]

---

## Key Findings

- **Community context, not intervention type, is the dominant predictor** of post-intervention prosociality. Platform explains more variance than what was said or who said it.
- **The four prosociality dimensions collapse into a single dominant factor.** PCA confirms that empathy, constructiveness, respect, and social cohesion are largely measuring one underlying construct at this scale.
- **The AI-assisted annotation pipeline is a noisy approximation of human judgment, with directional biases on specific scales rather than uniform underperformance.** (ICC and correlation metrics reported in `output/artifacts/claude_human_validation_report.txt`).

---

## Repository Structure

```
├── src/                        # Core analysis pipeline (Python)
│   ├── ingest_*.py             # Platform-specific data ingestion
│   ├── standardize.py          # Build unified intervention triples
│   ├── sample.py / split.py    # Stratified sampling and train/test splits
│   ├── label.py                # Claude API annotation (scoring pipeline)
│   ├── phase2_sample.py        # Phase 2 sample construction
│   ├── build_phase2_features.py
│   ├── analyze_phase2.py       # Main statistical analysis
│   ├── analyze_tier_d.py       # Tier D (departure) analysis
│   ├── analyze_confidence.py   # Annotation confidence analysis
│   ├── audit_phase2_scores.py  # Score distribution audit
│   ├── compare_claude_vs_humans.py  # Validation: Claude vs. human raters
│   ├── process_human_annotations.py
│   ├── sensitivity_respect.py  # Sensitivity analysis for respect dimension
│   └── validate_labels.py
│
├── scripts/                    # Utility and orchestration scripts
│   ├── add_intervener_role.py
│   ├── create_figures.py
│   └── format-xml-row.py
│
├── figures/                    # Thesis figures (PDF + PNG)
├── output/artifacts/           # All analysis outputs and tables
├── human-validation/           # Human annotation data and rubrics
│
├── CITATION.cff                # Machine-readable citation metadata
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) (required only to re-run the labeling step)

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and add your API key:

```
ANTHROPIC_API_KEY=your_key_here
```

---

## Running the Pipeline

The pipeline runs in sequential steps. Pre-computed outputs for all steps are included in `output/artifacts/` — you do not need to re-run from scratch to examine the results.

To reproduce from raw data:

```bash
# 1. Ingest raw platform data (requires downloaded datasets — see docs/DATA.md)
python src/ingest_reddit.py
python src/ingest_stackexchange.py
python src/ingest_wikipedia.py
python src/ingest_wikipedia_ns0.py

# 2. Build unified intervention triples
python src/standardize.py

# 3. Stratified sampling
python src/sample.py
python src/split.py

# 4. Claude API annotation (requires ANTHROPIC_API_KEY; resumes automatically if interrupted)
python src/label.py

# 5. Phase 2 analysis
python src/phase2_sample.py
python src/build_phase2_features.py
python src/analyze_phase2.py

# 6. Validation against human raters
python src/process_human_annotations.py
python src/compare_claude_vs_humans.py

# 7. Supporting analyses
python src/analyze_tier_d.py
python src/analyze_confidence.py
python src/audit_phase2_scores.py
python src/sensitivity_respect.py

# 8. Thesis figures
python scripts/create_figures.py
```

Full methodology and code-to-thesis mapping: **[docs/FOR_COMMITTEE.md](docs/FOR_COMMITTEE.md)**

Data sources and download instructions: **[docs/DATA.md](docs/DATA.md)**

---

## AI Tools Disclosure

This research uses two AI tools in distinct capacities. The Claude API (`claude-haiku-4-5-20251001`, Anthropic, 2025) was used as a computational instrument to score prosociality dimensions in approximately 100,000 Reddit, Stack Exchange, and Wikipedia records, and its outputs were validated against human annotations by three independent raters. Claude Code (Anthropic), an AI-assisted coding tool, was used to support development of analysis scripts. All code, analysis decisions, and conclusions are the author's own.

---

## Data and Ethical Considerations

This study analyzes publicly available user-generated content from Reddit, Stack Exchange, and Wikipedia under each platform's terms for academic research use. **No raw post or comment text is included in this repository.** All repository outputs are aggregate statistical results, summary tables, and figures. Usernames are not present in any analytical output.

See **[docs/DATA.md](docs/DATA.md)** for data source details, licensing, and download instructions.

---

## Citation

If you use this code or methodology, please cite the thesis:

```bibtex
@mastersthesis{linse2026,
  author      = {Linse, Tamara T.},
  title       = {Who Corrects and How: Large-Scale AI-Assisted Analysis of
                 Communication Style and Prosocial Behavioral Response Across
                 Peer, Expert, and Authority Contexts},
  school      = {University of Wyoming},
  department  = {Department of Electrical Engineering and Computer Science},
  year        = {2026},
  month       = {April},
  address     = {Laramie, Wyoming, USA}
}
```

A machine-readable `CITATION.cff` is also included for use with GitHub's "Cite this repository" feature and citation managers such as Zotero.

---

## License

MIT License — see [LICENSE](LICENSE). You are free to use, modify, and redistribute this code with attribution.
