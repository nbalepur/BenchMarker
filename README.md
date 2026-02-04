# BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks

**This repository is a work in progress!**

This repository is the official implementation of the paper **BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks**.

**Authors:** Nishant Balepur<sup>1,2</sup>, Bhavya Rajasekaran<sup>1</sup>, Jane Oh<sup>1</sup>, Michael Xie<sup>1</sup>, Atrey Desai<sup>1</sup>, Vipul Gupta<sup>3</sup>, Steven Moore<sup>4</sup>, Eunsol Choi<sup>2</sup>, Rachel Rudinger<sup>1</sup>, Jordan Boyd-Graber<sup>1</sup>  
<sup>1</sup>University of Maryland &nbsp; <sup>2</sup>New York University &nbsp; <sup>3</sup>Scale AI &nbsp; <sup>4</sup>George Mason University

We provide our code for detecting flaws in MCQA benchmarks and logic to rewrite multiple-choice questions (results coming soon 👀)

## Abstract

Multiple-choice question answering (MCQA) is standard in NLP, but benchmarks lack rigorous quality control. We present BenchMarker, an education-inspired toolkit using LLM judges to flag three common MCQ flaws: (1) **contamination**—items appearing exactly online; (2) **shortcuts**—cues in the choices that enable guessing; and (3) **writing errors**—structural/grammatical issues based on a 19-rule education rubric. We validate BenchMarker with human annotations, then run the tool to audit 12 benchmarks, revealing: (1) flaws persist in MCQA benchmarks, especially automatically-made and crowdsourced data—we predict 47% of TruthfulQA appears online and 100% of HellaSwag violates multiple writing rules; (2) contaminated MCQs tend to inflate accuracy, while writing errors tend to lower it and change rankings beyond random; and (3) prior benchmark repairs address their targeted issues (e.g., lowering accuracy with LLM-written distractors) but inadvertently add new flaws (e.g., implausible distractors, many correct answers). Overall, flaws in MCQs degrade NLP evaluation, but education research offers a path forward. We release BenchMarker to bridge the fields and improve MCQA benchmark design.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Outputs and score interpretation](#outputs-and-score-interpretation)
- [Parameter Reference](#parameter-reference)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Installation

### Prerequisites

- Python >= 3.11

### Setup

```bash
git clone <repository-url>
cd BenchMarker

pip install -e .
```

### Environment Variables

Copy `.env.example` to `.env` and set API keys: OpenAI, Anthropic; for contamination (web search) set at least one of Google Custom Search, Brave, Tavily, Serper, or Exa. See `.env.example` for variable names.

## Quick Start

### 1. Run full pipeline (skills → metrics → refine)

```bash
python cli.py --steps skills,metrics,refine
```

### 2. Run single steps

```bash
python cli.py --steps skills
python cli.py --steps metrics
python cli.py --steps refine
```

### 3. Override config from CLI

```bash
python cli.py --steps metrics --dataset /path/to/data.csv --metrics.num_samples 20
```

Config defaults live in `config/` (base.yaml, metrics.yaml, refine.yaml, skills.yaml). See `scripts/PARAMS.md` for full override examples.

## Outputs and score interpretation

### Where results are saved

- **Cache** (`cache_dir`, default `cache_logs/`): Pickled eval logs and IRT data under `eval_logs/<run_name>/` and `irt_logs/<run_name>/`. Used to skip re-running when `cache_type` is `cache`.
- **IRT parameters** (`cache_dir/irt/<run_name>/`): `*_params.json` with per-item difficulty and discriminability, and model abilities.
- **Plots** (`plot_dir`, default `plots/`): IRT training plots (log-probability over samples) under `plot_dir/irt/<run_name>/` (e.g. `*_training_plot.pdf`).
- **Refined dataset** (`dataset_save_dir`, default `refined_dataset/`): After the refine step (when saving is enabled), under `<dataset_save_dir>/<refine_run_name>/`: `fixed.jsonl`, `fixed.csv`, `hf_dataset/`, and `annotations.xlsx` (old vs new MCQs and refinements for human review).

### What scores mean

| Metric | Score | Interpretation | d
|--------|--------|----------------|
| **Difficulty** (IRT) | `difficulty` | Higher = harder item (higher ability required to get 50% correct). |
| | `discriminability` | Higher = item better separates high- vs low-ability models. |
| **Shortcuts** | 0 | No shortcut detected (model did not get the right answer from choices-only, or the question the model inferred matched the original). |
| | 1 | Shortcut detected (model answered correctly from choices-only and the question did not match the original). |
| **Contamination** | 0 | Item appears to appear online. |
| | 1 | No such match found (no_match or partial_match only). |
| **Writing flaws** | 0–1 | Fraction of the 19 writing rules the item passes. 1 = all pass; 0 = all fail. Lower = more flaws. |

Refinement uses these scores (and config cutoffs) to filter or rewrite items; e.g. items with shortcut score 1 or contamination 0 can be filtered or rewritten.

In general, higher scores indicate less flaws and a better dataset!

## Parameter Reference

### Base parameters (no prefix)

| Parameter | Type | Description |
|-----------|------|-------------|
| --dataset | string | Path to dataset file (CSV) |
| --metrics | string | Comma-separated: writing_flaws, shortcuts, contamination, difficulty |
| --cache_dir | string | Cache directory |
| --dataset_save_dir | string | Where to save refined datasets |
| --plot_dir | string | Where to save plots |
| --cache_type | string | none \| cache \| overwrite |
| --skill_run_name | string | Run name for skills step |
| --metric_run_name | string | Run name for metrics step |
| --refine_run_name | string | Run name for refine step |

### Metrics (--metrics. prefix)

| Parameter | Description |
|-----------|-------------|
| --metrics.num_samples | Number of samples for evaluation (null = all) |
| --metrics.difficulty.models | Comma-separated model IDs for difficulty/IRT |
| --metrics.difficulty.irt_model.num_draws | IRT MCMC draws |
| --metrics.difficulty.irt_model.num_tune | IRT tune steps |
| --metrics.difficulty.irt_model.chains | IRT chains |
| --metrics.difficulty.irt_model.cores | IRT cores |
| --metrics.shortcuts.model | Model for shortcuts (LLM judge) |
| --metrics.shortcuts.num_attempts | Shortcuts attempts |
| --metrics.contamination.model | Model for contamination |
| --metrics.contamination.search_type | google \| perplexity \| brave |
| --metrics.contamination.use_llm | Use LLM in contamination (flag) |
| --metrics.writing_flaws.model | Model for writing-flaws judge |

### Refine (--refine. prefix)

| Parameter | Description |
|-----------|-------------|
| --refine.difficulty.type | efficiency \| saturation \| informative \| none |
| --refine.difficulty.min_discrimination | Min discrimination to keep |
| --refine.difficulty.efficiency.max_size | Proportion or count to keep (efficiency) |
| --refine.difficulty.informative.max_size | Proportion or count (informative) |
| --refine.difficulty.saturation.max_size | Proportion or count (hardest items) |
| --refine.shortcuts.type | filter \| rewrite \| none |
| --refine.shortcuts.model | Model for rewriting |
| --refine.contamination.type | filter \| rewrite \| none |
| --refine.contamination.model | Model for rewriting |
| --refine.writing_flaws.type | filter \| rewrite \| none |
| --refine.writing_flaws.model | Model for rewriting |

### Skills (--skills. prefix)

| Parameter | Description |
|-----------|-------------|
| --skills.num_samples | Number of samples for skills eval |
| --skills.skill_datasets | Comma-separated paths to skill datasets |
| --skills.difficulty.models | Models for skills difficulty |
| --skills.difficulty.irt_model.* | Same IRT options as metrics.difficulty |

## Project Structure

```
BenchMarker/
├── cli.py                     # Entry point: --steps skills,metrics,refine
├── config/                    # YAML config (base, metrics, refine, skills)
│   ├── base.yaml
│   ├── metrics.yaml
│   ├── refine.yaml
│   └── skills.yaml
├── endpoints/                 # Step runners
│   ├── run_metrics.py         # Difficulty, shortcuts, contamination, writing_flaws
│   ├── run_refine.py         # Filter/rewrite by metric
│   └── run_skills.py         # Skills evaluation
├── scorers/                   # Per-metric scoring
│   ├── difficulty_scorer.py  # IRT
│   ├── shortcut_scorer.py
│   ├── contamination_scorer.py
│   └── writing_flaws_scorer.py
├── prompts/                   # Prompt templates (shortcuts, contamination, writing flaws)
├── data_utils/                # Load, merge, refine, save datasets
├── model_utils/               # IRT, web search
├── judge_experiments/         # Judge model + prompts + validation data
├── utils/                     # Config merge, cache, enums
├── scripts/
│   ├── PARAMS.md              # Full parameter docs and examples
│   ├── run_metrics.sh
│   ├── run_refine.sh
│   └── run_skills.sh
├── local_datasets/            # Default/local MCQA data
├── pyproject.toml
└── README.md
```

## Citation

```bibtex
TBD
```

## Contact

For questions or issues, please open an issue on the repository or contact nbalepur@umd.edu
