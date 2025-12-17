# Benchmarking Multiple-Choice Benchmarks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Official implementation of "BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks"**

</div>

## Overview

This repository is the official implementation of **BenchMarker**, an education-inspired toolkit for identifying and addressing flaws in multiple-choice question answering benchmarks. The framework provides comprehensive evaluation metrics including difficulty estimation, shortcut detection, contamination analysis, and writing quality assessment.

## 📁 Repository Structure

### 🔧 Core Components

#### `cli.py`
Main command-line interface for executing benchmark operations. Supports flexible configuration through command-line arguments with YAML-based defaults. Enables running individual steps (skills, metrics, refine) or complete pipelines.

#### `config/`
Configuration files defining default parameters for all benchmark operations:
- `base.yaml`: Core settings including dataset paths, metric selection, and output directories
- `metrics.yaml`: Configuration for metric evaluation (difficulty, shortcuts, contamination, writing flaws)
- `refine.yaml`: Settings for dataset refinement operations
- `skills.yaml`: Configuration for skills-based evaluation

#### `endpoints/`
Main execution endpoints for benchmark operations:
- `run_metrics.py`: Executes metric evaluation on datasets
- `run_refine.py`: Performs dataset refinement based on metric scores
- `run_skills.py`: Runs skills-based evaluation using anchor datasets

#### `scorers/`
Implementation of metric scorers:
- `difficulty_scorer.py`: Item Response Theory (IRT)-based difficulty estimation
- `shortcut_scorer.py`: Detection of answer shortcuts in questions
- `contamination_scorer.py`: Identification of training data contamination
- `writing_flaws_scorer.py`: Assessment of writing quality issues

#### `data_utils/`
Dataset loading and processing utilities:
- `load_mcqa_task.py`: Loads MCQA datasets in standardized format
- `merge_datasets.py`: Merges multiple datasets
- `refine_dataset.py`: Applies refinement operations to datasets
- `return_dataset.py`: Returns datasets in evaluation format
- `save_annotations.py`: Saves annotation results

#### `model_utils/`
Model-related utilities:
- `irt.py`: Item Response Theory model implementation using PyMC
- `web_search.py`: Web search integration for contamination detection

#### `prompts/`
Prompt templates for various evaluation tasks:
- `contamination_prompt.py`: Prompts for contamination detection
- `shortcut_prompts.py`: Prompts for shortcut identification
- `writing_flaw_prompts.py`: Prompts for writing quality assessment
- `run_mcqa_prompts.py`: Prompts for MCQA task execution
- `rewrite_prompts.py`: Prompts for question rewriting

#### `utils/`
Core utility functions:
- `argparse_config.py`: Configuration parsing and merging from YAML and CLI arguments
- `cache.py`: Caching utilities for evaluation results
- `enums.py`: Enumeration definitions for metrics and other constants
- `setup.py`: Setup and initialization utilities

### 🧪 Experimental Components

#### `judge_experiments/`
Experimental framework for judge model evaluation:
- `model/`: Judge model implementations and utilities
- `prompts/`: Prompt building and template loading for judge experiments
- `analysis/`: Analysis scripts for judge experiment results
- `validation_data/`: Validation datasets for judge models

#### `local_datasets/`
Local storage for MCQA datasets. Supports multiple dataset formats and includes loaders for various benchmark datasets used in the paper (ARC, MMLU, HellaSwag, etc.).

### 📜 Scripts

#### `scripts/`
Shell scripts for batch execution:
- `run_metrics.sh`: Batch execution of metrics evaluation
- `run_refine.sh`: Batch execution of refinement operations
- `run_skills.sh`: Batch execution of skills evaluation
- `PARAMS.md`: Complete parameter reference documentation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mcqa-bench

# Install dependencies (using uv or pip)
uv sync
# or
pip install -e .
```

### Basic Usage

```bash
# Run complete pipeline (skills → metrics → refine)
python cli.py --steps skills,metrics,refine

# Run individual steps
python cli.py --steps metrics
python cli.py --steps refine

# Override configuration parameters
python cli.py --steps metrics \
    --dataset /path/to/dataset.csv \
    --metrics.difficulty.models openai/gpt-5-2025-08-07 \
    --metrics.num_samples 100
```

### Configuration

Most parameters are defined in YAML configuration files (`config/`) and can be overridden via command-line arguments. See `scripts/PARAMS.md` for complete parameter documentation.

## 📦 Dependencies

The project requires **Python 3.11** and includes dependencies for:

| Category | Technologies |
|----------|-------------|
| **LLM Integration** | OpenAI, Anthropic, Google, Cohere |
| **Statistical Modeling** | PyMC, NumPyro, scipy, statsmodels |
| **Data Processing** | pandas, datasets |
| **Machine Learning** | transformers, torch, accelerate |
| **Web Search APIs** | Tavily, Exa, Perplexity |

See `pyproject.toml` for complete dependency specifications.

## 📊 Output Structure

Results are organized by run names specified in configuration:

| Output Type | Location |
|------------|----------|
| Evaluation logs and scores | `cache_logs/` |
| Refined datasets | `refined_dataset/` |
| Plots and visualizations | `plots/` |

## 📚 Key Features

### Metrics

- **🎯 Difficulty Estimation**: Item Response Theory (IRT) with Bayesian inference
- **🔍 Shortcut Detection**: Identifies answerable questions without full reasoning
- **⚠️ Contamination Analysis**: Detects training data contamination via web search
- **✍️ Writing Quality Assessment**: Evaluates questions for clarity and answerability

### Dataset Refinement

- **Filtering**: Remove questions based on metric thresholds
- **Rewriting**: Automatically rewrite questions to address identified issues
- **Difficulty-based Selection**: Select questions using IRT parameters (efficiency, discrimination, difficulty splits)

## 📄 Citation

If you use this codebase in your research, please cite:

```bibtex
@article{benchmarker2024,
  title={BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks},
  author={anonymous},
  journal={anonymous},
  year={2026}
}
```

