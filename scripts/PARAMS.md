# MCQA Benchmark - Complete Parameter Reference

This document lists all available parameters that can be overridden from the command line.

## Base Parameters (No Prefix)

These parameters are in `base.yaml` and don't need a prefix:

```bash
--dataset DATASET                    # Path to dataset file
--metrics METRICS                    # Comma-separated list of metrics
--save_dir SAVE_DIR                  # Directory to save results
--plot_dir PLOT_DIR                  # Directory to save plots  
--cache_type CACHE_TYPE              # Cache type (none, cache, overwrite)
--skill_run_name SKILL_RUN_NAME      # Name for skills run
--metric_run_name METRIC_RUN_NAME    # Name for metrics run
--refine_run_name REFINE_RUN_NAME    # Name for refine run
```

## Metrics Parameters (--metrics. prefix)

These parameters are in `metrics.yaml` and use the `--metrics.` prefix:

### General Metrics
```bash
--metrics.num_samples METRICS.NUM_SAMPLES    # Number of samples for evaluation
```

### Difficulty Metric
```bash
--metrics.difficulty.models METRICS.DIFFICULTY.MODELS                    # Models for difficulty evaluation
--metrics.difficulty.irt_model.num_draws METRICS.DIFFICULTY.IRT_MODEL.NUM_DRAWS    # IRT draws
--metrics.difficulty.irt_model.num_tune METRICS.DIFFICULTY.IRT_MODEL.NUM_TUNE      # IRT tune
--metrics.difficulty.irt_model.chains METRICS.DIFFICULTY.IRT_MODEL.CHAINS          # IRT chains
--metrics.difficulty.irt_model.cores METRICS.DIFFICULTY.IRT_MODEL.CORES            # IRT cores
```

### Shortcuts Metric
```bash
--metrics.shortcuts.model METRICS.SHORTCUTS.MODEL           # Model for shortcuts evaluation
--metrics.shortcuts.num_attempts METRICS.SHORTCUTS.NUM_ATTEMPTS  # Number of attempts
```

### Contamination Metric
```bash
--metrics.contamination.model METRICS.CONTAMINATION.MODEL                    # Model for contamination evaluation
--metrics.contamination.search_type METRICS.CONTAMINATION.SEARCH_TYPE        # Search type (google, perplexity, brave)
--metrics.contamination.use_llm                                              # Use LLM (flag)
```

### Writing Flaws Metric
```bash
--metrics.writing_flaws.model METRICS.WRITING_FLAWS.MODEL    # Model for writing flaws evaluation
```

## Refine Parameters (--refine. prefix)

These parameters are in `refine.yaml` and use the `--refine.` prefix:

### General Refine
```bash
--refine.new_dataset_suffix REFINE.NEW_DATASET_SUFFIX    # Suffix for refined dataset
```

### Difficulty Refinement
```bash
--refine.difficulty.type REFINE.DIFFICULTY.TYPE          # Refinement type (efficiency, difficulty_split, informative, none)
--refine.difficulty.min_discrimination REFINE.DIFFICULTY.MIN_DISCRIMINATION  # Minimum discrimination to keep (float)
--refine.difficulty.efficiency.max_size REFINE.DIFFICULTY.EFFICIENCY.MAX_SIZE      # Keep proportion (0-1) or count of most "efficient" items
--refine.difficulty.informative.max_size REFINE.DIFFICULTY.INFORMATIVE.MAX_SIZE    # Keep proportion (0-1) or count of most discriminative items
--refine.difficulty.difficulty_split.max_size REFINE.DIFFICULTY.DIFFICULTY_SPLIT.MAX_SIZE  # Keep proportion (0-1) or count of hardest items
```

### Shortcuts Refinement
```bash
--refine.shortcuts.type REFINE.SHORTCUTS.TYPE            # Refinement type (filter, rewrite, none)
--refine.shortcuts.model REFINE.SHORTCUTS.MODEL          # Model for rewriting
```

### Contamination Refinement
```bash
--refine.contamination.type REFINE.CONTAMINATION.TYPE    # Refinement type (filter, rewrite, none)
--refine.contamination.model REFINE.CONTAMINATION.MODEL  # Model for rewriting
```

### Writing Flaws Refinement
```bash
--refine.writing_flaws.type REFINE.WRITING_FLAWS.TYPE    # Refinement type (filter, rewrite, none)
--refine.writing_flaws.model REFINE.WRITING_FLAWS.MODEL  # Model for rewriting
```

## Skills Parameters (--skills. prefix)

These parameters are in `skills.yaml` and use the `--skills.` prefix:

### General Skills
```bash
--skills.num_samples SKILLS.NUM_SAMPLES                   # Number of samples for skills evaluation
--skills.skill_datasets SKILLS.SKILL_DATASETS             # Comma-separated list of skill datasets
```

### Skills Difficulty
```bash
--skills.difficulty.models SKILLS.DIFFICULTY.MODELS                    # Models for skills difficulty evaluation
--skills.difficulty.irt_model.num_draws SKILLS.DIFFICULTY.IRT_MODEL.NUM_DRAWS    # IRT draws
--skills.difficulty.irt_model.num_tune SKILLS.DIFFICULTY.IRT_MODEL.NUM_TUNE      # IRT tune
--skills.difficulty.irt_model.chains SKILLS.DIFFICULTY.IRT_MODEL.CHAINS          # IRT chains
--skills.difficulty.irt_model.cores SKILLS.DIFFICULTY.IRT_MODEL.CORES            # IRT cores
```

## Usage Examples

### Minimal Override (Only Change What You Need)
```bash
# Only change sample size - everything else uses YAML defaults
python cli.py --steps metrics --metrics.num_samples 20
```

### Common Overrides
```bash
# Change dataset and models
python cli.py --steps metrics \
    --dataset /path/to/your/dataset.csv \
    --metrics.difficulty.models openai/gpt-4o,openai/gpt-4o-mini

# Change IRT parameters
python cli.py --steps metrics \
    --metrics.difficulty.irt_model.num_draws 500 \
    --metrics.difficulty.irt_model.chains 4

# Enable refinement
python cli.py --steps refine \
    --refine.shortcuts.type rewrite \
    --refine.shortcuts.model openai/gpt-4o
```

### Pipeline with Multiple Overrides
```bash
python cli.py --steps skills,metrics,refine \
    --dataset /path/to/dataset.csv \
    --metrics.difficulty.models openai/gpt-4o \
    --metrics.num_samples 50 \
    --refine.shortcuts.type rewrite \
    --refine.contamination.type filter
```

## Parameter Types

### String Parameters
- `--dataset`, `--metrics`, `--save_dir`, etc.
- Use quotes if values contain spaces: `--metrics "difficulty, shortcuts"`

### Boolean Parameters (Flags)
- `--metrics.difficulty.models`, `--metrics.contamination.use_llm`
- Just include the flag to set to `true`, omit to set to `false`

### List Parameters
- `--metrics`, `--metrics.difficulty.models`, `--skills.skill_datasets`
- Use comma-separated values: `--metrics "difficulty,shortcuts,contamination"`

### Numeric Parameters
- `--metrics.num_samples`, `--metrics.difficulty.irt_model.num_draws`
- Pass as strings: `--metrics.num_samples "20"`

## Default Values

All parameters have default values defined in the YAML config files:
- `config/base.yaml` - Base parameters
- `config/metrics.yaml` - Metrics parameters  
- `config/refine.yaml` - Refine parameters
- `config/skills.yaml` - Skills parameters

You only need to override the parameters you want to change!
