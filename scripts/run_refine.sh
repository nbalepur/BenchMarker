#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

DATASET_NAME="ARC"
REWRITE_MODEL="openai/gpt-4.1-2025-04-14"

# flaw refinement
uv run python cli.py --refine.num_samples 5 --refine.rewrite_model ${REWRITE_MODEL} --steps refine --refine.difficulty.type none --metric_run_name ${DATASET_NAME}_large_scale_fixed --refine_run_name ${DATASET_NAME}_large_scale_fixed_rewrite  --refining_metrics shortcuts,writing_flaws,contamination

# add distractors
# uv run python cli.py --refine.num_samples 5 --refine.rewrite_model ${REWRITE_MODEL} --steps refine --refine.difficulty.type add_distractors --metric_run_name ${DATASET_NAME}_large_scale_fixed --refine_run_name ${DATASET_NAME}_large_scale_fixed_rewrite --refine.difficulty.num_distractors 2

# blooms taxonomy
# uv run python cli.py --refine.num_samples 5 --refine.rewrite_model ${REWRITE_MODEL} --steps refine --refine.difficulty.type blooms_taxonomy --metric_run_name ${DATASET_NAME}_large_scale_fixed --refine_run_name ${DATASET_NAME}_large_scale_fixed_rewrite --refine.difficulty.num_blooms_levels 2