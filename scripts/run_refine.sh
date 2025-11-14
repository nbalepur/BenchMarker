#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python cli.py --steps metrics,refine --refining_metrics writing_flaws,contamination,difficulty,shortcuts --scoring_metrics writing_flaws,contamination,difficulty,shortcuts --metric_run_name testing_irt --refine_run_name testing_irt --metrics.num_samples 30