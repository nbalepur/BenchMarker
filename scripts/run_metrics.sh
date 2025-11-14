#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python cli.py --steps metrics --metrics.num_samples 30 --scoring_metrics shortcuts --metric_run_name debug_shortcuts