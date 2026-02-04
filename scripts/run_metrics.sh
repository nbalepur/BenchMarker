#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python cli.py --steps metrics --metrics.num_samples 30 --metric_run_name my_run