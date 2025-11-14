#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python cli.py \
--steps skills,metrics \
--metrics.num_samples 20 \
--skills.num_samples 5