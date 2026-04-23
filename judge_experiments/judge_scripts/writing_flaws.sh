#!/bin/bash

# Example script for validating the writing-flaw judges (LLM only; no web search).
# For contamination (web search precompute + LLM judge), use judge_scripts/search.sh
# then judge_experiments/judge_scripts/contamination.sh.

# Set common parameters
DATASET_NAME="judge_experiments/validation_data/writing_flaws_judge.jsonl"
PROMPT_TYPE="writing_flaws"
RUN_NAME="default"

MODELS=("gemini/gemini-2.5-pro") # define your list of models here
GENERATION_STRATEGY="prompt_hf" # 'prompt_hf' for huggingface endpoints, 'prompt' for litellm endpoints

for MODEL in "${MODELS[@]}"; do
    echo "Running experiments for model: $MODEL"
    echo "========================================"

    uv run python3 -m judge_experiments.model.endpoints.run \
        --dataset_name "$DATASET_NAME" \
        --prompt_type "$PROMPT_TYPE" \
        --model_name "$MODEL" \
        --generation_strategy "$GENERATION_STRATEGY" \
        --res_dir "judge_experiments/results" \
        --run_name "$RUN_NAME"

    echo "Completed experiments for model: $MODEL"
    echo "========================================"
done

echo "All writing flaws experiments completed!"
