#!/bin/bash

# Example script showing how to validate the shortcuts judge

# Set common parameters
DATASET_NAME="judge_experiments/validation_data/shortcuts_judge.jsonl"
PROMPT_TYPE="shortcuts"
RUN_NAME="default"

MODELS=("gemini/gemini-2.5-pro") # define your list of models here
GENERATION_STRATEGY="prompt_hf" # 'prompt_hf' for huggingface endpoints, 'prompt' for litellm endpoints

# Loop through each model and search engine
for MODEL in "${MODELS[@]}"; do
    echo "Running experiments for model: $MODEL"
    echo "========================================"
    
    uv run python3 -m judge_experiments.model.endpoints.run \
        --dataset_name "$DATASET_NAME" \
        --prompt_type "$PROMPT_TYPE" \
        --model_name "$MODEL" \
        --generation_strategy "$GENERATION_STRATEGY" \
        --res_dir "judge_experiments/results" \
        --cache_dir "/fs/clip-scratch/nbalepur/cache" \
        --run_name "$RUN_NAME"

    
    echo "Completed all experiments for model: $MODEL"
    echo "========================================"
done

echo "All shortcuts experiments completed!"
