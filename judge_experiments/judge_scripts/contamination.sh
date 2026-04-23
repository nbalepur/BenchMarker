#!/bin/bash

# Two-step contamination judge: (1) precompute web search per engine, (2) run LLM judge per model × engine.
# Uses the same RUN_NAME, res_dir, search_engine, and max_results in both steps so cached search loads correctly.

DATASET_NAME="judge_experiments/validation_data/contamination_judge.jsonl"
MAX_RESULTS=10 # number of web search results
RUN_NAME="default"
RES_DIR="judge_experiments/results"

# Search engines
SEARCH_ENGINES=("google")
# LLM judges (liteLLM-style model ids for step 2)
MODELS=("gemini/gemini-2.5-pro")

for SEARCH_ENGINE in "${SEARCH_ENGINES[@]}"; do
    echo "Step 1 — web search for engine: $SEARCH_ENGINE"
    echo "========================================"

    uv run python3 -m judge_experiments.model.endpoints.run \
        --dataset_name "$DATASET_NAME" \
        --prompt_type "web_search" \
        --model_name "$SEARCH_ENGINE" \
        --generation_strategy "web_search" \
        --search_engine "$SEARCH_ENGINE" \
        --max_results "$MAX_RESULTS" \
        --res_dir "$RES_DIR" \
        --run_name "$RUN_NAME"

    echo "Step 2 — contamination judge for engine: $SEARCH_ENGINE"
    echo "========================================"

    for MODEL in "${MODELS[@]}"; do
        echo "  Judge model: $MODEL"

        uv run python3 -m judge_experiments.model.endpoints.run \
            --dataset_name "$DATASET_NAME" \
            --prompt_type "contamination" \
            --model_name "$MODEL" \
            --generation_strategy "prompt" \
            --search_engine "$SEARCH_ENGINE" \
            --max_results "$MAX_RESULTS" \
            --res_dir "$RES_DIR" \
            --run_name "$RUN_NAME"

        echo "  Completed judge for: $MODEL"
    done

    echo "Completed search engine: $SEARCH_ENGINE"
    echo "----------------------------------------"
done

echo "All contamination experiments completed!"
