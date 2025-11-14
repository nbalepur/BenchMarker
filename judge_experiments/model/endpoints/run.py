"""
Model testing endpoint for evaluating trained models or running inference.
Supports resuming from existing results and saving outputs in JSONL format.
"""

import json
from typing import List, Dict, Any
from tqdm import tqdm

from judge_experiments.model.generation_loader import GeneratorFactory
from judge_experiments.prompts.prompt_builder import PromptBuilder
from judge_experiments.model.training_args import setup_training_args
from judge_experiments.model.utils import (
    create_model_directories, 
    create_results_path,
    ensure_directory_exists,
    load_jsonl_file,
    write_jsonl_line
)


def setup_directories_and_paths(args) -> tuple[str, str]:
    """Setup directory structure and return model and results paths."""
    model_dirs = create_model_directories(args)
    model_dir = model_dirs['final']
    
    results_path = create_results_path(args)
    ensure_directory_exists(results_path)
    
    return model_dir, results_path


def write_results_to_file(results_path: str, existing_data: List[Dict], 
                         test_prompts: List[str], generator, start_idx: int = 0, args=None):
    """Write results to JSONL file, resuming from start_idx."""
    with open(results_path, 'w') as f:
        # Write existing data first
        for data in existing_data:
            write_jsonl_line(f, data)
        
        # Generate and write new results
        prompt_indices = list(range(start_idx, len(test_prompts)))
        total_cost = 0.0
        
        for prompt_idx in tqdm(prompt_indices, desc="Generating responses"):
            prompt = test_prompts[prompt_idx]

            generation_result = generator.generate(prompt)
            
            if isinstance(generation_result, dict):
                response = generation_result['response']
                cost = generation_result.get('cost', 0.0)
                if cost is None:
                    cost = 0.0
            else:
                response = generation_result
                cost = 0.0
            total_cost += cost
            
            # Handle web search results specially
            if args and args.generation_strategy.value == 'web_search':
                # For web search, we need to save structured data for later use
                result = {
                    'prompt': prompt,
                    'response': response,
                    'prompt_idx': prompt_idx,
                    'cost': cost,
                    'search_results': generation_result.get('search_results', []),
                    'num_results': generation_result.get('num_results', 0),
                    'search_type': generation_result.get('search_type', ''),
                    'success': generation_result.get('success', True),
                    'error': generation_result.get('error', None)
                }
            else:
                # Standard result format for other generators
                result = {
                    'prompt': prompt,
                    'response': response,
                    "reasoning_trace": generation_result.get('reasoning_trace', ''),
                    'prompt_idx': prompt_idx,
                    'cost': cost
                }
            
            write_jsonl_line(f, result)
        
        return total_cost


def main(args):
    """Main testing function."""
    # Setup paths and directories
    model_dir, results_path = setup_directories_and_paths(args)
    
    # Load prompts
    prompt_builder = PromptBuilder(args)
    test_prompts = prompt_builder.get_prompts(prompt_type=args.prompt_type)
    
    # Initialize generator
    generator_factory = GeneratorFactory(args)
    generator = generator_factory.get_generator(generation_strategy=args.generation_strategy)
    generator.load(model_dir)
    
    # Load existing results for resumption
    existing_data = load_jsonl_file(results_path)
    start_idx = len(existing_data)
    
    print(f"Found {len(existing_data)} existing results, starting from index {start_idx}")
    print(f"Total prompts to process: {len(test_prompts)}")
    print(f"Results will be saved to: {results_path}")
    
    # Generate and save results
    total_cost = write_results_to_file(results_path, existing_data, test_prompts, generator, start_idx, args)
    
    print(f"Testing completed. Results saved to: {results_path}")
    print(f"Total cost for this run: ${total_cost:.4f}")
    
    # Log cost information to a separate cost log file
    cost_log_path = results_path.replace('.jsonl', '_cost.log')
    with open(cost_log_path, 'a') as cost_file:
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        cost_file.write(f"{timestamp}: Processed {len(test_prompts) - start_idx} prompts, Total cost: ${total_cost:.4f}\n")
    
    print(f"Cost information logged to: {cost_log_path}")


def setup():
    """Setup training arguments."""
    return setup_training_args()


if __name__ == '__main__':
    args = setup()
    main(args)