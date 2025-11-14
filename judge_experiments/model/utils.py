"""
Shared utility functions used across model endpoints.
Contains common functionality for path construction, run naming, and file operations.
"""

import os
import json
from typing import List, Dict, Any, Tuple


def create_run_name(args) -> str:
    """
    Create a standardized run name from arguments.
    
    Args:
        args: Arguments object with generation_strategy, model_name, 
              prompt_type, and run_name attributes
              
    Returns:
        Formatted run name string
    """
    return f"{args.generation_strategy.value.upper()}_{args.model_name}_{args.prompt_type.value}_{args.run_name}"


def create_model_directories(args, run_name: str = None) -> Dict[str, str]:
    """
    Create standardized model directory paths.
    
    Args:
        args: Arguments object with generation_strategy, model_name, prompt_type, run_name
        run_name: Optional run name, will create if not provided
        
    Returns:
        Dictionary with model directory paths
    """
    if run_name is None:
        run_name = create_run_name(args)
    
    base_dir = f'./models/{run_name}'
    return {
        'base': base_dir,
        'adapter': f'{base_dir}_adapter',
        'final': f'{base_dir}_final'
    }


def create_results_path(args) -> str:
    """
    Create standardized results file path.
    
    Args:
        args: Arguments object with res_dir, model_name, run_name,
              generation_strategy, prompt_type, and optionally search_engine, max_results, try_scraping
        
    Returns:
        Path to results JSONL file
    """
    filename = args.prompt_type.value + '.jsonl'
    
    # For contamination experiments, include search engine details in filename
    if hasattr(args, 'prompt_type') and args.prompt_type.value == 'contamination':
        search_engine = getattr(args, 'search_engine', None)
        max_results = getattr(args, 'max_results', 5)
        try_scraping = getattr(args, 'try_scraping', False)
        
        if search_engine:
            filename = f"{args.prompt_type.value}_{search_engine.value}_{max_results}_{try_scraping}.jsonl"
    
    return (f'{args.res_dir}/{args.run_name}/{args.generation_strategy.value}/{args.model_name}/{filename}')


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the directory for a file path exists.
    
    Args:
        file_path: Path to file (directory will be created for its parent)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file, skipping corrupted lines.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries from valid JSON lines
    """
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping corrupted line {line_num} in {file_path}: {e}")
                    continue
    return data


def write_jsonl_line(file_handle, data: Dict[str, Any]) -> None:
    """
    Write a single line to a JSONL file.
    
    Args:
        file_handle: Open file handle
        data: Dictionary to write as JSON
    """
    json.dump(data, file_handle)
    file_handle.write('\n')
    file_handle.flush()