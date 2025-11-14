import random
from typing import List, Dict, Any
from data_utils.load_mcqa_task import load_mcqa_dataset, validate_and_summarize_dataset


def merge_and_shuffle_datasets(dataset_paths: List[str], shuffle_seed: int = 42) -> List[Dict[str, Any]]:
    """Merge multiple datasets and shuffle them evenly."""
    if not dataset_paths:
        raise ValueError("No dataset paths provided")
    
    all_datasets = []
    for path in dataset_paths:
        dataset = load_mcqa_dataset(path)
        all_datasets.append(list(dataset))
    
    # Round-robin shuffle for even distribution
    merged_samples = []
    dataset_indices = [0] * len(all_datasets)
    random.seed(shuffle_seed)
    
    while sum(dataset_indices) < sum(len(dataset) for dataset in all_datasets):
        available_datasets = [i for i, idx in enumerate(dataset_indices) if idx < len(all_datasets[i])]
        if not available_datasets:
            break
            
        selected_idx = random.choice(available_datasets)
        merged_samples.append(all_datasets[selected_idx][dataset_indices[selected_idx]])
        dataset_indices[selected_idx] += 1
    
    return merged_samples
