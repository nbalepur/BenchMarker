"""
Memory optimization utilities for reducing memory usage when working with large pickle files.
Provides lightweight data extraction functions that extract only needed fields from EvalLog objects.
"""

import gc
from typing import List


class LightweightScore:
    """Lightweight wrapper for score values that only stores needed fields."""
    def __init__(self, value, answer=None, explanation=None, metadata=None):
        self.value = value
        self.answer = answer
        self.explanation = explanation
        self.metadata = metadata or {}


class LightweightSample:
    """Lightweight wrapper for sample data that only stores needed fields."""
    def __init__(self, sample_id, input, choices, target, scores):
        self.id = sample_id
        self.input = input
        self.choices = choices
        self.target = target
        self.scores = scores


class LightweightEvalLog:
    """Lightweight wrapper for EvalLog that only stores needed fields."""
    def __init__(self, samples):
        self.samples = samples


def extract_lightweight_report_card_data(report_card_logs):
    """
    Extract only the fields we need from report_card_logs to reduce memory usage.
    Returns lightweight EvalLog-like objects with just: sample_id, input, choices, target, scores.
    
    This function creates lightweight wrapper objects that mimic the structure of EvalLog
    but only store the fields actually used by downstream code, significantly reducing
    memory footprint for large datasets.
    
    Args:
        report_card_logs: List of EvalLog objects loaded from pickle files
        
    Returns:
        List of LightweightEvalLog objects containing only necessary fields
    """
    lightweight_logs = []
    for eval_log in report_card_logs:
        lightweight_samples = []
        for sample in eval_log.samples:
            # Extract scores - only the fields we actually use
            lightweight_scores = {}
            for score_name, score_value in sample.scores.items():
                lightweight_scores[score_name] = LightweightScore(
                    value=score_value.value,
                    answer=getattr(score_value, 'answer', None),
                    explanation=getattr(score_value, 'explanation', None),
                    metadata=getattr(score_value, 'metadata', {})
                )
            
            lightweight_sample = LightweightSample(
                sample_id=sample.id,
                input=sample.input,
                choices=sample.choices,
                target=sample.target,
                scores=lightweight_scores
            )
            lightweight_samples.append(lightweight_sample)
        lightweight_logs.append(LightweightEvalLog(lightweight_samples))
    return lightweight_logs


class LightweightRefinementSample:
    """Lightweight wrapper for refinement sample that only stores needed metadata."""
    def __init__(self, sample_id, metadata):
        self.id = sample_id
        self.metadata = metadata


class LightweightRefinementLog:
    """Lightweight wrapper for refinement log that only stores needed metadata."""
    def __init__(self, samples):
        self.samples = samples


def extract_lightweight_refinement_data(refinement_logs):
    """
    Extract only the metadata we need from refinement_logs to reduce memory usage.
    Returns lightweight objects with just: sample_id and metadata (question, choices_list, target, old_question, etc.).
    
    This function creates lightweight wrapper objects that only store the metadata fields
    actually used by downstream code, significantly reducing memory footprint.
    
    Args:
        refinement_logs: List of EvalLog objects from refinement step
        
    Returns:
        List of LightweightRefinementLog objects containing only necessary metadata
    """
    lightweight_logs = []
    for eval_log in refinement_logs:
        lightweight_samples = []
        for sample in eval_log.samples:
            # Extract only the metadata fields we need
            needed_metadata = {}
            if hasattr(sample, 'metadata') and sample.metadata:
                # Copy only the fields we actually use
                for key in ['question', 'choices_list', 'target', 'old_question', 
                           'old_choices_list', 'old_target', 'should_skip', 
                           'explanation', 'refinement_type']:
                    if key in sample.metadata:
                        needed_metadata[key] = sample.metadata[key]
            
            lightweight_sample = LightweightRefinementSample(
                sample_id=sample.id,
                metadata=needed_metadata
            )
            lightweight_samples.append(lightweight_sample)
        lightweight_logs.append(LightweightRefinementLog(lightweight_samples))
    return lightweight_logs


def load_and_extract_lightweight(cache, cache_type, id: str, run_name: str, key: str, cleanup_message: str = None):
    """
    Load data from cache, extract lightweight version, and free original data.
    
    This is a convenience function that combines loading, extraction, and cleanup
    in a single call to reduce memory usage immediately after loading large pickle files.
    
    Args:
        cache: Cache instance to load from
        cache_type: CacheType enum value
        id: Cache ID (e.g., 'eval_logs')
        run_name: Run name identifier
        key: Cache key (e.g., 'report_card_logs')
        cleanup_message: Optional message to print after cleanup
        
    Returns:
        Lightweight data extracted from the loaded pickle file, or None if not found
    """
    from utils.cache import Cache, CacheType
    
    # Load the full data
    full_data = cache.load(id, run_name, key)
    if full_data is None:
        return None
    
    # Extract lightweight version
    print(f"Extracting lightweight data from {key}...")
    lightweight_data = extract_lightweight_report_card_data(full_data)
    
    # Free original data
    del full_data
    gc.collect()
    
    if cleanup_message:
        print(cleanup_message)
    else:
        print(f"Cleanup complete! Extracted lightweight data from {key}.")
    
    return lightweight_data

