import os
import json
import csv
import pandas as pd
from typing import Dict, List, Any
from inspect_ai.log import EvalLog
from inspect_ai.dataset import MemoryDataset, csv_dataset, json_dataset, hf_dataset, Sample
import string
from datasets import Dataset
import time

def load_mcqa_dataset_from_logs(logs: list[EvalLog]) -> Any:
    """Load MCQA dataset from logs."""
    samples = [Sample(
        input=record.input,
        choices=record.choices,
        target=record.target,
    ) for record in logs[0].samples]
    return MemoryDataset(samples)

def _save_dataset(new_mcqs: list[dict], dataset_dir: str, dataset_name: str) -> str:
    """Save dataset in JSONL, CSV, and HuggingFace formats."""
    base_path = os.path.join(dataset_dir, dataset_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = os.path.join(base_path, "fixed.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for mcq in new_mcqs:
            json.dump(mcq, f, ensure_ascii=False)
            f.write('\n')
    
    # Save as CSV
    csv_path = os.path.join(base_path, "fixed.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'choices', 'answer'])
        writer.writeheader()
        for mcq in new_mcqs:
            writer.writerow({
                'question': mcq['question'],
                'choices': str(mcq['choices']),
                'answer': mcq['answer']
            })
    
    # Save as HuggingFace dataset
    hf_path = os.path.join(base_path, "hf_dataset")
    hf_dataset = Dataset.from_list(new_mcqs)
    hf_dataset.save_to_disk(hf_path)
    
    print(f"Saved new dataset in jsonl: {jsonl_path}")
    print(f"Saved new dataset in csv: {csv_path}")
    print(f"Saved new dataset in HuggingFace format: {hf_path}")
    
    return jsonl_path

def load_mcqa_dataset(file_path: str) -> Any:
    """Load MCQA dataset from CSV, JSON, Excel, or HuggingFace."""
    file_path_lower = file_path.lower()
    
    if file_path_lower.endswith('.csv'):
        return csv_dataset(file_path, sample_fields=_record_to_sample)
    elif file_path_lower.endswith('.json') or file_path_lower.endswith('.jsonl'):
        return json_dataset(file_path, sample_fields=_record_to_sample)
    else:
        return hf_dataset(file_path, sample_fields=_record_to_sample)

def _parse_choices(choices):
    """Parse choices from various formats to list."""
    if isinstance(choices, list):
        return choices
    
    if isinstance(choices, str):
        try:
            return eval(choices)
        except:
            return [choice.strip() for choice in choices.split(',')]
    return choices


def _record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert record to Sample object."""
    return Sample(
        input=record["question"],
        choices=_parse_choices(record["choices"]),
        target=record["answer"],
    )


def validate_and_summarize_dataset(file_path: str) -> Dict[str, Any]:
    """
    Validate MCQA dataset and return summary.
    
    Args:
        file_path: Path to dataset file
        verbose: Whether to print validation report
        
    Returns:
        Dictionary with validation results and summary
    """
    try:
        raw_data = _load_raw_data(file_path)
    except Exception as e:
        result = {"valid": False, "errors": [f"Failed to load data: {e}"], "summary": {}}
        _print_report(result)
        return result
    
    result = {"valid": True, "errors": [], "warnings": [], "summary": {}}
    
    _validate_structure(raw_data, result)
    _generate_summary(raw_data, result)
    _print_report(result)
    
    return result


def _load_raw_data(file_path: str) -> List[Dict[str, Any]]:
    """Load raw data for validation."""
    file_path_lower = file_path.lower()
    
    if file_path_lower.endswith('.csv'):
        return pd.read_csv(file_path).to_dict('records')
    elif file_path_lower.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path_lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path).to_dict('records')
    else:
        raise ValueError(f"Cannot validate HuggingFace dataset '{file_path}' directly")


def _validate_structure(data: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    """Validate dataset structure and types."""
    if not isinstance(data, list) or len(data) == 0:
        result["valid"] = False
        result["errors"].append("Dataset must be a non-empty list of records")
        return
    
    required_fields = ["question", "choices", "answer"]
    
    for i, record in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in record:
                result["valid"] = False
                result["errors"].append(f"Record {i}: Missing '{field}'")
        
        if not result["valid"]:
            continue
            
        # Validate types
        if not isinstance(record["question"], str):
            result["valid"] = False
            result["errors"].append(f"Record {i}: 'question' must be string")
        
        # Validate choices - parse first, then validate
        try:
            parsed_choices = _parse_choices(record["choices"])
            if not isinstance(parsed_choices, list):
                result["valid"] = False
                result["errors"].append(f"Record {i}: 'choices' must parse to list")
            elif len(parsed_choices) == 0:
                result["warnings"].append(f"Record {i}: Empty choices")
            else:
                for j, choice in enumerate(parsed_choices):
                    if not isinstance(choice, str):
                        result["valid"] = False
                        result["errors"].append(f"Record {i}: Choice {j} must be string")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Record {i}: Failed to parse choices: {e}")
        
        # Validate answer
        if not isinstance(record["answer"], str):
            result["valid"] = False
            result["errors"].append(f"Record {i}: 'answer' must be string")
        else:
            answer = record["answer"].upper()
            num_choices = len(record["choices"]) if isinstance(record["choices"], list) else 0
            
            if num_choices > 0:
                valid_letters = string.ascii_uppercase[:num_choices]
                if answer not in valid_letters:
                    result["valid"] = False
                    result["errors"].append(
                        f"Record {i}: Answer '{answer}' invalid. Must be {valid_letters[0]}-{valid_letters[-1]}"
                    )


def _generate_summary(data: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
    """Generate dataset summary statistics."""
    if not data:
        return
    
    summary = {"total_questions": len(data)}
    
    # Question lengths
    q_lengths = [len(r["question"].split()) for r in data if isinstance(r.get("question"), str)]
    if q_lengths:
        summary["question_length"] = {
            "min": min(q_lengths), "max": max(q_lengths),
            "mean": sum(q_lengths) / len(q_lengths),
            "median": sorted(q_lengths)[len(q_lengths) // 2]
        }
    
    # Choice counts
    c_counts = [len(_parse_choices(r["choices"])) for r in data if isinstance(r.get("choices"), (str, list))]
    if c_counts:
        summary["choice_counts"] = {
            "min": min(c_counts), "max": max(c_counts),
            "mean": sum(c_counts) / len(c_counts),
            "median": sorted(c_counts)[len(c_counts) // 2]
        }
        
        # Check for unusual choice counts
        unique_counts = set(c_counts)
        standard_counts = {3, 4, 5}
        unusual_counts = unique_counts - standard_counts
        
        if unusual_counts:
            result["warnings"].append(
                f"Non-standard choice counts detected: {sorted(unusual_counts)}. "
                f"Standard MCQA typically uses 3, 4, or 5 choices per question."
            )
    
    # Answer distribution
    answer_counts = {}
    for record in data:
        if isinstance(record.get("answer"), str):
            answer = record["answer"].upper()
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
    summary["answer_distribution"] = answer_counts
    
    # Check for non-uniform answer distribution using chi-square test
    if len(answer_counts) > 1:
        from scipy.stats import chisquare
        
        total_answers = sum(answer_counts.values())
        expected_per_answer = total_answers / len(answer_counts)
        
        # Prepare observed and expected frequencies
        observed = list(answer_counts.values())
        expected = [expected_per_answer] * len(answer_counts)
        
        # Perform chi-square goodness-of-fit test
        chi2_stat, p_value = chisquare(observed, expected)
        
        if p_value < 0.05:  # Significant deviation from uniform distribution
            result["warnings"].append(
                f"Non-uniform answer distribution detected (χ²={chi2_stat:.2f}, p={p_value:.3f} < 0.05). "
                f"Expected uniform distribution (~{100/len(answer_counts):.1f}% each), "
                f"but found: {', '.join([f'{k}: {v/total_answers*100:.1f}%' for k, v in sorted(answer_counts.items())])}"
            )
    
    result["summary"] = summary


def _print_report(result: Dict[str, Any]) -> None:
    """Print clean validation report."""
    status = "VALID ✅" if result["valid"] else "INVALID ❌"
    print(f"Dataset Status: {status}")
    
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  • {error}")
    
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  • {warning}")
    
    summary = result["summary"]
    if summary:
        print(f"Summary: {summary['total_questions']} questions")
        
        if "question_length" in summary:
            ql = summary["question_length"]
            print(f"Question length: {ql['min']}-{ql['max']} words (avg: {ql['mean']:.1f})")
        
        if "choice_counts" in summary:
            cc = summary["choice_counts"]
            print(f"Choices: {cc['min']}-{cc['max']} per question (avg: {cc['mean']:.1f})")
        
        if "answer_distribution" in summary:
            dist = summary["answer_distribution"]
            total = summary["total_questions"]
            print("Answer distribution:", end=" ")
            print(", ".join([f"{k}: {v/total*100:.1f}%" for k, v in sorted(dist.items())]))

    if status == "❌ INVALID":
        raise ValueError("Dataset is invalid")
