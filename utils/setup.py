import yaml
import os
from typing import Any, Dict, Optional, Union
import argparse
from .enums import Metrics


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary (typically from YAML config)
        override: Override dictionary (typically from user parameters)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def _validate_metrics_in_config(config: Dict[str, Any]) -> None:
    """
    Validate that metrics lists in the config contain only valid enum values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If any metrics list contains invalid metric names
    """
    metrics_fields = ["scoring_metrics", "refining_metrics"]
    
    for field in metrics_fields:
        if field in config:
            try:
                Metrics.validate_metrics_list(config[field])
            except ValueError as e:
                raise ValueError(f"Configuration error in '{field}': {e}")


def load_config(config_path: str | None = None, 
                config_name: str = "config.yaml",
                user_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load config.yaml and validate required fields. Optionally merge with user overrides.
    
    Args:
        config_path: Path to config directory
        config_name: Name of config file
        user_overrides: Optional dictionary of user overrides to merge with config
        
    Returns:
        Loaded and optionally merged config dictionary
    """

    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_dir = os.path.join(repo_root, config_path, config_name)
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config file not found: {config_dir}")

    with open(config_dir, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("config.yaml root must be a mapping/object")
    
    # Apply user overrides if provided
    if user_overrides:
        data = _deep_merge_dict(data, user_overrides)

    # Validate metrics lists if they exist
    _validate_metrics_in_config(data)

    return data


def show_all_parameters():
    """
    Show all available parameters for MCQA Benchmark CLI.
    This function displays all configurable parameters that can be overridden.
    """
    from .argparse_config import create_config_parser
    
    print("MCQA Benchmark - All Available Parameters")
    print("=" * 60)
    print("Most parameters are defined in config files (base.yaml, metrics.yaml, etc.)")
    print("but can be overridden from the command line using the arguments below:")
    print()
    
    # Load configs
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_dir = os.path.join(repo_root, 'config')
    
    configs = {}
    for config_name in ['base', 'metrics', 'refine', 'skills']:
        config_path = os.path.join(config_dir, f'{config_name}.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs[config_name] = yaml.safe_load(f) or {}
    
    # Create parser and add dynamic arguments
    parser = create_config_parser()
    parser.add_dynamic_arguments(configs)
    
    # Show help
    parser.parser.print_help()
    
    print("\n" + "=" * 60)
    print("Usage Examples:")
    print("  # Override specific parameters")
    print("  python endpoints/cli.py --steps metrics --metrics_num_samples 20")
    print("  python endpoints/cli.py --steps metrics --metrics_difficulty.models openai/gpt-4o,openai/gpt-4o-mini")
    print("  python endpoints/cli.py --steps refine --refine_shortcuts.type rewrite")
    print("  python endpoints/cli.py --steps skills --skills_difficulty.irt_model.num_draws 500")
    print()
    print("  # All parameters are optional - only override what you want to change!")
    print("  # Everything else uses the defaults from your YAML config files.")