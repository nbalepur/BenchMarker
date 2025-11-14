"""
Robust argument parsing utility for YAML config overrides.

This module provides functionality to:
1. Parse command line arguments that can override any YAML config parameter
2. Handle nested parameters using dot notation (e.g., --difficulty.models)
3. Support config-specific prefixes (--refine, --metrics, --base)
4. Deep merge user overrides with YAML configurations
"""

import argparse
import yaml
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary (typically from YAML config)
        override: Override dictionary (typically from command line args)
        
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


def _parse_nested_key(key: str) -> List[str]:
    """
    Parse a dot-notation key into a list of nested keys.
    
    Args:
        key: Key in dot notation (e.g., "difficulty.models")
        
    Returns:
        List of keys for nested access
    """
    return key.split('.')


def _set_nested_value(config: Dict[str, Any], key_path: List[str], value: Any) -> None:
    """
    Set a value in a nested dictionary using a list of keys.
    
    Args:
        config: Dictionary to modify
        key_path: List of keys representing the path to the value
        value: Value to set
    """
    current = config
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[key_path[-1]] = value


def _convert_value(value: Any, expected_type: Any = None) -> Any:
    """
    Convert value to appropriate Python type.
    
    Args:
        value: Value from command line (could be str, bool, etc.)
        expected_type: The expected type from YAML config (used to determine if single values should be wrapped in list)
        
    Returns:
        Converted value (int, float, bool, list, or str)
    """
    # If already converted by argparse, return as-is
    if isinstance(value, (bool, int, float)):
        return value
    
    # Convert to string for processing
    if not isinstance(value, str):
        value = str(value)
    
    # Handle boolean values
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Handle None
    if value.lower() in ('none', 'null'):
        return None
    
    # Handle lists (comma-separated)
    if ',' in value:
        return [_convert_value(item.strip()) for item in value.split(',')]
    
    # If expected type is a list and we have a single value, wrap it in a list
    if isinstance(expected_type, list):
        # Convert the single value first, then wrap in list
        converted = value
        try:
            if '.' in value:
                converted = float(value)
            else:
                converted = int(value)
        except ValueError:
            pass
        return [converted]
    
    # Handle numbers
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Return as string if no conversion possible
    return value


def _get_nested_value(config: Dict[str, Any], key_path: List[str]) -> Any:
    """
    Get a value from a nested dictionary using a list of keys.
    
    Args:
        config: Dictionary to read from
        key_path: List of keys representing the path to the value
        
    Returns:
        The value at the path, or None if not found
    """
    current = config
    for key in key_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _build_override_dict(args: Dict[str, Any], prefix: Optional[str] = None, original_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a nested dictionary from flat command line arguments.
    
    Args:
        args: Dictionary of command line arguments
        prefix: Optional prefix to filter arguments (e.g., 'refine', 'metrics')
        original_config: Optional original config to determine expected types
        
    Returns:
        Nested dictionary representing the overrides
    """
    override = {}
    
    # List of known config prefixes to exclude when building base overrides
    config_prefixes = ['metrics.', 'refine.', 'skills.']
    
    for key, value in args.items():
        if value is None:
            continue
            
        # Filter by prefix if specified
        if prefix:
            if not key.startswith(f"{prefix}."):
                continue
            # Remove prefix from key
            key = key[len(f"{prefix}."):]
        else:
            # When no prefix specified (building base overrides), exclude config-specific keys
            if any(key.startswith(cp) for cp in config_prefixes):
                continue
        
        # Skip non-config arguments
        if key in ['config_dir', 'help', 'verbose']:
            continue
            
        # Convert dot notation to nested structure
        key_path = _parse_nested_key(key)
        
        # Get expected type from original config if available
        expected_type = None
        if original_config:
            expected_type = _get_nested_value(original_config, key_path)
        
        _set_nested_value(override, key_path, _convert_value(value, expected_type))
    
    return override


class ConfigArgumentParser:
    """
    Enhanced argument parser for YAML config overrides.
    
    This class provides functionality to:
    1. Add arguments for any YAML config parameter
    2. Handle nested parameters with dot notation
    3. Support config-specific prefixes
    4. Merge overrides with YAML configs
    """
    
    def __init__(self, description: str = "MCQA Benchmark Configuration"):
        self.parser = argparse.ArgumentParser(description=description)
        self.configs = {}
    
    def load_config(self, config_path: str, config_name: str) -> Dict[str, Any]:
        """
        Load a YAML config file.
        
        Args:
            config_path: Path to config directory
            config_name: Name of config file
            
        Returns:
            Loaded config dictionary
        """
        full_path = os.path.join(config_path, config_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Config file not found: {full_path}")
        
        with open(full_path, "r") as f:
            return yaml.safe_load(f) or {}
    
    def parse_args_with_overrides(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse command line arguments and return merged configs with overrides.
        
        Args:
            args: Optional list of arguments (defaults to sys.argv)
            
        Returns:
            Dictionary containing merged configs with keys: 'base', 'metrics', 'refine', etc.
        """
        parsed_args = vars(self.parser.parse_args(args))
        
        # Determine config directory
        config_dir = parsed_args.get('config_dir', 'config')
        if not os.path.isabs(config_dir):
            repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_dir = os.path.join(repo_root, config_dir)
        
        # Load base configs
        configs = {}
        
        # Load base config (no prefix)
        base_config_path = os.path.join(config_dir, 'base.yaml')
        if os.path.exists(base_config_path):
            configs['base'] = self.load_config(config_dir, 'base.yaml')
            # Apply base overrides (no prefix) - pass original config for type checking
            base_overrides = _build_override_dict(parsed_args, original_config=configs['base'])
            configs['base'] = _deep_merge_dict(configs['base'], base_overrides)
        
        # Load other configs with prefixes
        for config_name in ['metrics', 'refine', 'skills']:
            config_path = os.path.join(config_dir, f'{config_name}.yaml')
            if os.path.exists(config_path):
                configs[config_name] = self.load_config(config_dir, f'{config_name}.yaml')
                # Apply config-specific overrides - pass original config for type checking
                config_overrides = _build_override_dict(parsed_args, config_name, configs[config_name])
                configs[config_name] = _deep_merge_dict(configs[config_name], config_overrides)
        
        return configs
    
    def add_dynamic_arguments(self, configs: Dict[str, Any]):
        """
        Dynamically add arguments based on loaded configs.
        
        This method scans the configs and adds command line arguments for every
        parameter, using dot notation for nested parameters.
        
        Args:
            configs: Dictionary of loaded configs
        """
        def _add_args_from_dict(d: Dict[str, Any], prefix: str = "", parent_prefix: str = ""):
            for key, value in d.items():
                full_key = f"{parent_prefix}.{key}" if parent_prefix else key
                arg_name = f"--{prefix}.{full_key}" if prefix else f"--{full_key}"
                
                if isinstance(value, dict):
                    _add_args_from_dict(value, prefix, full_key)
                else:
                    # Determine argument type and default
                    if isinstance(value, bool):
                        # Use string type to avoid store_true/store_false always setting a value
                        self.parser.add_argument(
                            arg_name,
                            type=str,
                            help=f"Override {full_key} (use 'true' or 'false', default: {value})"
                        )
                    elif isinstance(value, list):
                        self.parser.add_argument(
                            arg_name,
                            type=str,
                            help=f"Override {full_key} (comma-separated list)"
                        )
                    else:
                        self.parser.add_argument(
                            arg_name,
                            type=str,
                            help=f"Override {full_key} (default: {value})"
                        )
        
        # Add arguments for each config
        for config_name, config_data in configs.items():
            if config_name == 'base':
                _add_args_from_dict(config_data)
            else:
                _add_args_from_dict(config_data, config_name)


def create_config_parser() -> ConfigArgumentParser:
    """
    Create a pre-configured argument parser for MCQA Benchmark.
    
    Returns:
        Configured ConfigArgumentParser instance
    """
    parser = ConfigArgumentParser("MCQA Benchmark - Flexible Configuration Override System")    
    return parser


def load_config_with_overrides(config_path: str = "config", 
                             config_name: str = "base.yaml",
                             user_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced load_config function that accepts user overrides.
    
    Args:
        config_path: Path to config directory
        config_name: Name of config file to load
        user_overrides: Optional dictionary of user overrides
        
    Returns:
        Merged config dictionary
    """
    # Load base config
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    full_config_path = os.path.join(repo_root, config_path, config_name)
    
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Config file not found: {full_config_path}")
    
    with open(full_config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Apply user overrides if provided
    if user_overrides:
        config = _deep_merge_dict(config, user_overrides)
    
    return config


# Convenience function for backward compatibility
def get_merged_configs(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get all configs merged with command line overrides.
    
    Args:
        args: Optional command line arguments
        
    Returns:
        Dictionary of merged configs
    """
    parser = create_config_parser()
    
    # First, load configs to understand structure
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    config_dir = os.path.join(repo_root, 'config')
    
    configs = {}
    for config_name in ['base', 'metrics', 'refine', 'skills']:
        config_path = os.path.join(config_dir, f'{config_name}.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs[config_name] = yaml.safe_load(f) or {}
    
    # Add dynamic arguments based on config structure
    parser.add_dynamic_arguments(configs)
    
    # Parse and return merged configs
    return parser.parse_args_with_overrides(args)
