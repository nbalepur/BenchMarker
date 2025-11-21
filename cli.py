#!/usr/bin/env python3
"""
MCQA Benchmark CLI - Flexible Configuration Override System

This CLI allows you to run any MCQA benchmark operation with full control over
all configuration parameters through command line arguments.

Usage Examples:
    # Single steps
    python cli.py --steps skills
    python cli.py --steps metrics --metrics.difficulty.models openai/gpt-4o,openai/gpt-4o-mini
    python cli.py --steps refine --refine.shortcuts.model openai/gpt-4o --refine.shortcuts.type rewrite
    
    # Multiple steps (pipelines)
    python cli.py --steps skills,metrics          # skills → metrics
    python cli.py --steps skills,metrics,refine   # skills → metrics → refine
    python cli.py --steps metrics,refine          # metrics → refine
    
    # With parameter overrides
    python cli.py --steps metrics --metrics.difficulty.models openai/gpt-4o --metrics.num_samples 20
    python cli.py --steps skills,metrics,refine --dataset /path/to/dataset.csv --refine.shortcuts.type rewrite
"""

import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

from utils.argparse_config import get_merged_configs
from endpoints.run_metrics import run_metrics_eval
from endpoints.run_refine import run_refine_eval
from endpoints.run_skills import run_skills_eval

def filter_config_args(all_args, command_name):
    """Filter out CLI-specific arguments and return only relevant config arguments."""
    config_args = []
    skip_next = False
    
    # Define which argument patterns are relevant for each command
    relevant_patterns = {
        'main': [
            # All config arguments since we have one entry point
            '--dataset', '--metrics', '--save_dir', '--plot_dir', '--cache_type',
            '--skill_run_name', '--metric_run_name', '--refine_run_name',
            '--scoring_metrics', '--refining_metrics', '--cache_dir', '--dataset_save_dir',
            '--metrics.', '--refine.', '--skills.'
        ]
    }
    
    # Get relevant patterns for this command
    patterns = relevant_patterns.get(command_name, [])
    
    i = 0
    while i < len(all_args):
        arg = all_args[i]
        
        # Skip CLI-specific arguments
        if arg.startswith('--steps'):
            if arg.startswith('--steps='):
                i += 1
                continue  # Skip --steps=value
            elif arg == '--steps':
                i += 2  # Skip --steps and its value
                continue
        else:
            # Only include arguments that match relevant patterns
            if any(arg.startswith(pattern) for pattern in patterns):
                config_args.append(arg)
                # If this is a flag that takes a value, include the next argument too
                if i + 1 < len(all_args) and not all_args[i + 1].startswith('--'):
                    config_args.append(all_args[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    
    return config_args


def create_main_parser():
    """Create the main argument parser with steps parameter."""
    parser = argparse.ArgumentParser(
        description="MCQA Benchmark - Flexible Configuration Override System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Configuration System:
Most parameters are defined in config files (base.yaml, metrics.yaml, etc.) but can be 
overridden from the command line. Only specify parameters you want to change - everything 
else uses the YAML defaults.

For a complete list of all available parameters:
    - See PARAMS.md for detailed documentation
    - Run: python -c "from utils.setup import show_all_parameters; show_all_parameters()"

Examples:
    # Run with default settings (uses YAML configs)
    python cli.py --steps metrics
    
    # Override specific parameters
    python cli.py --steps metrics --metrics.num_samples 20 --metrics.difficulty.models openai/gpt-4o
    
    # Run pipeline with custom dataset
    python cli.py --steps skills,metrics,refine --dataset /path/to/dataset.csv

All parameters are optional - only override what you want to change!"""
    )
    
    parser.add_argument(
        '--steps',
        default='skills,metrics,refine',
        help='Comma-separated list of steps to run (e.g., skills,metrics,refine). Default: skills,metrics,refine'
    )
    
    return parser


def run_steps(args):
    """Run specified steps with merged configs."""
    # Parse steps
    steps = [step.strip() for step in args.steps.split(',')]
    
    # Get merged configs once for all steps
    config_args = filter_config_args(sys.argv[1:], 'main')
    configs = get_merged_configs(config_args)
    
    # Run each step in sequence, passing configs directly
    for step in steps:
        if step == 'skills':
            run_skills_eval(
                skills_cfg=configs.get('skills', {}),
                base_cfg=configs.get('base', {})
            )
        elif step == 'metrics':
            run_metrics_eval(
                metrics_cfg=configs.get('metrics', {}),
                base_cfg=configs.get('base', {})
            )
        elif step == 'refine':
            run_refine_eval(
                metrics_cfg=configs.get('metrics', {}),
                refine_cfg=configs.get('refine', {}),
                base_cfg=configs.get('base', {})
            )
        else:
            print(f"Unknown step: {step}")
            print("Available steps: skills, metrics, refine")
            exit(1)


def main():
    """Main entry point for the CLI."""
    parser = create_main_parser()
    
    args, _ = parser.parse_known_args()
    run_steps(args)

if __name__ == "__main__":
    main()
