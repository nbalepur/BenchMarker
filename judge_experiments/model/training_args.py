# Unified training arguments for all model endpoints
import argparse
import os
from dotenv import load_dotenv
from judge_experiments.model.enums import PromptType, GenerationStrategy, SearchEngine


def enum_type(enum):
    """Helper function to create enum type converter for argparse"""
    enum_members = {e.name: e for e in enum}

    def converter(input):
        out = []
        for x in input.split():
            if x in enum_members:
                out.append(enum_members[x])
            else:
                raise argparse.ArgumentTypeError(f"You used {x}, but value must be one of {', '.join(enum_members.keys())}")
        return out[0]

    return converter


def load_environment_tokens():
    """Load tokens from environment variables using python-dotenv"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set tokens from environment variables
    hf_token = os.getenv('HUGGINGFACE_TOKEN', os.getenv('HF_TOKEN', ''))
    
    return hf_token


def setup_training_args():
    """Setup and return unified training arguments parser"""
    # Load tokens from environment
    hf_token = load_environment_tokens()
    
    parser = argparse.ArgumentParser()
    
    # Run identification
    parser.add_argument(
        "--run_name",
        type=str,
        help="String to identify this run",
        default="default",
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model on the official API (huggingface, OpenAI, etc)",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--generation_strategy",
        type=enum_type(GenerationStrategy),
        help="How to run the model",
        default="prompt",
    )
    parser.add_argument(
        "--load_in_8bit",
        action='store_true',
        help="Should we load the model in 8 bit?",
        default=False,
    )
    parser.add_argument(
        "--load_in_4bit",
        action='store_true',
        help="Should we load the model in 4 bit?",
        default=False,
    )
    parser.add_argument(
        "--device_map",
        type=str,
        help="Where to load the model ('cuda', 'auto', 'cpu')",
        default="auto",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Local path or huggingface path pointing to the dataset",
        default="nbalepur/cheating-reasoners",
    )

    # Prompt configuration
    parser.add_argument(
        "--prompt_type",
        type=enum_type(PromptType),
        help="Prompt type to use",
        default="mcqa",
    )
    
    # Search engine configuration
    parser.add_argument(
        "--search_engine",
        type=enum_type(SearchEngine),
        help="Search engine type for contamination detection",
        default="google",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        help="Maximum number of search results to retrieve",
        default=5,
    )
    parser.add_argument(
        "--max_tokens_per_page",
        type=int,
        help="Maximum tokens per page for search results",
        default=512,
    )
    parser.add_argument(
        "--try_scraping",
        action='store_true',
        help="Whether to try web scraping for additional content",
        default=False,
    )
    
    # Directory paths
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Absolute directory of the cache folder for models",
        default="./",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Absolute directory of the results folder",
        default="./",
    )
    
    args = parser.parse_args()
    
    # Add tokens to args object
    args.hf_token = hf_token
    
    print(args)
    return args