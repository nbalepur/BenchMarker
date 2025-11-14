from judge_experiments.model.enums import PromptType
import os
import json
import datasets
from pathlib import Path
from datasets.utils.logging import disable_progress_bar
from typing import Any
disable_progress_bar()

from abc import ABC, abstractmethod

class DataFetcher(ABC):
    @abstractmethod
    def get_data(self):
        """Retrieve data from the source."""
        """Output: List of keyword arguments in dictionary form"""
        pass

    def collect_files(self, keywords, res_dir):
        """Collect all files with valid keywords"""
        valid_files = []
        directory = Path(res_dir)
        for file in directory.rglob('*'):
            if file.is_file():
                valid = True
                for substr in keywords:
                    if substr.strip() not in str(file):
                        valid = False
                        break
                if valid:
                    valid_files.append(file)
        return valid_files

class JudgeDataFetcher(DataFetcher):
    """Loads the judge data to validate on"""

    def __init__(self, args):
        self.args = args
        self.data = self.load_json_dataset(args.dataset_name)

    def load_json_dataset(self, json_path: str) -> Any:
        """Load data from a JSONL file"""
        data = []
        with open(json_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
            
    def get_data(self):
        """Return data in the expected format"""
        inputs = []
        outputs = []
        labels = []
        
        for item in self.data:
            input_data = {
                'question': item['question'],
                'choices': item['choices']
            }
            
            # Add search engine for web search prompts
            if hasattr(self.args, 'prompt_type') and self.args.prompt_type.value == 'web_search':
                input_data['search_engine'] = self.args.search_engine
            
            inputs.append(input_data)
            outputs.append(item['answer'])
            labels.append(item['label'])
            
            # extra inputs
            for k, v in item.items():
                if k not in {'question', 'choices', 'answer', 'label', 'dataset'}:
                    input_data[k] = v
        
        return {
            'input': {'test': inputs},
            'output': {'test': outputs},
            'label': {'test': labels},
        }

class JudgeDataFetcherWithSearch(JudgeDataFetcher):
    """Loads the judge data with search results for contamination detection"""

    def __init__(self, args):
        super().__init__(args)
        self.search_results_data = {}
        self.load_search_results()

    def load_search_results(self):
        """Load pre-computed search results from the dedicated web_search results JSONL, keyed by prompt_idx.

        Expected path example:
        {res_dir}/{run_name}/web_search/{search_engine}/web_search.jsonl
        """
        from judge_experiments.model.utils import load_jsonl_file

        # Build path using the web_search generation strategy and the search engine name
        res_dir = getattr(self.args, 'res_dir', 'judge_experiments/results')
        run_name = getattr(self.args, 'run_name', 'default')
        search_engine = getattr(self.args, 'search_engine', None)
        if search_engine is None or not hasattr(search_engine, 'value'):
            print("Warning: search_engine not specified; cannot load web search results.")
            return

        search_results_path = os.path.join(
            res_dir,
            run_name,
            'web_search',               # generation strategy used to create search results
            search_engine.value,        # directory is the engine name (e.g., 'google')
            'web_search.jsonl'          # standard filename for web search outputs
        )

        if not os.path.exists(search_results_path):
            print(f"Warning: Search results file not found at {search_results_path}")
            return

        try:
            data_rows = load_jsonl_file(search_results_path)
            for row in data_rows:
                prompt_idx = row.get('prompt_idx')
                if prompt_idx is None:
                    continue
                self.search_results_data[prompt_idx] = {
                    'search_results': row.get('search_results', []),
                    'num_results': row.get('num_results', 0),
                    'search_type': row.get('search_type', ''),
                    'success': row.get('success', True),
                    'error': row.get('error', None),
                }

            print(f"Loaded {len(self.search_results_data)} pre-computed search results from {search_results_path}")
        except Exception as e:
            print(f"Error loading search results from {search_results_path}: {e}")

    def get_data(self):
        """Return data in the expected format with search results included"""
        inputs = []
        outputs = []
        labels = []
        extra_inputs = []
        
        for idx, item in enumerate(self.data):
            input_data = {
                'question': item['question'],
                'choices': item['choices']
            }
            
            search_results = self.search_results_data.get(idx, {'search_results': [], 'num_results': 0})
            input_data['search_results'] = search_results
            
            inputs.append(input_data)
            outputs.append(item['answer'])
            labels.append(item['label'])
            
            # Handle optional extra_inputs field
            if 'extra_inputs' in item:
                extra_inputs.append(item['extra_inputs'])
            else:
                extra_inputs.append(None)
        
        return {
            'input': {'test': inputs},
            'output': {'test': outputs},
            'label': {'test': labels},
            'extra_inputs': {'test': extra_inputs}
        }

    def _process_content(self, search_results, try_scraping=False):
        """Process search results content based on try_scraping flag"""
        if 'search_results' not in search_results:
            return search_results
        
        processed_results = []
        for result in search_results['search_results']:
            processed_result = result.copy()
            
            # If metadata contains snippet and/or scraped content, process them
            if 'metadata' in result and isinstance(result['metadata'], dict):
                metadata = result['metadata']
                
                # Get snippet and scraped page content
                snippet = metadata.get('snippet', '')
                scraped_page = metadata.get('scraped_page', '')
                
                if try_scraping:
                    # When try_scraping is enabled: use snippet + " " + scraped_page
                    if snippet and scraped_page and scraped_page.strip():
                        combined_content = snippet + " " + scraped_page
                    elif snippet:
                        combined_content = snippet
                    elif scraped_page and scraped_page.strip():
                        combined_content = scraped_page
                    else:
                        combined_content = ''
                else:
                    # When try_scraping is disabled: use just snippet
                    combined_content = snippet
                
                processed_result['content'] = combined_content
                processed_result['snippet'] = snippet
                processed_result['scraped_page'] = scraped_page
                
                # Keep other metadata
                processed_result['metadata'] = metadata
            
            processed_results.append(processed_result)
        
        search_results['search_results'] = processed_results
        return search_results

class DataFetcherFactory:

    @staticmethod
    def get_data_fetcher(args: Any, prompt_type: PromptType):
        if prompt_type in {PromptType.writing_flaws, PromptType.web_search, PromptType.shortcuts}:
            return JudgeDataFetcher(args=args)
        if prompt_type == PromptType.contamination:
            return JudgeDataFetcherWithSearch(args=args)
        else:
            raise ValueError(f"Unsupported DataFetcher type: {prompt_type}")