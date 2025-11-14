import json
from judge_experiments.model.enums import GenerationStrategy, PromptType, SearchEngine

from abc import ABC, abstractmethod
import time
import os
import traceback
import copy

import litellm
from litellm.caching.caching import Cache
litellm.cache = Cache()

from model_utils.web_search import WebSearchType, create_web_searcher


class Generator(ABC):

    MIN_TOKENS = 3
    MAX_TOKENS = 81920
    TEMP = 1.0
    
    @abstractmethod
    def train(self, save_dir):
        """Train the model"""
        pass

    @abstractmethod
    def save(self, save_dir):
        """Save the model"""
        pass

    @abstractmethod
    def merge(self, adapter_dir, save_dir):
        """Merge the PEFT model and save it"""
        pass

    @abstractmethod
    def load(self, save_dir):
        """Load the model"""
        pass

    @abstractmethod
    def upload(self, save_dir):
        """Save the model (upload to the hub)"""
        pass

    @abstractmethod
    def generate(self, prompt):
        """Generate with the model (should only be done after loading)"""
        pass

class HuggingfacePromptGenerator(Generator):
    """Prompt-based generation using Huggingface"""

    def __init__(self, args):

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.args = args
        
        # Use model name directly (assumes correct LiteLLM format)
        self.model_name = self.args.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=args.cache_dir)
        use_8_bit = (self.model_name in {"meta-llama/Llama-3.1-70B-Instruct", "google/gemma-3-27b-it", "Qwen/Qwen3-32B", "Qwen/Qwen3-14B", "google/gemma-3-12b-it"})
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", cache_dir=args.cache_dir, load_in_8bit=use_8_bit)

    @staticmethod
    def parse_response(response_text: str, keys_map: dict) -> str | None:
        """
        Check if a response is in the appropriate JSON format.
        """
        if not response_text:
            return None

        response_text = response_text.replace('`', '').replace('json', '')
        if '{' in response_text and '}' in response_text:
            response_text = response_text[response_text.index('{'):response_text.rindex('}') + 1]
        response_text = response_text.strip()

        try:
            response_json = json.loads(response_text)
            for key, value in keys_map.items():
                if key not in response_json:
                    return None
                if type(value) == type([]) and len(value) > 0:
                    if response_json[key] not in value:
                        return None
                else:
                    if type(value) != type(response_json[key]):
                        return None
            return response_json
        except Exception as e:
            print(f"Error parsing response: {e}", flush=True)
            print(response_text, flush=True)
            return None

    def train(self, run_name, save_dir, train_dataset, val_dataset):
        # No training for prompt-based generation
        pass

    def save(self, save_dir):
        # No model to save for prompt-based generation
        pass

    def merge(self, adapter_dir, save_dir):
        # No adapter to merge for prompt-based generation
        pass

    def load(self, save_dir):
        # No model loading needed for API-based generation
        print(f"PromptGenerator configured to use model: {self.model_name}", flush=True)
        pass

    def generate(self, prompt: str, max_retries: int = 5):
        """Generate response with automatic retry for failed responses.
        
        Args:
            prompt: The input prompt
            max_retries: Maximum number of retries for failed responses
            
        Returns:
            Dictionary with response, cost, and reasoning_trace
        """

        import torch
        
        # Try to generate a valid response
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": str(prompt)}]

                input_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
                ).to("cuda")

                eos = getattr(self.tokenizer, "eos_token_id", None) or getattr(self.model.generation_config, "eos_token_id", None)
                pad = getattr(self.tokenizer, "pad_token_id", None) or eos

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=1024,
                        eos_token_id=eos,
                        pad_token_id=pad,
                        use_cache=True,
                    ).to("cpu")

                # Keep 2D
                new_tokens = outputs[:, input_ids.shape[1]:]
                response_content = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
                
                # Check if response contains valid JSON format
                KEYS_MAP = {
                    PromptType.contamination: {'result': ['exact_match', 'no_match', 'partial_match', 'question_match'], 'citations': [], 'explanation': ''},
                    PromptType.shortcuts: {'decision': ['exact_match', 'knowledge_match', 'no_match'], 'explanation': ''},
                    PromptType.writing_flaws: {'result': ['pass', 'fail'], 'explanation': '', 'confidence': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                }
                parsed_response = self.parse_response(response_content, KEYS_MAP[self.args.prompt_type])
                if parsed_response is None:
                    print(f"Attempt {attempt + 1}/{max_retries}: Response missing valid answer pattern, retrying...", flush=True)
                    print(response_content, flush=True)
                    print(flush=True)
                    continue
                
                # Response is valid, extract cost and reasoning trace
                cost = 0.0
                
                return {
                    'response': parsed_response,
                    'cost': cost,
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed in HF generation: {e}", flush=True)
                print(f"Traceback: {traceback.format_exc()}", flush=True)
                if attempt == max_retries - 1:  # Last attempt
                    return {
                        'response': f"Error: {e}",
                        'cost': 0.0,
                        'reasoning_trace': ""
                    }
                time.sleep(2**attempt)  # Wait before retry

        # If we get here, all attempts failed to produce a valid response
        return {
            'response': "Error: Unable to generate valid response after all attempts",
            'cost': 0.0,
            'reasoning_trace': ""
        }

    def upload(self, save_dir):
        # No model to upload for API-based generation
        print("PromptGenerator uses API calls - no model to upload", flush=True)
        pass

class PromptGenerator(Generator):
    """Prompt-based generation using LiteLLM API"""

    def __init__(self, args):
        self.args = args
        
        # Set API keys from args if available
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            os.environ["HUGGINGFACE_API_KEY"] = hf_token
        
        # Set OpenAI API key if available in environment
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            
        # Set Anthropic API key if available in environment  
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        
        # Use model name directly (assumes correct LiteLLM format)
        self.model_name = self.args.model_name

        if 'sonnet' in self.model_name:
            self.MAX_TOKENS = 64000
        if 'haiku' in self.model_name or 'command-r' in self.model_name:
            self.MAX_TOKENS = 8192
        if 'opus' in self.model_name or 'command-a-reasoning' in self.model_name:
            self.MAX_TOKENS = 32000
        if 'gpt-4.1' in self.model_name:
            self.MAX_TOKENS = 32768

    @staticmethod
    def parse_response(response_text: str, keys_map: dict) -> str | None:
        """
        Check if a response is in the appropriate JSON format.
        """
        if not response_text:
            return None

        response_text = response_text.replace('`', '').replace('json', '')
        if '{' in response_text and '}' in response_text:
            response_text = response_text[response_text.index('{'):response_text.rindex('}') + 1]
        response_text = response_text.strip()

        try:
            response_json = json.loads(response_text)
            for key, value in keys_map.items():
                if key not in response_json:
                    return None
                if type(value) == type([]) and len(value) > 0:
                    if response_json[key] not in value:
                        return None
                else:
                    if type(value) != type(response_json[key]):
                        return None
            return response_json
        except Exception as e:
            print(f"Error parsing response: {e}", flush=True)
            print(response_text, flush=True)
            return None

    def train(self, run_name, save_dir, train_dataset, val_dataset):
        # No training for prompt-based generation
        pass

    def save(self, save_dir):
        # No model to save for prompt-based generation
        pass

    def merge(self, adapter_dir, save_dir):
        # No adapter to merge for prompt-based generation
        pass

    def load(self, save_dir):
        # No model loading needed for API-based generation
        print(f"PromptGenerator configured to use model: {self.model_name}", flush=True)
        pass

    def generate(self, prompt: str, max_retries: int = 5):
        """Generate response with automatic retry for failed responses.
        
        Args:
            prompt: The input prompt
            max_retries: Maximum number of retries for failed responses
            
        Returns:
            Dictionary with response, cost, and reasoning_trace
        """
        
        # Try to generate a valid response
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": str(prompt)}]

                llm_args = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.MAX_TOKENS,
                    "temperature": self.TEMP,
                }

                if self.model_name in {'together_ai/deepseek/DeepSeek-R1'}:
                    self.generate_deepseek(messages)
                    
                elif self.model_name in {'openai/gpt-5-2025-08-07', 'openai/gpt-5-mini-2025-08-07', 'openai/gpt-5-nano-2025-08-07'}:

                    gpt_llm_args = copy.deepcopy(llm_args)
                    gpt_llm_args["input"] = gpt_llm_args["messages"]
                    del gpt_llm_args["messages"]
                    response = litellm.responses(**gpt_llm_args, reasoning={"effort": "minimal", "summary": "detailed"})
                    response = litellm.get_responses(response_id=response.id)
                    outputs = response.output
                    response_content = [o for o in outputs if o.type == "message"][0].content[0].text
                    reasoning_content = '\n<nishant delimeter>\n'.join(
                        s.text
                        for o in outputs if o.type == "reasoning"
                        for s in (o.summary if isinstance(o.summary, list) else [o.summary])
                    )
                    num_reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                    reasoning_content += f'\n<nishant end delimeter>\nReasoning tokens: {num_reasoning_tokens}'

                else:

                    response = litellm.completion(**llm_args)

                    if not response or not response.choices[0].message.content or response.choices[0].finish_reason != "stop":
                        print(f"Attempt {attempt + 1}/{max_retries}: Invalid response format, retrying...", flush=True)
                        continue
                    
                    response_content = response.choices[0].message.content
                    reasoning_content = ""
                    
                    try:    
                        reasoning_content = response.choices[0].message.reasoning_content
                    except Exception as _:
                        pass
                
                KEYS_MAP = {
                    PromptType.contamination: {'result': ['exact_match', 'no_match', 'partial_match', 'question_match'], 'citations': [], 'explanation': ''},
                    PromptType.shortcuts: {'decision': ['exact_match', 'knowledge_match', 'no_match'], 'explanation': ''},
                    PromptType.writing_flaws: {'result': ['pass', 'fail'], 'explanation': '', 'confidence': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                }
                parsed_response = self.parse_response(response_content, KEYS_MAP[self.args.prompt_type])
                if parsed_response is None:
                    print(f"Attempt {attempt + 1}/{max_retries}: Response missing valid answer pattern, retrying...", flush=True)
                    print(response_content, flush=True)
                    print(flush=True)
                    continue

                # Response is valid, extract cost and reasoning trace
                cost = 0.0
                if hasattr(response, '_hidden_params') and response._hidden_params:
                    cost = response._hidden_params.get('response_cost', 0.0)
                elif hasattr(response, 'usage') and response.usage:
                    cost = getattr(response.usage, 'total_cost', 0.0)
                
                return {
                    'response': parsed_response,
                    'cost': cost,
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed in LiteLLM generation: {e}", flush=True)
                print(f"Traceback: {traceback.format_exc()}", flush=True)
                if attempt == max_retries - 1:  # Last attempt
                    return {
                        'response': f"Error: {e}",
                        'cost': 0.0,
                        'reasoning_trace': ""
                    }
                time.sleep(2**attempt)  # Wait before retry

        # If we get here, all attempts failed to produce a valid response
        return {
            'response': "Error: Unable to generate valid response after all attempts",
            'cost': 0.0,
            'reasoning_trace': ""
        }

    def upload(self, save_dir):
        # No model to upload for API-based generation
        print("PromptGenerator uses API calls - no model to upload", flush=True)
        pass

class WebSearchGenerator(Generator):
    """Web search generator that performs web searches and saves results"""

    def __init__(self, args):
        self.args = args
        self.search_type = self._convert_search_engine(args.search_engine)
        self.web_searcher = create_web_searcher(
            search_type=self.search_type,
            try_scraping=args.try_scraping
        )

    def _convert_search_engine(self, search_engine: SearchEngine) -> WebSearchType:
        """Convert SearchEngine enum to WebSearchType enum."""
        if search_engine == SearchEngine.google:
            return WebSearchType.GOOGLE
        elif search_engine == SearchEngine.brave:
            return WebSearchType.BRAVE
        elif search_engine == SearchEngine.perplexity:
            return WebSearchType.PERPLEXITY
        elif search_engine == SearchEngine.exa:
            return WebSearchType.EXA
        elif search_engine == SearchEngine.tavily:
            return WebSearchType.TAVILY
        elif search_engine == SearchEngine.serper:
            return WebSearchType.SERPER
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")

    def train(self, run_name, save_dir, train_dataset, val_dataset):
        # No training needed for web search
        pass

    def save(self, save_dir):
        # No model saving needed for web search
        pass

    def merge(self, adapter_dir, save_dir):
        # No merging needed for web search
        pass

    def load(self, save_dir):
        # No model loading needed for web search
        print(f"WebSearchGenerator configured to use search engine: {self.search_type.value}", flush=True)
        pass

    def upload(self, save_dir):
        # No uploading needed for web search
        pass

    def generate(self, prompt):
        """Generate web search results for the given prompt.
        
        Args:
            prompt: The search query (should contain question and answer)
            
        Returns:
            Dictionary with search results formatted as citations
        """
        try:
            # Extract question and answer from prompt
            # The prompt should be formatted as a search query
            query = str(prompt)
            
            # Perform web search
            search_results = self.web_searcher.search(
                query, 
                max_results=self.args.max_results, 
                max_tokens_per_page=self.args.max_tokens_per_page
            )
            
            # Format results for storage
            formatted_results = []
            citations = []
            
            for idx, result in enumerate(search_results):
                formatted_result = {
                    'content': result.content,
                    'metadata': result.metadata,
                    'citation_id': idx + 1
                }
                formatted_results.append(formatted_result)
                
                # Create citation string
                citation = f"<citation {idx+1}>\n{result.content}\n</citation {idx+1}>"
                citations.append(citation)
            
            # Return results in the same format as other generators
            return {
                'response': formatted_results,
                'cost': 0.0,  # Web search doesn't have LLM costs
                'reasoning_trace': '',
                'search_results': formatted_results,
                'num_results': len(search_results),
                'search_type': self.search_type.value,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'response': ["Error: " + str(e)],
                'cost': 0.0,
                'reasoning_trace': '',
                'search_results': [],
                'num_results': 0,
                'search_type': self.search_type.value,
                'success': False,
                'error': str(e)
            }

class GeneratorFactory:

    def __init__(self, args):
        self.args = args

    def get_generator(self, generation_strategy: GenerationStrategy):
        if generation_strategy == GenerationStrategy.prompt:
            return PromptGenerator(self.args)
        elif generation_strategy == GenerationStrategy.prompt_hf:
            return HuggingfacePromptGenerator(self.args)
        elif generation_strategy == GenerationStrategy.web_search:
            return WebSearchGenerator(self.args)
        else:
            raise ValueError(f"Unsupported Generator type: {generation_strategy}. Supported types: grpo, sft, prompt, prompt_hf, web_search")

    