from enum import Enum

class PromptType(Enum):
    contamination_pretrain = 'contamination_pretrain'
    shortcuts = 'shortcuts'
    writing_flaws = 'writing_flaws'
    contamination = 'contamination'
    web_search = 'web_search'

class GenerationStrategy(Enum):
    prompt = 'prompt'
    pretraining = 'pretraining'
    prompt_hf = 'prompt_hf'
    web_search = 'web_search'

class SearchEngine(Enum):
    google = 'google'
    brave = 'brave'
    perplexity = 'perplexity'
    exa = 'exa'
    tavily = 'tavily'
    serper = 'serper'