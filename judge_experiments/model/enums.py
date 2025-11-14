from enum import Enum

class PromptType(Enum):
    shortcuts = 'shortcuts'
    writing_flaws = 'writing_flaws'
    contamination = 'contamination'
    web_search = 'web_search'

class GenerationStrategy(Enum):
    prompt = 'prompt'
    prompt_hf = 'prompt_hf'
    web_search = 'web_search'

class SearchEngine(Enum):
    google = 'google'
    brave = 'brave'
    perplexity = 'perplexity'
    exa = 'exa'
    tavily = 'tavily'
    serper = 'serper'