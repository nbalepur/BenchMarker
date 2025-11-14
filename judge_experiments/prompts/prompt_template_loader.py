"""
Abstract base class implementation of a Prompt Template. Uses a specified experiment to obtain a prompt template (e.g. f-string) that can include data inputs.
"""

from judge_experiments.model.enums import PromptType, SearchEngine
from abc import ABC, abstractmethod
import json
import os

# Import existing prompt functions from scorers
from prompts.contamination_prompt import get_contamination_prompt
from prompts.shortcut_prompts import QUESTION_DETECTION_PROMPT
from prompts.writing_flaw_prompts import get_writing_flaw_prompts

# Abstract base class for implementing prompts
class Prompt(ABC):

    def __init__(self, prompt_file, delim='\n\n'):
        if prompt_file:
            with open(prompt_file, 'r') as f:
                self.template = f.read()
            f.close()
        self.delim = delim

    @abstractmethod
    def create_inference_prompt(self):
        """Create the inference part of the prompt"""
        pass

    def create_prompt(self, **kwargs):
        """Create the full prompt"""
        input_text = self.create_inference_prompt(**kwargs)
        return input_text

class ContaminationJudgePrompt(Prompt):
    """Prompt for contamination judge - evaluates if model answers are contaminated"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_inference_prompt(self, question, choices, answer, search_results=None):
        contamination_prompt = get_contamination_prompt()
        
        # Format citations from search results
        if search_results is not None:
            citations = []
            for result in search_results.get('search_results', []):
                # Use the processed content (already formatted based on try_scraping flag)
                content = result.get('content', '')
                
                if content:
                    citation = f"<citation {result['citation_id']}>\n{content}\n</citation {result['citation_id']}>"
                    citations.append(citation)
            citations = '\n'.join(citations) if citations else "No search results found."
        else:
            citations = "No search results found."
        
        correct_answer = choices[ord(answer) - ord('A')]
        return contamination_prompt.format(
            question=question,
            correct_answer=correct_answer,
            citations=citations
        )

class ShortcutsJudgePrompt(Prompt):
    """Prompt for shortcuts judge - evaluates if model uses shortcuts instead of reasoning"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_inference_prompt(self, question, choices, answer, model_response, inferred_question):
        return QUESTION_DETECTION_PROMPT.format(
            question=question,
            response=model_response,
            inferred_question=inferred_question
        )

class WritingFlawsJudgePrompt(Prompt):
    """Prompt for writing flaws judge - evaluates writing quality issues"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_inference_prompt(self, question, choices, answer, flaw_type):
        writing_flaw_prompts = get_writing_flaw_prompts()
        template = [prompt for (key, prompt) in writing_flaw_prompts if key.value == flaw_type][0]
        choices_str = [f'({chr(ord("A") + idx)}) {c}' for idx, c in enumerate(choices)]
        return template.format(
            question=question,
            choices='\n' + '\n'.join(choices_str),
            answer=answer,
            lbrace="{",
            rbrace="}"
        )

class WebSearchPrompt(Prompt):
    """Prompt for web search - creates search queries from questions and answers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_inference_prompt(self, question, choices, answer, search_engine=None):
        # Create search query similar to contamination scorer
        answer_text = choices[ord(answer) - ord('A')]
        query = f'"{question}" "{answer_text}"'
        
        # Handle query length limits for Brave search engine only
        if search_engine and search_engine == SearchEngine.brave and len(question.split() + answer_text.split()) > 50:
            # Brave has a 50 word query limit
            query = ' '.join(question.split()[:45])
            query = f'"{query}"'
        
        return query

class PromptFactory:

    def __init__(self, args):        
        # No template files needed - prompts are created programmatically
        self.dir = None
        self.args = args
        
    def get_prompt(self, prompt_type) -> Prompt:
        if prompt_type == PromptType.contamination:
            return ContaminationJudgePrompt(prompt_file=self.dir)
        elif prompt_type == PromptType.shortcuts:
            return ShortcutsJudgePrompt(prompt_file=self.dir)
        elif prompt_type == PromptType.writing_flaws:
            return WritingFlawsJudgePrompt(prompt_file=self.dir)
        elif prompt_type == PromptType.web_search:
            return WebSearchPrompt(prompt_file=self.dir)
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")