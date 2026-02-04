from dataclasses import dataclass

from inspect_ai.solver import solver
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState
from prompts.writing_flaw_prompts import WritingFlaw, rules
from prompts.rewrite_prompts import REWRITE_PROMPT, BLOOMS_TAXONOMY_PROMPT, ADD_DISTRACTORS_PROMPT
from utils.enums import Metrics
from enum import Enum
import json
import random

class RefineType(Enum):
    REWRITE = 'rewrite'
    FILTER = 'filter'
    FEEDBACK = 'feedback'
    NONE = 'none'

class DifficultyRefineType(Enum):
    EFFICIENCY = 'efficiency'
    SATURATION = 'saturation'
    INFORMATIVE = 'informative'
    ADD_DISTRACTORS = 'add_distractors'
    BLOOMS_TAXONOMY = 'blooms_taxonomy'
    NONE = 'none'

class RewriteExperiment(Enum):
    FLAW_CORRECTION = 'flaw_correction'
    BLOOMS_TAXONOMY = 'blooms_taxonomy'
    ADD_DISTRACTORS = 'add_distractors'

@dataclass
class MCQ:
    question: str
    choices: list[str]
    answer: str

    def to_json(self):
        return {
            'question': self.question,
            'choices': self.choices,
            'answer': self.answer
        }

    def to_prompt(self) -> str:
        return "Question Stem: " + self.question + "\nChoices:\n" + "\n".join([f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(self.choices)]) + "\nAnswer: " + self.answer

async def rewrite_mcq(mcq: MCQ, feedback: str, refine_config: dict, rewrite_experiment: RewriteExperiment) -> MCQ:    

    rewrite_model = get_model(refine_config.get('rewrite_model'))
    
    if rewrite_experiment == RewriteExperiment.FLAW_CORRECTION:
        rewrite_prompt = REWRITE_PROMPT.format(mcq=mcq.to_prompt(), feedback=feedback)
    elif rewrite_experiment == RewriteExperiment.BLOOMS_TAXONOMY:
        rewrite_prompt = BLOOMS_TAXONOMY_PROMPT.format(mcq=mcq.to_prompt(), num_blooms_levels=refine_config.get('difficulty').get('num_blooms_levels'))
    elif rewrite_experiment == RewriteExperiment.ADD_DISTRACTORS:
        rewrite_prompt = ADD_DISTRACTORS_PROMPT.format(mcq=mcq.to_prompt(), num_distractors=refine_config.get('difficulty').get('num_distractors'))

    for _ in range(refine_config.get('rewrite_attempts')):
        try:
            output = await rewrite_model.generate(rewrite_prompt)
            output = output.completion.replace('`', '').replace('json', '')
            if '{' in output:
                output = output[output.index('{'):output.rindex('}') + 1]
            output = json.loads(output)
            if output.get('question', '') == '' or output.get('choices', []) == [] or output.get('answer', '') == '' or output.get('explanation', '') == '':
                continue
            
            # For the 'add distractors' experiment, shuffle the choices to avoid having the model-generated distractors at the end
            if rewrite_experiment == RewriteExperiment.ADD_DISTRACTORS:
                answer_letter = output['answer']
                correct_idx = ord(answer_letter.upper()) - ord('A')
            
                indices = list(range(len(output['choices'])))
                random.shuffle(indices)
                
                shuffled_choices = [output['choices'][current_idx] for current_idx in indices]
                
                new_correct_idx = indices.index(correct_idx)
                new_answer_letter = chr(ord('A') + new_correct_idx)
                
                new_mcq = MCQ(question=output['question'], choices=shuffled_choices, answer=new_answer_letter)
                print('new mcq:\n', new_mcq.to_prompt())
            else:
                new_mcq = MCQ(question=output['question'], choices=output['choices'], answer=output['answer'])
            
            return new_mcq, output['explanation']
        except Exception as e:
            print(f"Error rewriting MCQ: {e}")
            continue

    return mcq, ''
  
async def refine_mcq(mcq: MCQ, refinements: list[dict], refine_config: dict, rewrite_experiment: RewriteExperiment) -> MCQ:
    """Main function to refine the MCQ"""

    combined_feedback = ''

    if rewrite_experiment == RewriteExperiment.FLAW_CORRECTION:

        feedback_parts = []
        all_feedback = []

        for refinement in refinements:
            if not refinement['should_refine'] or refinement['refine_type'] == RefineType.NONE:
                continue
            if refinement['refine_type'] == RefineType.FILTER:
                return None, ''
            
            # Get individual instructions for this refinement and format them
            instructions = refinement['feedback']
            metric_name = refinement["score_name"].replace("_", " ").title()
            formatted_feedback = f'<{metric_name}>\n' + '\n'.join(instructions) + f'\n</{metric_name}>'
            
            # Always collect feedback for logging
            all_feedback.append(formatted_feedback)
            
            # Only add to rewrite feedback if it's not FEEDBACK type
            if refinement['refine_type'] != RefineType.FEEDBACK:
                feedback_parts.append(formatted_feedback)
        
        # If only FEEDBACK types, return original MCQ with logged feedback
        if len(feedback_parts) == 0 and len(all_feedback) > 0:
            print("MCQ has feedback-only refinements, returning the original with logged feedback")
            return mcq, 'Feedback generated but no rewriting performed'
        
        if len(feedback_parts) == 0:
            print("MCQ passed all metrics, returning the original")
            return mcq, ''
        
        # Concatenate all feedback into a single string for rewriting
        combined_feedback = '\n'.join(feedback_parts)

    return await rewrite_mcq(mcq, combined_feedback, refine_config, rewrite_experiment)


def _get_metric_feedback(score_name: str, score_value, metadata: dict) -> list[str]:
    """Extract individual feedback instructions for a given metric."""
    if score_name == Metrics.WRITING_FLAWS.value:
        instructions = []
        for flaw in metadata['writing_flaws']:
            flaw_name = flaw['name'].replace("_", " ").title()
            rule = rules[WritingFlaw(flaw['name'])]["rule"]
            examples = rules[WritingFlaw(flaw['name'])]["examples"]
            fix = rules[WritingFlaw(flaw['name'])]["fix"]
            explanation = flaw['explanation']
            
            instruction = f"- Violated the writing rule {flaw_name}: {rule}. Here's why: {explanation}"
            instructions.append(instruction)
        return instructions
    
    elif score_name == Metrics.CONTAMINATION.value:
        match_type = score_value.answer.replace("_", " ").title()
        explanation = score_value.explanation
        cite_str = '\n'.join([f'[{data["citation_id"]}] {data["content"]}' 
                             for data in ([] if metadata is None else metadata.get("citation_data", ""))])
        
        instruction = f"- The question was found online and should be rewritten. Here's what was found online: {explanation}"
        return [instruction]
    
    elif score_name == Metrics.SHORTCUTS.value:
        explanation = metadata['choices_only_response']
        instruction = f"- A test-taker could answer this multiple-choice question by ignoring the question stem and using shortcuts in the choices, without correctly guessing what the original question was. Here's how one test-taker did it: {explanation}"
        return [instruction]
    
    return []

@solver
def refine_dataset(refine_config: dict = None, base_config: dict = None, report_card_logs: list[dict] = None):
    async def solve(state: TaskState, generate: Generate) -> TaskState:

        state.metadata['old_question'] = state.user_prompt.text
        state.metadata['old_choices_list'] = [c.value for c in state.choices]
        state.metadata['old_choices'] = "\n".join([f"{chr(ord('A') + i)}) {c.value}" for i, c in enumerate(state.choices)])
        state.metadata['old_target'] = state.target.text

        # Use passed configs
        if refine_config is None or base_config is None:
            raise ValueError("refine_config and base_config must be provided")

        curr_mcq = MCQ(question=state.user_prompt.text, choices=[c.value for c in state.choices], answer=state.target.text)

        metric_list = base_config.get("refining_metrics", [])
        if not metric_list:
            raise ValueError("You must specify at least one metric")
        
        # Validate metrics list
        Metrics.validate_metrics_list(metric_list)

        seen_metrics = set()
        refinements = []
        for eval_log in report_card_logs:
            for eval_sample in eval_log.samples:
                if eval_sample.id != state.sample_id:
                    continue
                for score_name, score_value in eval_sample.scores.items():
                    seen_metrics.add(score_name)

                    if score_name not in metric_list or score_name == Metrics.DIFFICULTY.value:
                        continue

                    feedback = _get_metric_feedback(score_name, score_value, score_value.metadata)
                    refinements.append({
                        'score_name': score_name, 
                        'feedback': feedback, 
                        'refine_type': RefineType(refine_config.get(score_name).get('type')),
                        'should_refine': score_value.value < refine_config.get(score_name).get('cutoff')
                    })

        rewrite_experiment = RewriteExperiment.FLAW_CORRECTION

        # Experiments for difficulty (distractors or blooms)
        if Metrics.WRITING_FLAWS.value in metric_list and refine_config.get('difficulty').get('type') in {DifficultyRefineType.BLOOMS_TAXONOMY.value, DifficultyRefineType.ADD_DISTRACTORS.value}:
            rewrite_experiment = RewriteExperiment(refine_config.get('difficulty').get('type'))
        else:
            metric_diff = set(metric_list).difference(seen_metrics)
            if metric_diff != set() and metric_diff != {Metrics.DIFFICULTY.value}:
                print(f"Warning: You have specified metrics that were not evaluated. These metrics will be skipped: {metric_diff}")

        refined_mcq, explanation = await refine_mcq(curr_mcq, refinements, refine_config, rewrite_experiment)
        if refined_mcq == None:
            state.metadata['should_skip'] = True
            return state        

        # Check if this was a FEEDBACK-only refinement (no rewriting occurred)
        is_feedback_only = (refined_mcq.question == curr_mcq.question and 
                           refined_mcq.choices == curr_mcq.choices and 
                           refined_mcq.answer == curr_mcq.answer)

        if is_feedback_only:
            # For FEEDBACK-only refinements, keep the original question but log the feedback
            state.metadata['question'] = curr_mcq.question
            state.metadata['choices_list'] = [c for c in curr_mcq.choices]
            state.metadata['choices'] = "\n".join([f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(curr_mcq.choices)])
            state.user_prompt.text = curr_mcq.question + '\n' + state.metadata['choices']
            state.metadata['target'] = curr_mcq.answer
            state.metadata['should_skip'] = False  # Don't skip, we want to log the feedback
            state.metadata['explanation'] = explanation
            state.metadata['refinement_type'] = 'feedback_only'
        else:
            # Normal rewriting occurred
            state.metadata['question'] = refined_mcq.question
            state.metadata['choices_list'] = [c for c in refined_mcq.choices]
            state.metadata['choices'] = "\n".join([f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(refined_mcq.choices)])
            state.user_prompt.text = refined_mcq.question + '\n' + state.metadata['choices']
            state.metadata['target'] = refined_mcq.answer
            state.metadata['should_skip'] = (refined_mcq.question == curr_mcq.question and 
                                            refined_mcq.choices == curr_mcq.choices and 
                                            refined_mcq.answer == curr_mcq.answer)
            state.metadata['explanation'] = explanation
            state.metadata['refinement_type'] = 'rewrite'
        
        return state
    
    return solve