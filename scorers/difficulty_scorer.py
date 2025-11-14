import copy
import string
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import parse_answers, set_choices_based_on_generated_response
from inspect_ai.model import get_model
from inspect_ai.scorer import scorer, Score, mean, stderr, choice
from prompts.run_mcqa_prompts import SINGLE_ANSWER_TEMPLATE_COT
from inspect_ai.scorer._metric import CORRECT

def get_accuracy_scorer_with_name(name: str):

    @scorer(name=name, metrics=[mean(), stderr()])
    def accuracy_scorer(model: str = None, sample_to_score: dict = None):

        async def _score(state: TaskState, target: Target) -> Score:
            # If prior logs are provided and MCQ was not refined, use cached scores
            if sample_to_score and state.metadata.get('should_skip', False):
                score_data = sample_to_score[state.sample_id]['accuracy'][name[len('accuracy_'):]]
                return Score(
                    value=score_data['score'],
                    answer=score_data['answer'],
                    explanation=score_data['explanation'],
                    metadata={'cached': True}
                )
            
            # Otherwise, generate new model outputs
            mcqa_model = get_model(model)
            template = SINGLE_ANSWER_TEMPLATE_COT
            score_prompt = template.format(
                question=state.metadata['question'],
                choices=state.metadata['choices'],
                letters=",".join(string.ascii_uppercase[:len(state.choices)]),
            )

            try:
                output = await mcqa_model.generate(score_prompt, config={'max_retries': 3})
                
                # Create a copy of the state with the model's output
                temp_state = TaskState(
                    model=state.model,
                    sample_id=state.sample_id,
                    epoch=state.epoch,
                    input=state.input,
                    messages=state.messages,
                    choices=copy.deepcopy(state.choices),
                    output=output,
                    metadata=state.metadata
                )
                answers = parse_answers(temp_state, multiple_correct=False)
                set_choices_based_on_generated_response(temp_state, answers)
                score = await choice()(temp_state, target)
                return Score(
                    value=int(score.value == CORRECT),
                    answer=score.answer,
                    explanation=score.explanation,
                    metadata={'name': name, 'cached': False}
                )
            except Exception as e:
                return Score(
                    value=0,
                    answer='ERROR',
                    explanation=str(e),
                    metadata={'cached': False}
                )

        return _score

    return accuracy_scorer

@scorer(name="avg_accuracy", metrics=[mean(), stderr()])
def avg_accuracy_scorer(sample_to_score: dict = None):
    """Scorer that returns the average accuracy of the model."""
    async def _score(state: TaskState, target: Target) -> Score:
        if not sample_to_score:
            return Score(value=0.0, explanation="No accuracy values provided", metadata={'cached': False})
        
        # Only use cached results if MCQ was not refined
        avg_accuracy = 1.0 * sum([s['score'] for s in sample_to_score[state.sample_id]['accuracy'].values()]) / len(sample_to_score[state.sample_id]['accuracy'])
        return Score(
            value=avg_accuracy,
            metadata={
                'cached': True, 
                'model_accuracy': [{
                    'name': metric_name[len('accuracy_'):],
                    'accuracy': score_data['score'],
                    'answer': score_data['answer'],
                    'explanation': score_data['explanation'],
                } for metric_name, score_data in sample_to_score[state.sample_id]['accuracy'].items()]
            }
        )

    return _score

@scorer(name="diff", metrics=[mean(), stderr()])
def difficulty_scorer(sample_to_score: dict = None):
    """Scorer that returns IRT difficulty scores for each question."""
    
    async def _score(state: TaskState, target: Target) -> Score:
        
        if not sample_to_score:
            return Score(value=0.0, explanation="No IRT values provided", metadata={'cached': False})
        
        # Use cached results if available (sample_to_score only contains valid cached results)
        if state.sample_id in sample_to_score and 'difficulty' in sample_to_score[state.sample_id]:
            return Score(
                value=sample_to_score[state.sample_id]['difficulty'],
                explanation="",
                metadata={'cached': True}
            )
        else:
            return Score(value=0.0, explanation="No cached difficulty available", metadata={'cached': False})
    
    return _score


@scorer(name="disc", metrics=[mean(), stderr()])
def discriminability_scorer(sample_to_score: dict = None):
    """Scorer that returns IRT discriminability scores for each question."""
    
    async def _score(state: TaskState, target: Target) -> Score:
        if not sample_to_score:
            return Score(value=0.0, explanation="No IRT values provided", metadata={'cached': False})
        
        # Use cached results if available (sample_to_score only contains valid cached results)
        if state.sample_id in sample_to_score and 'discriminability' in sample_to_score[state.sample_id]:
            return Score(
                value=sample_to_score[state.sample_id]['discriminability'],
                explanation="",
                metadata={'cached': True}
            )
        else:
            return Score(value=0.0, explanation="No cached discriminability available", metadata={'cached': False})
    
    return _score

