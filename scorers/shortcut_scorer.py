import json
import string
from inspect_ai.model import get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.scorer import scorer, Score, mean, stderr

from prompts.shortcut_prompts import SINGLE_ANSWER_TEMPLATE_NO_QUESTION, QUESTION_DETECTION_PROMPT


@scorer(name="shortcuts", metrics=[mean(), stderr()])
def shortcut_scorer(model: str = None, num_attempts: int = 3, sample_to_score: dict = None):

    async def _score(state: TaskState, target: Target) -> Score:
        # If cached results are available and MCQ was not refined, use them
        if (sample_to_score and state.sample_id in sample_to_score and 'shortcuts' in sample_to_score[state.sample_id]):
            cached_data = sample_to_score[state.sample_id]['shortcuts']
            metadata = cached_data.get('metadata', {})
            metadata['cached'] = True
            return Score(
                value=cached_data['value'],
                answer=cached_data['answer'],
                explanation=cached_data.get('explanation', ''),
                metadata=metadata
            )

        mcqa_model = get_model(model)
        
        curr_choices = state.metadata['choices_list']
        score_prompt = SINGLE_ANSWER_TEMPLATE_NO_QUESTION.format(
            choices="\n".join([f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(curr_choices)]).strip(),
            letters=",".join(string.ascii_uppercase[:len(curr_choices)]),
        )

        answer, inferred_question, choices_only_explanation = None, None, 'The model failed to produce a valid answer'
        for _ in range(num_attempts):
            try:
                output = await mcqa_model.generate(score_prompt)
                output = output.completion.replace('`', '').replace('json', '')
                if '{' in output and '}' in output:
                    output = output[output.index('{'):output.rindex('}')+1]
                output = json.loads(output)
                if output.get('answer', '') not in ",".join(string.ascii_uppercase[:len(curr_choices)]) or output.get('explanation', '') == '' or output.get('question', '') == '':
                    continue
                answer, inferred_question, choices_only_explanation = output['answer'], output['question'], output['explanation']
                break

            except Exception as e:
                print("Error in shortcut scorer - choices-only accuracy:", str(e))
                continue
        
        if answer == None or answer.replace('(', '').replace(')', '').strip() != state.metadata['target']:
            return Score(
                value=1,
                answer=answer,
                explanation='The model did not answer the question correctly with just the choices',
                metadata={'cached': False, 'choices_only_success': 0, 'choices_only_response': choices_only_explanation, 'inferred_question': inferred_question}
            )

        detection_prompt = QUESTION_DETECTION_PROMPT.format(question=state.metadata['question'], response=choices_only_explanation, inferred_question=inferred_question)
        decision, explanation = 'failed', 'The model failed to produce a valid explanation'
        #print(detection_prompt)
        for _ in range(num_attempts):
            try:
                output = await mcqa_model.generate(detection_prompt)
                output = output.completion.replace('`', '').replace('json', '')
                if '{' in output and '}' in output:
                    output = output[output.index('{'):output.rindex('}')+1]
                output = json.loads(output)
                #print(output)
                if output.get('decision', '') not in {'exact_match', 'knowledge_match', 'no_match'} or output.get('explanation', '') == '':
                    continue
                decision, explanation = output['decision'], output['explanation']
                break

            except Exception as e:
                print("Error in shortcut scorer - question detection:", str(e))
                continue      

        return Score(
            value=int(decision not in {'no_match'}),
            answer=answer,
            explanation=explanation,
            metadata={'cached': False, 'choices_only_success': 1, 'choices_only_response': choices_only_explanation, 'inferred_question': inferred_question}
        )

    return _score

