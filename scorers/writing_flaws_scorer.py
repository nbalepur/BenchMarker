import json
from inspect_ai.model import get_model
from inspect_ai.scorer import Target
from inspect_ai.scorer import scorer, Score, mean, stderr
from inspect_ai.solver import TaskState

from prompts.writing_flaw_prompts import get_writing_flaw_prompts, OPTIMAL_WRITING_FLAW_JUDGES

@scorer(name="writing_flaws", metrics=[mean(), stderr()])
def writing_flaws_scorer(model: str = None, attempts: int = 3, sample_to_score: dict = None):

    async def _score(state: TaskState, target: Target) -> Score:
        # If cached results are available and MCQ was not refined, use them
        if (sample_to_score and state.sample_id in sample_to_score and 'writing_flaws' in sample_to_score[state.sample_id]):
            cached_data = sample_to_score[state.sample_id]['writing_flaws']
            metadata = cached_data.get('metadata', {})
            metadata['cached'] = True
            return Score(
                value=cached_data['value'],
                answer=cached_data['answer'],
                explanation=cached_data.get('explanation', ''),
                metadata=metadata
            )
    
        writing_flaw_prompts = get_writing_flaw_prompts()

        metadata = {
            'writing_flaws': [],
            'acceptable': []
        }

        total_score = 0.0
        all_answers = []
        for flaw_name, curr_prompt in writing_flaw_prompts:
            format_params = {'lbrace': "{", 'rbrace': "}"}
            if '{question}' in curr_prompt:
                format_params['question'] = state.metadata['question']
            if '{choices}' in curr_prompt:
                format_params['choices'] = state.metadata['choices']
            if '{answer}' in curr_prompt:
                format_params['answer'] = state.metadata['target']
            curr_prompt = curr_prompt.format(**format_params)
            
            if model is None:
                print(f"Using optimal judge model for {flaw_name.value}: {OPTIMAL_WRITING_FLAW_JUDGES[flaw_name.value]}")
                judge_model = get_model(OPTIMAL_WRITING_FLAW_JUDGES[flaw_name.value])
            else:
                judge_model = get_model(model)

            formatted_output = {'result': 'pass', 'explanation': 'The LLM failed to parse the output'}
            for _ in range(attempts): # try 3 times
                try:
                    output = await judge_model.generate(curr_prompt)
                    curr_output = output.completion.replace('`', '').replace('json', '')
                    curr_output = json.loads(curr_output[curr_output.index("{"):curr_output.rindex("}")+1])
                    if ('result' in curr_output and curr_output['result'] in {'pass', 'fail'}) and 'explanation' in curr_output:
                        formatted_output = curr_output
                        break
                except Exception as e:
                    continue

            curr_metadata = {'name': flaw_name.value, 'score': formatted_output['result'], 'explanation': formatted_output['explanation'], 'confidence': formatted_output.get('confidence', 0)}
            total_score += float(formatted_output['result'] == 'pass')
            all_answers.append(formatted_output['result'])
            metadata['acceptable' if formatted_output['result'] == 'pass' else 'writing_flaws'].append(curr_metadata)

        metadata['cached'] = False
        return Score(value=total_score / len(writing_flaw_prompts), answer=str(all_answers), metadata=metadata)

    return _score