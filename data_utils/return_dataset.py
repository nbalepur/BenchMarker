from inspect_ai.solver import TaskState
from inspect_ai.solver import Generate
from inspect_ai.solver import solver


@solver
def return_dataset():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata['question'] = state.user_prompt.text
        state.metadata['choices_list'] = [c.value for c in state.choices]
        state.metadata['choices'] = "\n".join([f"{chr(ord('A') + i)}) {c.value}" for i, c in enumerate(state.choices)])
        state.user_prompt.text = state.user_prompt.text + '\n' + state.metadata['choices']
        state.metadata['target'] = state.target.text
        return state
    return solve
