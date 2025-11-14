EXPLANATION_TO_INSTRUCTIONS = """You are an expert at decomposing feedback into a self-refinement instructions.

Given a multiple-choice question and feedback on how the multiple-choice question can be improved, your goal is to decompose the feedback into a series of instructions that the user can follow to improve the multiple-choice question.

Here is the multiple-choice question:
<multiple-choice question>
{mcq}
</multiple-choice question>

Here is the feedback on how the multiple-choice question can be improved:
<feedback>
{feedback}
</feedback>

<feedback description>
Here are the types of feedback that can be given, as well as guidelines on how to responsd to each type of feedback.
- Writing Flaws: Issues in the grammar, structure, or style of the multiple-choice question. The feedback here should lead to writing improvements the user can make. For example, if the question is grammatically incorrect, the instruction could be "Modify choice "X" so it is a noun and gramatically consistent with the question stem which asks for a noun". If you are suggesting more plausible distractors, they must not be the correct answer and should be unambiguously incorrect.
- Contamination: A question stem either appearing partially or verbatim on the Internet. If there is an exact match in the question, the instruction should be to perturb the question slightly so it will likely no longer appear online. If all of the choices are also present in the citations, the instruction should be to minimally modify the choices. For example, the instruction could be "Minimally modify the wording of the question stem since it was found exactly online..."
- Shortcuts: Models are able to answer the question without reading the question and just by looking at the choices. The model will give an explanation as to how it was able to answer the question, by finding the "odd-one-out" and inferring what the original question was. There are two options for rewriting: 1) rewrite the correct answer such that it no longer applies to the distractor choice; and 2) rewrite a distractor choice such that it now applies to the question stem the model inferred, but not the original question stem. As an example for (1), if the choices are "(A) car engine (B) refrigerator (C) frying pan (D) glass bottle" with the question "Which of these is an exothermic reaction?" and the inferred question "Which of these is a non-kitchen item?", you should instruct the model to rewrite the correct answer to make it a kitchen item with a exothermic reaction (e.g. "heating a gas stove"). As an example for (2), you should instruct the model to rewrite one of the distractors and make it a non-kitchen item as well (e.g. Change "glass bottle" to "glass window"). You should decide between (1) and (2) based on which you could do without making the multiple-choicequestion impossible to answer.
</feedback description>

<important rules>
- All of your instructions, if followed, should lead to a high-quality, improved multiple-choice question with a clear question stem, an unambiguous correct answer, and plausible but ultimately incorrect distractor choices.
- Never instruct the user to modify the order of the choices
- Never instruct the user to modify which letter is the correct answer to the question
- Never instruct the user to largely rewrite the entire multiple-choice question. Total rewriting should only happen in rare cases, like if there is a very obvious shortcut in the choices, as the correct answer would need to change.
- The instructions should not lead to a question with a different correct answer.
- It is best to instruct the model to change as little as possible. For example, it's preferred to just change one answer choice or phrases in the question, rather than rewriting the whole item.
</important rules>

<instructions format>
- Instructions should be two sentences. The first sentence should explain the issue in the question according to the feedback. The second sentence should explain how to modify the question given this feedback and how that would fix the issue.
- Instructions should be clear and easy to understand by a user who has not seen the feedback
- Instructions should be specific to the feedback and not general
- Instructions should give specific, actionable suggestions that the user can take to improve the question. These could include specific words to rephrase or choices to change.
- There does not need to be a one-to-one mapping between feedback and instructions. You can have multiple instructions for the same feedback or a single instruction for multiple feedbacks.
- Do not add instructions unrelated to the feedback
</instructions format>

<output format>
Return your output as valid JSON with a single key "instructions" that is a list of strings:
{{
  "instructions": [
    "instruction 1",
    "instruction 2",
    "instruction 3"
  ]
}}
Do not include anything else.
</output format>
"""

REWRITE_PROMPT = """You are an expert at rewriting multiple-choice questions based on instructions.

Given a multiple-choice question and instructions on how to improve the multiple-choice question, your goal is to rewrite the multiple-choice question according to the instructions.

Here is the multiple-choice question:
<multiple-choice question>
{mcq}
</multiple-choice question>

Here are the instructions on how to improve the multiple-choice question:
<instructions>
{instructions}
</instructions>

<question format>
- You should return a high-quality, improved multiple-choice question with a clear question stem, an unambiguous correct answer, and plausible but ultimately incorrect distractor choices.
- You should not modify the order of the choices
- You should not add or remove any choices. The number of choices should be the same as the original question.
- You should not modify which letter is the correct answer to the question
- You should not largely rewrite the entire multiple-choice question. Total rewriting should only happen in rare cases.
- It is best to change as little as possible and keep parts of the original question **VERBATIM**. For example, it's preferred to just change one answer choice or phrases in the question, rather than rewriting the whole item.
- Similarly, do not change the style of the modified question or choices unless it is required by one of the instructions, such as length, tone, word choice, etc.
</question format>

<output format>
Return your output as valid JSON with three keys: "question" (the rewritten question stem), "choices" (a list of strings representing the rewritten choices), and "answer" (the letter of the correct answer).
{{
  "question": "rewritten question stem",
  "choices": ["rewritten choice 1", "rewritten choice 2", "rewritten choice 3", ...],
  "answer": "letter of the correct answer",
  "explanation": "explanation of your changes"
}}
Do not include anything else.
</output format>
"""