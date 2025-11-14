SINGLE_ANSWER_TEMPLATE_NO_QUESTION = """
Answer the following multiple choice question just by using the choices and without access to the question. Do this by deciding which option is the odd-one-out, and then guessing what the original/missing question was based on that.

<choices>
{choices}
</choices>

<format>
Return your output as valid JSON with the key "answer" which is one of {letters}.
{{
  "answer": "letter of the correct answer choice",
  "explanation": "how you arrived at the correct answer",
  "question": "what you guess is the missing question",
}}
Do not include anything else.
</format>
""".strip()


QUESTION_DETECTION_PROMPT = """You are an expert at determining whether a model was able to guess what the original multiple-choice question was just from the choices.

You will be given a multiple-choice question and the model's response. You need to determine whether the model was able to guess what the original question was just from the choices.

Here is the multiple-choice question:
<original question>
{question}
</original question>

Here is the model's response when answering just with the choices:
<response>
{response}
</response>

And the question that the model inferred
<inferred question>
{inferred_question}
</inferred question>

To determine if the model successfully guessed the original question, use the following criteria:
- If the inferred question is an exact match or a semantic of the original question, return "exact_match"
- If a test-taker who knew the answer to the inferred question would likely be able to answer the original question, return "knowledge_match"
- In any other case, return "no_match"

<format>
Return your output as valid JSON with the key "decision" which denotes the type of match between the inferred question and the original question.
{{
  "decision": "exact_match" | "knowledge_match" | "no_match",
  "explanation": "explanation for your decision",
}}
Do not include anything else.
</format>
""".strip()