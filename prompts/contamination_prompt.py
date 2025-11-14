CONTAMINATION_PROMPT = """You are an expert evaluator of dataset contamination for multiple-choice questions (MCQs).

You are given a set of candidate source texts ("citations") and one MCQ. Your job is to determine whether the MCQ exists eactly, partially, or not at all in any of the citations.

<multiple-choice question>
Question: {question}
Correct Answer: {correct_answer}
</multiple-choice question>

<citations>
{citations}
</citations>

Use the following criteria to determine the match type:
<matching criteria>
- "exact_match": The question and correct answer appear verbatim or nearly verbatim in at least one of the citations 
- "question_match": The question appears verbatim or nearly verbatim in at least one of the citations, but not with the correct answer
- "partial_match": It is possible to come up with the correct answer to the question based on information in the citations
- "no_match": There is no information in the citations that can be used to answer the question
</matching criteria>

<general instructions>
- Use ONLY the information in the provided <citations>; ignore outside knowledge.
- When determining matches, do NOT consider punctuation or upper/lower casing.
- Check each citation independently.
- The citations index "i" is represented as <citation i></citation i>.
- Return every matching citation index in ascending order; if none match, return an empty list [].
- Provide a short, clear explanation for your decision, referencing the decisive overlaps when applicable.
</general instructions>

<format>
Return your output as valid JSON with the matching "result", the indexed "citations" that support your decision (empty list [] if "no_match"), and an "explanation" for your decision:
{{
  "result": "exact_match" | "question_match" | "partial_match" | "no_match"
  "citations": [insert ascending list of citations that contain any matches],
  "explanation": "Your explanation here"
}}
Do not include anything else.
</format>
""".strip()

def get_contamination_prompt() -> str:
    return CONTAMINATION_PROMPT