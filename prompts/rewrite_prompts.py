REWRITE_PROMPT = """You are an expert at rewriting multiple-choice questions to correct flaws in the item.

Given a multiple-choice question and issues with the item, your goal is to rewrite the multiple-choice question to correct the issues.

Here is the multiple-choice question, choices, and correct answer:
<multiple-choice question>
{mcq}
</multiple-choice question>

Here are the flaws in the multiple-choice question, which are issues in the grammar, structure, or style of the item. The item should be rewritten so that these issues are corrected:
<flaws>
{feedback}
</flaws>

<rewriting guidelines>
- The rewritten multiple-choice question should have a clear question stem, an unambiguous correct answer, and plausible but ultimately incorrect distractor choices.
- Generate each choice directly, do not precede them with a letter, number, or symbol.
- Only change what is needed to correct the flaws in the multiple-choice question. For example, if the question has a grammatical error in the question stem, only change the question stem, not the choices.
- In your explanation, state the flaws that the original question had and how you corrected them in the rewritten question.
</rewriting guidelines>

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

BLOOMS_TAXONOMY_PROMPT = """You are an expert at rewriting multiple-choice questions to increase the difficulty of the question.

Given a multiple-choice question, your goal is to make the question more difficult by targeting a higher level of cognitive skill according to Bloom's Taxonomy. You should increase the difficulty by {num_blooms_levels} level(s) of Bloom's Taxonomy.

Here are the levels of Bloom's Taxonomy:
<bloom's taxonomy>
1. Remember: Recall facts, terms, definitions, or basic concepts.  
2. Understand: Explain ideas or concepts in your own words; summarize, interpret, or classify information.  
3. Apply: Use knowledge or procedures in new but familiar situations.  
4. Analyze: Break information into parts to explore relationships, patterns, causes, or underlying structure.  
5. Evaluate: Make judgments based on criteria or standards; justify decisions or compare alternatives.  
6. Create: Generate new ideas, solutions, or products by combining elements in novel ways.
</bloom's taxonomy>

Here is the multiple-choice question, choices, and correct answer:
<multiple-choice question>
{mcq}
</multiple-choice question>

Here are the flaws in the multiple-choice question:
<rewriting guidelines>
- The rewritten multiple-choice question should have a clear question stem, an unambiguous correct answer, and plausible but ultimately incorrect distractor choices.
- Generate each choice directly, do not precede them with a letter, number, or symbol.
- Only change the question stem, the choices, and the answer to make the question more difficult to answer.
- Do not add or remove any choices.
- Increase the skill level of the question by exactly{num_blooms_levels} level(s) of Bloom's Taxonomy.
- The domain and concepts being tested in the multiple-choice question should be the same as the original question.
- Increase difficulty specifically by requiring higher-order cognitive skills according to Bloom's Taxonomy, not by making wording more confusing.
- In your explanation, state the level of Bloom's Taxonomy that the original question is targeting and how you increased this level in the rewritten question.
</rewriting guidelines>

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

ADD_DISTRACTORS_PROMPT = """You are an expert at rewriting multiple-choice questions to add distractors to the question.

Given a multiple-choice question, your goal is to rewrite the multiple-choice question to add {num_distractors} new distractor(s) to the question, targetting common misconceptions and making the question more difficult to answer.

Here is the multiple-choice question, choices, and correct answer:
<multiple-choice question>
{mcq}
</multiple-choice question>

Here are the flaws in the multiple-choice question:
<rewriting guidelines>
- The rewritten multiple-choice question should have a clear question stem, an unambiguous correct answer, and plausible but ultimately incorrect distractor choices.
- Generate each choice directly, do not precede them with a letter, number, or symbol.
- Return the original distractors plus exactly {num_distractors} new distractor(s).
- The question stem and correct answer should be the same as the original question.
- The initial distractors in the original question should be kept in the rewritten question.
- The new distractors should be common misconceptions or mistakes that a test-taker would make when answering the question.
- All of the new distractors should be distinct from the original distractors, or else test-takers would be able to rule them out by process of elimination.
- In your explanation, state how the new distractors target common misconceptions and make the question more difficult to answer.
</rewriting guidelines>

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