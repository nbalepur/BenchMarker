SINGLE_ANSWER_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

SINGLE_ANSWER_TEMPLATE_COT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()