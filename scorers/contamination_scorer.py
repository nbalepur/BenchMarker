import json
from inspect_ai.model import get_model
from inspect_ai.scorer import Target
from inspect_ai.scorer import scorer, Score, mean, stderr
from inspect_ai.solver import TaskState

from model_utils.web_search import WebSearchType, create_web_searcher
from prompts.contamination_prompt import get_contamination_prompt

def get_citation_data(question, answer, search_type=WebSearchType.GOOGLE, max_results=5, max_tokens_per_page=512, try_scraping=False):
    """
    Helper function to get citation data for contamination detection.
    
    Args:
        question: The question text
        answer: The correct answer
        search_type: Type of search engine to use
        max_results: Maximum number of search results
        max_tokens_per_page: Maximum tokens per page
        try_scraping: Whether to try web scraping
        
    Returns:
        Formatted citations string for use in contamination prompts
    """
    # Create web searcher
    web_searcher = create_web_searcher(
        search_type=search_type,
        try_scraping=try_scraping
    )
    
    # Create search query similar to contamination scorer
    query = f'"{question}" "{answer}"'
    if search_type == WebSearchType.BRAVE and len(question.split() + answer.split()) > 50:
        # Brave has a 50 word query limit
        query = ' '.join(question.split()[:45])
        query = f'"{query}"'
    
    # Perform web search
    try:
        search_results = web_searcher.search(
            query, 
            max_results=max_results, 
            max_tokens_per_page=max_tokens_per_page
        )
        
        # Format citations
        if search_results:
            citations = '\n'.join([result.to_citation(idx+1) for idx, result in enumerate(search_results)])
        else:
            citations = "No search results found."
            
    except Exception as e:
        print(f"Error performing web search: {e}")
        citations = "Error occurred during search."
    
    return citations, search_results

@scorer(name="contamination", metrics=[mean(), stderr()])
def contamination_scorer(model: str = None, use_llm: bool = True, search_type: WebSearchType = WebSearchType.GOOGLE, max_results: int = 5, max_tokens_per_page: int = 512, try_scraping: bool = False, attempts: int = 3, sample_to_score: dict = None):

    async def _score(state: TaskState, target: Target) -> Score:
        # If cached results are available and MCQ was not refined, use them
        if (sample_to_score and state.sample_id in sample_to_score and 'contamination' in sample_to_score[state.sample_id]):
            cached_data = sample_to_score[state.sample_id]['contamination']
            metadata = cached_data.get('metadata', {})
            metadata['cached'] = True
            return Score(
                value=cached_data['value'],
                answer=cached_data['answer'],
                explanation=cached_data.get('explanation', ''),
                metadata=metadata
            )
        
        correct_answer = state.metadata['choices_list'][ord(target.target[0]) - ord('A')]
        question = state.metadata['question']
        
        # Use helper function to get citation data
        citations, search_results = get_citation_data(
            question=question,
            answer=correct_answer,
            search_type=search_type,
            max_results=max_results,
            max_tokens_per_page=max_tokens_per_page,
            try_scraping=try_scraping
        )
        
        if not use_llm:
            # For non-LLM mode, we need to check if citations contain actual results
            has_results = citations != "No search results found." and citations != "Error occurred during search."
            return Score(value=int(not has_results), # if no results, not contaminated
                         answer='',
                         explanation='',
                         metadata={'citations': citations, 'cached': False})

        if len(search_results) == 0 or citations == "No search results found.":
            return Score(value=1.0, explanation='No search results found', answer='no_match', metadata={'cached': False})

        contamination_prompt = get_contamination_prompt().format(citations=citations, question=question, correct_answer=correct_answer)

        judge_model = get_model(model)
    
        formatted_output = {'result': 'no_match', 'explanation': 'The LLM failed to parse the output', 'citations': []}
        for _ in range(attempts): # try 3 times
            try:
                output = await judge_model.generate(contamination_prompt)
                curr_output = output.completion.replace('`', '').replace('json', '')
                curr_output = json.loads(curr_output[curr_output.index("{"):curr_output.rindex("}")+1])
                if ('result' in curr_output and curr_output['result'] in {'exact_match', 'question_match', 'partial_match', 'no_match'}) and 'explanation' in curr_output and (curr_output['result'] == 'no_match' or ('citations' in curr_output and type(curr_output['citations']) == list)):
                    formatted_output = curr_output
                    break
            except Exception as _:
                continue

        all_search_data = [{
            'result_id': citation_id,
            'content': search_results[citation_id].content,
            'metadata': search_results[citation_id].metadata,
        } for citation_id in range(len(search_results))]
        metadata = {'all_search_data': all_search_data}

        if formatted_output['result'] != 'no_match':
            citation_data = [{
                'citation_id': citation_id,
                'content': search_results[citation_id-1].content,
                'metadata': search_results[citation_id-1].metadata,
            } for citation_id in formatted_output.get('citations', []) if 0 <= citation_id-1 < len(search_results)]
            metadata['citation_data'] = citation_data

        metadata['cached'] = False
        return Score(value=int(formatted_output['result'] not in {'exact_match', 'question_match'}),
                     answer=formatted_output['result'],
                     explanation=formatted_output.get('explanation', ''),
                     metadata=metadata)

    return _score
