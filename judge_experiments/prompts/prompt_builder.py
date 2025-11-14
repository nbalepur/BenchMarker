"""
Helper functions/class to retrieve prompts for a specific experiment. Prompts consist of two parts:
1) a prompt template (e.g. f-string)
2) a list of data inputs (keyword arguments) that can infill the prompt template
Combining (1) and (2), you get a list of prompts
"""

from judge_experiments.prompts.prompt_template_loader import PromptFactory, PromptType
from judge_experiments.prompts.prompt_data_loader import DataFetcherFactory
import datasets

class PromptBuilder:

    def __init__(self, args):
        self.args = args

    def build_dataset(self, inputs, outputs):
        combined_ds = datasets.Dataset.from_dict({'question': inputs, 'answer': outputs})
        combined_ds = combined_ds.map(
            lambda x: {
                "prompt": [
                    {"role": "user", "content": x["question"]},
                ],
                "ground_truth": x["answer"],
            }
        )
        return combined_ds

    def get_prompts(self, prompt_type: PromptType):

        # data inputs used to infill the prompt template
        data_factory = DataFetcherFactory()
        data_fetcher = data_factory.get_data_fetcher(prompt_type=prompt_type, args=self.args)

        prompt_data = data_fetcher.get_data()
        prompt_inputs, prompt_outputs = prompt_data['input'], prompt_data['output']
        
        # Data fetchers only provide 'test' split, not train/val
        test_inputs = prompt_inputs['test']
        test_outputs = prompt_outputs['test']

        # template for the prompt
        prompt_factory = PromptFactory(args=self.args)
        prompt_parser = prompt_factory.get_prompt(prompt_type)

        # construct prompts - only test split available
        prompts = []
        for i, input_data in enumerate(test_inputs):
            if input_data is not None:
                # Add the answer to the input data for prompt creation
                input_data['answer'] = test_outputs[i]
                prompt = prompt_parser.create_prompt(**input_data)
                prompts.append(prompt)
        
        return prompts