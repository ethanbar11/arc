import os

from config import config
from datasets.small_arc_dataset import SmallArcDirectGridDataset
from prompting_classes.common import register_class
from prompting_classes.zero_shot_prompt_convertor import ZeroShotPromptConvertor


@register_class
class CoTPromptConvertor(ZeroShotPromptConvertor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = SmallArcDirectGridDataset(config['dataset_location'])
        self.tasks = config.prompt_config.cot_tasks

    def get_task_as_cot_prompt(self, task_id):
        location = os.path.join(config.prompt_config.cot_prompt_location, f'{task_id}.txt')
        data, trains, tests = self.dataset.get_by_task_id(task_id)
        prompt_allged_input = ZeroShotPromptConvertor.convert_task_to_prompt(self, trains, tests)
        with open(location, 'r') as f:
            alleged_output = f.read()
        for idx in range(len(self.to_prompt_dict)):
            alleged_output = alleged_output.replace('{' + f'{str(idx)}' + '}', self.to_prompt_dict[idx])
        out_as_text = self.convert_mat_to_text(tests[0]['output'])
        alleged_output = alleged_output.replace('{out}', out_as_text)
        return prompt_allged_input + alleged_output

    def convert_task_to_prompt(self, train, test):
        cot_prompts = []
        for task in self.tasks:
            cot_prompts.append(self.get_task_as_cot_prompt(task))
        total_cot = ''
        for idx, prompt in enumerate(cot_prompts):
            total_cot += cot_prompts[idx] + '\n'
        start_prompt = self.prompt_start  # .format(len(self.train))
        training_prompt = "\n"
        for idx, example in enumerate(train):
            inp = self.convert_mat_to_text(example['input'])
            out = self.convert_mat_to_text(example['output'])
            # training_prompt += numbers_to_letters[idx] + ".\n"
            training_prompt += f"input {idx}:\n{inp}\noutput {idx}:\n{out}\n\n"
        test_prompt = self.prompt_test + '\n'
        test_prompt += f"input:\n{self.convert_mat_to_text(test[0]['input'])}"

        return total_cot + start_prompt + training_prompt + test_prompt + self.prompt_end
