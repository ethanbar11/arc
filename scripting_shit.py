from config import config
from datasets.small_arc_dataset import SmallArcDirectGridDataset
from prompting_classes import prompt_cls

if __name__ == '__main__':
    prompter = prompt_cls(using_numbers=True)
    dataset = SmallArcDirectGridDataset(config['dataset_location'])
    for idx in range(1,2):
        data, trains, tests = dataset[idx]
        print(f'Task ID: {data["Task_ID"]}')
        prompt = prompter.convert_task_to_prompt(trains, tests)
        correct_output = prompter.convert_mat_to_text(tests[0]['output'])
        print(f'{idx}: \n\n')
        print('Prompt: \n')
        print(prompt)
        print('Correct output:')
        print(correct_output,'\n\n')
