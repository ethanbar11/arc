import tqdm
import pandas as pd
import timeit
from datasets.small_arc_dataset import SmallArcDirectGridDataset
import yaml
import os
from pipelines.arcathon_pipeline import ArcathonPipeline
from prompting_classes.zero_shot_prompt_convertor import ZeroShotPromptConvertor

CONFIG_LOCATION = './config/config.yaml'


def get_config():
    with open(CONFIG_LOCATION) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_to_csv(csv_path, data, success, matrix, complete_out):
    experiment_name = csv_path.split('/')[-1].split('.')[0]
    data = data.copy()
    data[f'{experiment_name} success'] = success
    data[f'{experiment_name} out matrix'] = matrix
    data[f'{experiment_name} complete output'] = complete_out
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=data.index)
    # Append row to the dataframe
    df = pd.concat([df, pd.DataFrame([data])])
    df.to_csv(csv_path)


if __name__ == '__main__':
    # SET cuda visible devices only to 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = get_config()
    dataset = SmallArcDirectGridDataset(config['dataset_location'])
    prompt_convertor = ZeroShotPromptConvertor(config['prompt_start_location'], delimiter=config['prompt_delimiter'])
    pipeline = ArcathonPipeline(prompt_convertor, config['model_id'], config['max_return_length'],config['temperature'])

    idx = 1
    for data, trains, tests in tqdm.tqdm(dataset):
        start = timeit.default_timer()
        success, matrix, complete_out = pipeline(trains, tests)
        end = timeit.default_timer()
        print(f'Finished working on {data["Task_ID"]}, success: {success}, {idx}/{len(dataset)} took {end - start}s')
        save_to_csv(config['output_path'], data, success, matrix, complete_out)
        idx += 1
