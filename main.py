import tqdm
import pandas as pd
import timeit
from datasets.small_arc_dataset import SmallArcDirectGridDataset
import yaml
import os
from pipelines.arcathon_pipeline import ArcathonPipeline
from config import config
from prompting_classes import prompt_cls


def save_to_csv(csv_path, data, success, complete_out):
    experiment_name = csv_path.split('/')[-1].split('.')[0]
    data = data.copy()
    data[f'{experiment_name} success'] = success
    data[f'{experiment_name} complete output'] = complete_out
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=data.index)
    # Append row to the dataframe
    df = pd.concat([df, pd.DataFrame([data])])
    df.to_csv(csv_path)


if __name__ == '__main__':
    print(f'Before starting: output file location is {config["output_path"]}')
    dataset = SmallArcDirectGridDataset(config['dataset_location'])
    print('Running with the following config:')
    print(yaml.dump(config, default_flow_style=False))
    prompt_converter = prompt_cls(**config.prompt_config)
    pipeline = ArcathonPipeline(prompt_converter, **config.arcathon_pipeline)

    idx = 1
    success_amount = 0
    for data, trains, tests in tqdm.tqdm(dataset):
        print('Starting to work on', data['Task_ID'])
        success, saved_outputs = pipeline(trains, tests)
        if success:
            success_amount += 1
        print(
            f'Finished working on {data["Task_ID"]}, success: {success}, in index {idx}/{len(dataset)} and so far {success_amount} successes.')
        save_to_csv(config['output_path'], data, success, saved_outputs)
        idx += 1
