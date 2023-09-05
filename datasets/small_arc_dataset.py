from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json


class SmallArcDirectGridDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tasks_data = json.loads(self.data.loc[idx, 'Task_json'])
        train = tasks_data['train']
        # for example in train:
        #     input,output = example['input'],example['output']
        test = tasks_data['test']

        return self.data.iloc[idx], train, test
