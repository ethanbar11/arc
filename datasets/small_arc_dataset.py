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
        test = tasks_data['test']
        return self.data.iloc[idx], train, test

    def get_by_task_id(self, task_id):
        series = self.data[self.data['Task_ID'] == task_id]
        tasks_data = json.loads(series['Task_json'][0])
        train = tasks_data['train']
        test = tasks_data['test']
        return None, train, test


