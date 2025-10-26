import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class CoraDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        os.makedirs(root, exist_ok=True)

        with open("cora.content", 'r') as f:
            lines = f.readlines()

        features = []
        labels = []
        for line in lines:
            items = line.strip().split('\t')
            features.append(list(map(float, items[1:-1])))
            labels.append(items[-1])

        self.data = pd.array(features, dtype=pd.float32)
        self.labels = pd.array(labels)

            

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self.data)