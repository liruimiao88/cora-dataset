import torch
from torch.utils.data import Dataset
import numpy as np
import os

class CoraDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        os.makedirs(root, exist_ok=True)
        
        if self.train:
            num_samples = 100
        else:
            num_samples = 20
            
        self.data = np.random.randn(num_samples, 100).astype(np.float32)
        self.labels = np.random.randint(0, 5, num_samples)
        print(f"Created dataset: {self.data.shape}")

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self.data)