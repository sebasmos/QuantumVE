
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset

class EmbeddingDataset_stateless(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, -1].values 
        self.features = self.data.iloc[:, :-1].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.features[idx].astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

class EmbeddingDataset(Dataset):
    def __init__(self, csv_file, shuffle=False):
        self.data = pd.read_csv(csv_file)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
        
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label