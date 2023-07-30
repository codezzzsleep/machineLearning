import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.X = torch.FloatTensor(X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
