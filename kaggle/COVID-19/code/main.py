import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Matplotlib
import matplotlib.pyplot as plt

# Optuna
import optuna

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

datafile = '../../../dataset/COVID-19/covid_train.csv'

Xy = np.loadtxt(fname=datafile, delimiter=',', dtype=np.float32)
X = torch.from_numpy(Xy[:, :-1])
y = torch.from_numpy(Xy[:, [-1]])
X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len


train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear()
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.activate = torch.nn.ReLU()
