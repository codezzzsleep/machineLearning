import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, min(input_dim / 2, 64))
        self.linear2 = nn.Linear(min(input_dim / 2, 64), min(input_dim / 4, 16))
        self.linear3 = nn.Linear(min(min(input_dim / 4, 16)), min(input_dim / 4, 10))
        self.linear4 = nn.Linear(min(input_dim / 4, 10), 1)
        self.sigmoid = nn.Sigmoid()
        self.activate = nn.ReLU()

    def forward(self, X):
        X = self.activate(self.linear1(X))
        X = self.activate(self.linear2(X))
        X = self.activate(self.linear3(X))
        X = self.sigmoid(self.linear4(X))
        return X
