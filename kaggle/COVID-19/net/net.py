import torch


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.linear1 = torch.nn.Linear()
        self.linear2 = torch.nn.Linear()
        self.linear3 = torch.nn.Linear()
        self.sigmoid = torch.nn.Sigmoid()
        self.activate = torch.nn.ReLU()

    def forward(self, X):
        X = self.activate(self.linear1(X))
        X = self.activate(self.linear2(X))
        X = self.sigmoid(self.linear3(X))
        return X
