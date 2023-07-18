import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.sigmoid(self.l1(x))
        x = self.sigmoid(self.l2(x))
        x = self.sigmoid(self.l3(x))
        x = self.sigmoid(self.l4(x))
        x = self.sigmoid(self.l5(x))
        return x
