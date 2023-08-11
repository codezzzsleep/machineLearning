import torch


class MyNet(torch.nn.Module):
    def __init__(self, config, input_dim):
        super(MyNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, config['layer'][0])
        self.linear2 = torch.nn.Linear(config['layer'][0], config['layer'][1])
        self.linear3 = torch.nn.Linear(config['layer'][0], 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = x.squeeze(1)
        return x
