import torch


class Net(torch.nn.Module):
    def __int__(self):
        super(Net, self).__int__()
        self.cov1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.cov2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.relu(self.pooling(self.cov1(x)))
        x = torch.relu(self.pooling(self.cov2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
