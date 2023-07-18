import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


loss_list = []
rounds = 100
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = MyModel()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(rounds):
        for i, data in enumerate(train_loader, 0):
            # print(i)
            # print(data)
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss_list.append(loss.item())
            print(epoch, ":", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epoch_list = np.arange(1, len(loss_list) + 1).tolist()

    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('epoch-Loss_logistic')
    plt.grid(True)
    plt.savefig('photo/' + str(rounds) + ' epoch-Loss_diabetes_dataloader.png')
    plt.show()
