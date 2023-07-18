import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('../../res/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
loss_list = []
rounds = 10000


class MyModel(torch.nn.Module):
    def __init__(self):
        # RuntimeError: all elements of input should be between 0 and 1

        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Hardsigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = MyModel()
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(rounds):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, ': ', loss.item())
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
epoch_list = np.arange(1, rounds + 1).tolist()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_logistic')
plt.grid(True)
plt.savefig('photo/Hardsigmoid/' + str(rounds) + ' epoch-Loss_diabetes.png')
plt.show()
