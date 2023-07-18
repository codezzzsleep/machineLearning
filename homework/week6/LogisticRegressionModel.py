import matplotlib.pyplot as plt
import torch.nn
# import torch.nn.functional as F
import numpy as np

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])
loss_list = []
rounds = 1000


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(model(torch.tensor([[4.0]])).item())

for epoch in range(rounds):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[4.0]])).item())

epoch_list = np.arange(1, rounds + 1).tolist()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_logistic')
plt.grid(True)
plt.savefig('epoch-Loss_logistic.png')
plt.show()
