import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
loss_list = []

rounds = 10000
epoch_list = np.arange(1, rounds+1).tolist()


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
# 损失函数方法更改
# 下面代码被替换为
# criterion = torch.nn.MSELoss(size_average=False)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_pytorch')
plt.grid(True)
plt.savefig('Adam/'+str(rounds) + ' epoch-Loss_pytorch_Adam优化器.png')
plt.show()
