import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
loss_list = []

rounds = 1000



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
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
length = 1


def closure():
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    return loss


print(model(torch.tensor([[4.0]])).item())
for epoch in range(rounds):
    optimizer.step(closure=closure)

print(model(torch.tensor([[4.0]])).item())
epoch_list = np.arange(1, len(loss_list) + 1).tolist()
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_pytorch')
plt.grid(True)
plt.savefig('LBFGS/' + str(rounds) + ' epoch-Loss_pytorch_LBFGS优化器.png')
plt.show()
