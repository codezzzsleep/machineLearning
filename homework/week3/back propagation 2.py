import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

a_list = []
b_list = []
c_list = []
loss_list = []
a = torch.tensor([1.0])
a.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True
c = torch.tensor([1.0])
c.requires_grad = True
rounds = 100000


def forward(x):
    return a * x * x + b * x + c


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2


print('predict:', forward(4).data)

for epoch in range(rounds):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        a.data = a.data - 0.01 * a.grad.data
        a.grad.data.zero_()
        b.data = b.data - 0.01 * b.grad.data
        b.grad.data.zero_()
        c.data = c.data - 0.01 * c.grad.data
        c.grad.data.zero_()
        a_list.append(a.data)
        b_list.append(b.data)
        c_list.append(c.data)
        loss_list.append(l.item())
    print('\t grad', x, y, l.item())
print('predict', forward(4).item())

epoch_list = np.arange(1, 3 * rounds + 1).tolist()
plt.plot(epoch_list, a_list)
plt.xlabel('epoch')
plt.ylabel('a')
plt.title('epoch-a_back2')
plt.grid(True)
plt.savefig(str(rounds) + 'epoch-a_back2.png')
plt.show()
plt.plot(epoch_list, b_list)
plt.xlabel('epoch')
plt.ylabel('b')
plt.title('epoch-b_back2')
plt.grid(True)
plt.savefig(str(rounds) + 'epoch-b_back2.png')
plt.show()
plt.plot(epoch_list, c_list)
plt.xlabel('epoch')
plt.ylabel('c')
plt.title('epoch-c_back2')
plt.grid(True)
plt.savefig(str(rounds) + 'epoch-c_back2.png')
plt.show()
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_back2')
plt.grid(True)
plt.savefig(str(rounds) + 'epoch-Loss_back2.png')
plt.show()
