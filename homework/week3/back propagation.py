import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True
w_list = []
loss_list = []
grad_list = []
l_item = []
epoch_list = np.arange(1, 300 + 1).tolist()


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2


print('predict', 4, forward(4).item())
for epoch in range(100):

    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w_list.append(w.data)
        grad_list.append(w.grad.item())
        loss_list.append(l.item())
        w.grad.data.zero_()

    print('progress:', epoch, l.item())
    l_item.append(l.item())

print('predict', 4, forward(4).item())

plt.plot(epoch_list, w_list)
plt.xlabel('epoch')
plt.ylabel('W')
plt.title('epoch-W_back')
plt.grid(True)
plt.savefig('epoch-W_back.png')
plt.show()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_back')
plt.grid(True)
plt.savefig('epoch-Loss_back.png')
plt.show()

plt.plot(epoch_list, grad_list)
plt.xlabel('epoch')
plt.ylabel('grad')
plt.title('epoch-grad_back')
plt.grid(True)
plt.savefig('epoch-grad_back.png')
plt.show()
arr_list = np.arange(1, 101).tolist()
plt.plot(arr_list, l_item)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('epoch-grad_back_l.item')
plt.grid(True)
plt.savefig('epoch-grad_back_l.item.png')
plt.show()
