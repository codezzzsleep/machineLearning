import numpy as np
import matplotlib.pyplot as plt

w = 1.0
vector = 0.01

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_list = []
mes_list = []
rounds = 100


def forward(x):
    return x * w


def cost(xs, ys):
    mes = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        mes += (y - y_pred) ** 2
    return mes / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


for epoch in range(rounds):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= vector * grad_val
    w_list.append(w)
    mes_list.append(cost_val)

epoch_list = np.arange(1, rounds + 1).tolist()
plt.plot(epoch_list, mes_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss')
plt.grid(True)
plt.savefig('epoch-Loss.png')
plt.show()

plt.plot(epoch_list, w_list)
plt.xlabel('epoch')
plt.ylabel('W')
plt.title('epoch-W')
plt.grid(True)
plt.savefig('epoch-W.png')
plt.show()
