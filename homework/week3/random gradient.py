import numpy as np
import matplotlib.pyplot as plt

w = 1.0
vector = 0.01

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_list = []
loss_list = []
rounds = 100


def forward(x):
    return x * w


def cost(xs, ys):
    mes = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        mes += (y - y_pred) ** 2
    return mes / len(xs)


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


def gradient_random(x, y):
    return 2 * x * (x * w - y)


for epoch in range(rounds):
    l = 0
    for x, y in zip(x_data, y_data):
        grad = gradient_random(x, y)
        w = w - vector * grad
        l += loss(x, y)
    w_list.append(w)
    loss_list.append(l / len(x_data))

epoch_list = np.arange(1, rounds + 1).tolist()
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('epoch-Loss_random')
plt.grid(True)
plt.savefig('epoch-Loss_random.png')
plt.show()

plt.plot(epoch_list, w_list)
plt.xlabel('epoch')
plt.ylabel('W')
plt.title('epoch-W')
plt.grid(True)
plt.savefig('epoch-W_random.png')
plt.show()
