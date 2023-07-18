import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) * (y - y_pred)


class Best:
    __w = 0
    __b = 0
    __mes = -1

    def save_model(self, mes, w, b):
        if self.__mes != -1:
            if mes < self.__mes:
                self.__mes = mes
                self.__w = w
                self.__b = b
        else:
            self.__mes = mes

    def get_w(self):
        return self.__w

    def get_b(self):
        return self.__b

    def get_mes(self):
        return self.__mes


w_list = []
b_list = []
mse_list = []
best = Best()
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-4.1, 4.1, 0.1):
        l_sum = 0
        print('\t', 'w=', + w, 'b=', + b)
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', 'x_val', + x_val, 'y_val', + y_val, 'y_pred_val=', + y_pred_val, 'loss_val', + loss_val)
        best.save_model(l_sum / len(x_data), w, b)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / len(x_data))

print("==========================================")
print("done!")
print('\t', "best model", best.get_mes(), best.get_w(), best.get_b())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w_list, b_list, mse_list)
ax.set_xlabel('W')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

plt.savefig('line model p2.png')

plt.show()
