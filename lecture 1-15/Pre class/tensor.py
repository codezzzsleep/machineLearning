import torch
import numpy as np

x = np.array([[1, 1], [1, 1], [1, 1]])
print(x)
x = torch.from_numpy(x)
print(x)
x = torch.tensor([[1, 1], [1, 1], [1, 1]])
print(x)
x = torch.zeros([2, 2])
print(x)
y = torch.ones([2, 3, 4])
print(y)
y = torch.ones([2, 2])
print(y - x)

print(y.pow(2))
