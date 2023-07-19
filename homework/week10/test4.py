import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)

pooling = torch.nn.MaxPool2d(kernel_size=2)

output = pooling(input)
print(output)
