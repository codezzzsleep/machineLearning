import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

train_dataset = datasets.MNIST(root='../../dataset/mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='../../dataset/mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)
if __name__ == '__main__':
    for epoch in range(100):
        for batch_idx, data in enumerate(train_loader):
            inputs,labels = data
            
