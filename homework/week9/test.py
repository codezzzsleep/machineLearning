import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from net import Model

batch_size = 64
test_dataset = datasets.MNIST(root='../../dataset/mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)
model = Model()


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
