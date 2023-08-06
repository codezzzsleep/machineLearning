import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid('../dataset', name='Cora')


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.Conv1 = GCNConv(dataset.num_features, 16)
        self.Conv2 = GCNConv(16, dataset.num_classes)
        self.activate = nn.ReLU()

    def forward(self, data):
        X, edge_index = data.X, data.edge_index
        X = self.activate(self.Conv1(X, edge_index))

