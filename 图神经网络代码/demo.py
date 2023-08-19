import torch
import torch.nn as nn
import torch.nn.functional as F
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
        X, edge_index = data.x, data.edge_index
        X = self.activate(self.Conv1(X, edge_index))
        X = F.dropout(X, training=self.training)
        return F.log_softmax(X, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNet()

data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
torch.from_scipy_
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(data)
    loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
