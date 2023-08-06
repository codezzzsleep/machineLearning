import torch.nn as nn
import torch.optim


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(8, 4)
        self.linear2 = nn.Linear(4, 2)
        self.linear3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.activate = nn.ReLU()

    def forward(self, X):
        X = self.activate(self.linear1(X))
        X = self.activate(self.linear2(X))
        X = self.sigmoid(self.linear3(X))
        return X


model = MyNet()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_dataset = []
test_dataset = []
for epoch in range(100):
    loss_record = []
    for X, y in train_dataset:
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    mean_train_loss = sum(loss_record) / len(loss_record)
    print(f"Epoch {epoch + 1:03d}: Train Loss: {mean_train_loss:.4f}")
pred_test = model(test_dataset)
print(pred_test)
