import torch


def train(train_dataloader, val_dataloader, model, device, optimizer, criterion):
    for i in range(100):
        model.train()
        loss_list_train = []
        loss_list_val = []
        for X, y in train_dataloader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            loss_list_train.append(loss.item())
            loss.backward()
            optimizer.step()
        model.eval()
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                pred = model(X)
                loss = criterion(pred, y)
            loss_list_val.append(loss.item())
