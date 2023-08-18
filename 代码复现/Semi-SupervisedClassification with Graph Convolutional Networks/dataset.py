from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


dataset = MyDataset()
train_dataloader = DataLoader()
