from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, file):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


dataset = MyDataset(file='../../res/diabetes.csv.gz')
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
