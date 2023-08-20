from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_data():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    print(data)
    return data


if __name__ == "__main__":
    data = load_data()
    print(type(data))
