from torch_geometric.datasets import TUDataset

dataset = TUDataset('../dataset/', name='ENZYMES')

print(len(dataset))

print(dataset.num_classes)
print(dataset.num_features)

data = dataset[0]
print(data)
