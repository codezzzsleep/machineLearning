from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import CoraFull

# dataset = TUDataset('../dataset/', name='ENZYMES')
dataset2 = CoraFull('../dataset/')
print(len(dataset2))
print(dataset2)
print(dataset2.num_classes)
print(dataset2.num_features)

data = dataset2[0]
print(data)
