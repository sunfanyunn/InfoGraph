import torch
from torch_geometric.data import InMemoryDataset
import pandas as pd


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data.x = torch.tensor(
            [[0], [0], [0], [1], [1], [1], [1]], dtype=torch.float32
        )
        self.data.y = torch.tensor([0, 1], dtype=torch.int)

from torch_geometric.data import InMemoryDataset

class KKIData(InMemoryDataset):
    def __init__(self, root='../data/KKI/KKI/', transform= None, pre_transform=None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, _ = out

    @property
    def raw_file_names(self):
        return ['KKI_A.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']