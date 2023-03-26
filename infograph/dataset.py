import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd


class MyData(InMemoryDataset):
    def __init__(self, root='../mydata/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        x = torch.tensor([
            [0], [0], [0], [1], [1], [1], [1], [2], [2], [3], [3]
        ], dtype=torch.float32)
        edge_index = torch.tensor([
            [0, 1, 0, 3, 4, 5, 7, 9],
            [1, 2, 2, 4, 5, 6, 8, 19]
        ], dtype=torch.int64)
        y = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
        self.data = Data(x=x, edge_index=edge_index)
        self.slices = {
            'x': x, 'edge_index': edge_index
        }

    @property
    def processed_file_names(self):
        return ['data.pt']


class KKIData(InMemoryDataset):
    def __init__(self, root='../data/KKI/KKI/', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, _ = out

    # @property
    # def raw_file_names(self):
    #     return ['KKI_A.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']
