from typing import Optional, Any
from numpy.typing import NDArray
from torch_geometric.data import Data, Dataset, OnDiskDataset
import torch
import numpy as np

from torch_geometric.data.data import BaseData
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.loader.mixin import AffinityMixin

class OnDiskStencilGraph(OnDiskDataset):
    def __init__(self,
                 features: NDArray,
                 root: str,
                 transform=None,
                 pre_filter=None):

        self.features  = np.ascontiguousarray(features).astype(np.float32, copy=False)

        self.total_datapoints = features.shape[0]

        self.max_neighbours = self.features.shape[1]    # max number of neighbours

        self.edges_max = torch.tensor(
            [[i, 0] for i in range(1, self.max_neighbours)],
            dtype=torch.long).T


        self.schema = {
            'x': dict(dtype=torch.float32, size=(-1,2)),
            'edge_index': dict(dtype=torch.long, size=(2,-1))
        }

        super().__init__(root, transform, pre_filter, backend='sqlite', schema=self.schema)
        self.db.connect()


    @property
    def processed_file_names(self):
        # Name of database
        return ['data.db']

    # Unused since I'm directly using multi_insert in process
    def serialize(self, data: BaseData) -> Any:
        return {
            "x": data['x'],
            "edge_index": data['edge_index']
        }

    def deserialize(self, data: Any) -> BaseData:
        return  Data(x=data['x'],
                     edge_index=data['edge_index']
                     )


    def len(self) -> int:
        return len(self.db)


    def get(self, idx): # if I change the database name format  I'll have to change this
        return self.deserialize(self.db.get(idx))


    def process(self):

        multi_idx = []
        multi_data = []
        insert_interval = 1000
        for idx in tqdm(range(self.total_datapoints), desc="Processing graphs"):


            x = torch.from_numpy(self.features[idx].copy()).to(torch.float32)

            # slice down to actual degree
            edge_index = self.edges_max
            tmp = [1,0]
            rev_edge_index = edge_index[tmp, :]
            edge_index = torch.concat((edge_index, rev_edge_index), dim=1)

            data_dict = {
                        'x': x,
                        'edge_index': edge_index
            }


            multi_idx.append(idx)
            multi_data.append(data_dict)

            if (idx + 1) % insert_interval == 0:
                self.db.multi_insert(multi_idx, multi_data)
                multi_idx, multi_data = [], []

        if multi_idx:
            self.db.multi_insert(multi_idx, multi_data)



class CustomLoader(AffinityMixin, DataLoader):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.data = data


class VariableStencilGraph(Dataset):
    """
    In-memory star-graph dataset where every point has its own stencil size.

    `features` are the *raw* (un-normalised) neighbour offsets of shape [N, K, 2], ordered nearest first
    with the centre point at index 0. Point `i` only sees its first `stencil_sizes[i]` neighbours and its
    offsets are normalised by the largest distance inside that (current) stencil.

    `stencil_sizes` lives in shared memory, so it can be modified from the main process (see
    `functions/stencil_growth.py`) and the (persistent) dataloader workers pick the change up on the next epoch.
    """
    def __init__(self,
                 features: NDArray,
                 initial_size: int = 5,
                 transform=None):
        super().__init__(transform=transform)

        self.features = torch.from_numpy(np.ascontiguousarray(features).astype(np.float32, copy=False))
        self.total_datapoints = self.features.shape[0]
        self.max_neighbours = self.features.shape[1]
        self.min_size = min(initial_size, self.max_neighbours)

        self.stencil_sizes = torch.full((self.total_datapoints,), self.min_size, dtype=torch.long)
        self.stencil_sizes.share_memory_()

        # star graph edges for every possible stencil size k: (i -> 0) and (0 -> i), for i in [1, k)
        self._edges = [None] * (self.max_neighbours + 1)
        for k in range(1, self.max_neighbours + 1):
            e = torch.tensor([[i, 0] for i in range(1, k)], dtype=torch.long).reshape(-1, 2).T
            self._edges[k] = torch.cat((e, e[[1, 0], :]), dim=1)

    def len(self) -> int:
        return self.total_datapoints

    def get(self, idx: int) -> Data:
        k = int(self.stencil_sizes[idx])
        x = self.features[idx, :k]
        max_r = torch.linalg.norm(x, dim=-1).max()
        return Data(x=x / max_r,
                    edge_index=self._edges[k],
                    point_idx=torch.tensor([idx], dtype=torch.long))
