from __future__ import annotations
import torch, torch.nn as nn
from .encoders import TimeSeriesEncoder
from .rel_gnn import MultiRelationGNN
from torch_geometric.nn import SAGEConv, global_mean_pool

class StockGNN(nn.Module):
    def __init__(self, cfg, rel_names):
        super().__init__()
        idim = cfg['model']['in_dim']
        hdim = cfg['model']['hid_dim']
        heads = cfg['model']['heads']
        drop = cfg['model']['dropout']
        self.encoder = TimeSeriesEncoder(idim, idim)  # same dim out
        self.gnn = MultiRelationGNN(rel_names, idim, hdim, heads, drop)
        self.predictor = nn.Sequential(nn.Linear(hdim,64),nn.ReLU(),
                                       nn.Linear(64,1))

    def forward(self, data):
        # data['stock'].x: [N,F]
        h0 = self.encoder(data['stock'].x.unsqueeze(1))  # dummy T=1
        edge_index_dict={k:data[k].edge_index for k in data.edge_types}
        edge_attr_dict ={k:data[k].edge_attr for k in data.edge_types}
        h, alpha = self.gnn(h0, edge_index_dict, edge_attr_dict)
        out = self.predictor(h).squeeze()
        return out, alpha


class GNNEncoder(nn.Module):
    """Shared encoder for both pre-train and downstream tasks."""

    def __init__(self, in_dim: int, hid_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hid_dim, hid_dim))
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        return global_mean_pool(x, batch)     # graph-level embedding


class ProjectionHead(nn.Module):
    """2-layer MLP used in GraphCL to obtain contrastive views."""
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class GraphCLModel(nn.Module):
    """Encoder + projection head wrapper."""
    def __init__(self, in_dim: int, hid_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hid_dim)
        self.projector = ProjectionHead(hid_dim, proj_dim)

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.batch)
        z = self.projector(h)
        z = nn.functional.normalize(z, dim=-1)
        return z
