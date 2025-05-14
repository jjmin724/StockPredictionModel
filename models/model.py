import torch, torch.nn as nn
from .encoders import TimeSeriesEncoder
from .rel_gnn import MultiRelationGNN

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
