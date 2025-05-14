import torch, torch.nn as nn
from torch_geometric.nn import GATConv, LayerNorm

class MultiRelationGNN(nn.Module):
    def __init__(self, rels, in_dim, hid, heads, dropout):
        super().__init__()
        self.rels = rels
        self.convs = nn.ModuleDict({
            r: GATConv(in_dim, hid//heads, heads=heads, dropout=dropout, add_self_loops=False)
            for r in rels
        })
        self.gate = nn.Parameter(torch.ones(len(rels)))
        self.norm = LayerNorm(hid)
        self.rescale = nn.Linear(in_dim, hid) if in_dim!=hid else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_dict, edge_attr_dict):
        outs=[]
        for i,r in enumerate(self.rels):
            ei, ea = edge_index_dict[r], edge_attr_dict[r]
            outs.append(self.convs[r](x, ei, ea))
        alpha = torch.softmax(self.gate,0)
        h = torch.stack(outs,0)
        h = (alpha[:,None,None]*h).sum(0)
        h = self.norm(h+self.rescale(x))
        return self.dropout(h), alpha.detach().cpu().tolist()
