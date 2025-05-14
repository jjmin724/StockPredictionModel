import torch, torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    """Bi-directional GRU + LayerNorm"""
    def __init__(self, in_dim, hid):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid//2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hid)

    def forward(self, x):           # x: [B,T,F]
        _, h = self.gru(x)
        h = torch.cat([h[0],h[1]],dim=-1)   # [B,hid]
        return self.norm(h)
