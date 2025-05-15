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
        # 시계열 인코더: 주식 노드 (이변량), 거시 노드 (단변량), 이벤트 노드 (선형 변환)
        self.stock_encoder = TimeSeriesEncoder(2, idim)
        self.macro_encoder = TimeSeriesEncoder(1, idim)
        M = len(cfg['data']['macro_list'])
        self.event_encoder = nn.Linear(M + 1, idim)
        # 다중 관계 GNN (GATConv 기반)
        self.gnn = MultiRelationGNN(rel_names, idim, hdim, heads, drop)
        # 최종 예측기 (다음날 수익률 회귀)
        self.predictor = nn.Sequential(
            nn.Linear(hdim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # 주식 노드 시계열 인코딩 (shape: [N_stock, T, 2])
        h_stock = None
        if 'stock' in data.node_types:
            if hasattr(data['stock'], 'x_seq'):
                h_stock = self.stock_encoder(data['stock'].x_seq)
            else:
                h_stock = self.stock_encoder(data['stock'].x.unsqueeze(1))
        # 거시 노드 시계열 인코딩 (shape: [N_macro, T, 1])
        h_macro = None
        if 'macro' in data.node_types:
            if hasattr(data['macro'], 'x_seq'):
                h_macro = self.macro_encoder(data['macro'].x_seq)
            else:
                h_macro = self.macro_encoder(data['macro'].x.unsqueeze(1))
        # 이벤트 노드 인코딩 (정적 특징 → 임베딩)
        h_event = None
        if 'event' in data.node_types:
            h_event = self.event_encoder(data['event'].x)
        # 모든 노드 임베딩을 하나의 텐서로 결합 (stock, macro, event 순서)
        h_list = []
        if h_stock is not None:
            h_list.append(h_stock)
        if h_macro is not None:
            h_list.append(h_macro)
        if h_event is not None:
            h_list.append(h_event)
        h0 = torch.cat(h_list, dim=0) if len(h_list) > 1 else h_list[0]
        # 각 관계별 글로벌 edge_index 및 edge_attr 구성
        N_stock = data['stock'].num_nodes
        N_macro = data['macro'].num_nodes if 'macro' in data.node_types else 0
        N_event = data['event'].num_nodes if 'event' in data.node_types else 0
        offsets = {'stock': 0, 'macro': N_stock, 'event': N_stock + N_macro}
        edge_index_dict = {}
        edge_attr_dict = {}
        for (src, rel, dst) in data.edge_types:
            ei = data[(src, rel, dst)].edge_index
            ea = data[(src, rel, dst)].edge_attr
            src_off = offsets[src]; dst_off = offsets[dst]
            ei_global = ei.clone()
            if src_off:
                ei_global[0, :] += src_off
            if dst_off:
                ei_global[1, :] += dst_off
            key = f"{src}__{rel}__{dst}"
            edge_index_dict[key] = ei_global
            edge_attr_dict[key] = ea
        # GNN 메시지 패싱 (GATConv 적용, multi-relation gating)
        h, alpha = self.gnn(h0, edge_index_dict, edge_attr_dict)
        # 주식 노드에 대한 다음날 수익률 예측 출력
        out = self.predictor(h[:N_stock]).squeeze(-1)
        return out, alpha
