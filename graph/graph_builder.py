import pandas as pd, numpy as np, torch
from scipy.stats import pearsonr
from fastdtw import fastdtw
from statsmodels.tsa.stattools import grangercausalitytests
from torch_geometric.data import HeteroData
from core.utils import load_json, save_json
from data.data_utils import build_event_table

class GraphBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def corr_edges(self, mat):
        edges, attrs = [], []
        thr = self.cfg['graph']['corr_thr']
        for i in range(mat.shape[0]):
            for j in range(i+1, mat.shape[0]):
                r, _ = pearsonr(mat[i], mat[j])
                if abs(r) >= thr:
                    edges.append([i, j]); edges.append([j, i])
                    attrs += [r, r]
        return torch.tensor(edges).t(), torch.tensor(attrs).unsqueeze(-1)

    def dtw_edges(self, series, k):
        N = len(series)
        dist_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                dist, _ = fastdtw(series[i], series[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
        edges, attrs = [], []
        for i in range(N):
            idx = np.argsort(dist_mat[i])[:k+1]  # 자기 자신 포함
            for j in idx:
                if i == j: 
                    continue
                edges.append([i, j])
                attrs.append(1.0/(dist_mat[i, j] + 1e-6))
        return torch.tensor(edges).t(), torch.tensor(attrs).unsqueeze(-1)

    def granger_edges(self, series, maxlag=5, p_thr=0.05):
        N = len(series)
        edges = []
        for i in range(N):
            for j in range(N):
                if i == j: 
                    continue
                gtest = grangercausalitytests(np.vstack([series[j], series[i]]).T,
                                              maxlag=maxlag, verbose=False)
                pvals = [gtest[L][0]['ssr_chi2test'][1] for L in range(1, maxlag+1)]
                if min(pvals) < p_thr:
                    edges.append([i, j])
        e = torch.tensor(edges).t()
        return e, torch.ones(e.size(1), 1)

    def build(self, feat_path, out_path):
        df = pd.read_parquet(feat_path)
        tickers = df['ticker'].unique().tolist()
        mapping = {t: i for i, t in enumerate(tickers)}
        data = HeteroData()
        # 주식 노드 특징 (표준화된 수익률/거래량의 평균값 사용)
        data['stock'].x = torch.tensor(
            df.groupby('ticker')[['ret_z', 'vol_z']].mean().values,
            dtype=torch.float
        )
        # 주식 노드 레이블 (다음날 수익률)
        labels_path = feat_path.replace('features.parquet', 'labels.json')
        try:
            label_map = load_json(labels_path)
            data['stock'].y = torch.tensor(
                [label_map.get(t, 0.0) for t in tickers], dtype=torch.float
            )
        except:
            pass
        # 거시지표 연속 시계열 및 이벤트 추출
        macro_series, events = build_event_table(
            self.cfg['data']['macro_list'],
            self.cfg['data']['root'],
            self.cfg['graph']['shock_lambda']
        )
        macro_names = list(macro_series.keys()) if macro_series else []
        # 주식 노드 시계열 (마지막 window일 기준)
        window = self.cfg['data'].get('window', 60)
        stock_seq_data = []
        for t in tickers:
            sub = df[df.ticker == t][['ret_z', 'vol_z']]
            seq = sub.tail(window).values
            if len(seq) < window:
                seq = np.vstack([np.zeros((window - len(seq), 2)), seq])
            stock_seq_data.append(seq)
        data['stock'].x_seq = torch.tensor(np.array(stock_seq_data), dtype=torch.float)
        # 거시지표 노드 시계열 (일간/주간 지표에 한함)
        if macro_names:
            macro_seq_data = []
            for m in macro_names:
                s = macro_series[m]
                arr = np.array(s.tail(window))
                if len(arr) < window:
                    arr = np.concatenate([np.zeros(window - len(arr)), arr])
                arr = arr.reshape(-1, 1)
                macro_seq_data.append(arr)
            data['macro'].x_seq = torch.tensor(np.array(macro_seq_data), dtype=torch.float)
        # 거시지표 노드 특징 (연속 지표의 평균 및 표준편차)
        if macro_names:
            macro_feat = []
            for m in macro_names:
                s = macro_series[m]
                macro_feat.append([s.mean(), s.std()])
            data['macro'].x = torch.tensor(macro_feat, dtype=torch.float)
        # 이벤트 노드 특징 (거시 지표 종류 one-hot + 충격 크기)
        if events:
            M = len(self.cfg['data']['macro_list'])
            event_feat = []
            for ev in events:
                one_hot = [0] * M
                if ev['macro'] in self.cfg['data']['macro_list']:
                    idx = self.cfg['data']['macro_list'].index(ev['macro'])
                    one_hot[idx] = 1
                feat_vec = one_hot + [ev['shock']]
                event_feat.append(feat_vec)
            data['event'].x = torch.tensor(event_feat, dtype=torch.float)
        # 주식-주식 관계 그래프 (상관관계, DTW, 그랜저 인과)
        series = [df[df.ticker == t]['ret_z'].values for t in tickers]
        e_corr, w_corr = self.corr_edges(np.stack(series))
        data['stock', 'corr', 'stock'].edge_index = e_corr
        data['stock', 'corr', 'stock'].edge_attr = w_corr
        e_dtw, w_dtw = self.dtw_edges(series, self.cfg['graph']['dtw_k'])
        data['stock', 'dtw', 'stock'].edge_index = e_dtw
        data['stock', 'dtw', 'stock'].edge_attr = w_dtw
        e_gr, w_gr = self.granger_edges(series, p_thr=self.cfg['graph']['granger_p'])
        data['stock', 'granger', 'stock'].edge_index = e_gr
        data['stock', 'granger', 'stock'].edge_attr = w_gr
        # (Optional) 산업 기반 엣지 추가 가능 (예: 동일 섹터 주식 간 edges)
        # 거시지표 (연속형) -> 주식 엣지 (기술적 유사도: 상관계수 활용)
        if macro_names:
            stock_series = {t: df[df.ticker == t].set_index('Date')['ret'] for t in tickers}
            macro_edges = []; macro_weights = []
            for mi, m in enumerate(macro_names):
                s_macro = macro_series[m]
                for t in tickers:
                    s_stock = stock_series[t]
                    r = s_stock.corr(s_macro)
                    if pd.isna(r):
                        continue
                    macro_edges.append([mi, mapping[t]])
                    macro_weights.append(r)
            if macro_edges:
                data['macro', 'influence', 'stock'].edge_index = torch.tensor(macro_edges).t().contiguous()
                data['macro', 'influence', 'stock'].edge_attr = torch.tensor(macro_weights, dtype=torch.float).unsqueeze(-1)
        # 거시 이벤트 -> 주식 엣지 (충격 전이)
        if events:
            event_edges = []; event_weights = []
            for ei in range(len(events)):
                for j in range(len(tickers)):
                    event_edges.append([ei, j])
                    event_weights.append(events[ei]['shock'])
            data['event', 'shock', 'stock'].edge_index = torch.tensor(event_edges).t().contiguous()
            data['event', 'shock', 'stock'].edge_attr = torch.tensor(event_weights, dtype=torch.float).unsqueeze(-1)
        # 그래프 데이터 저장
        torch.save(data, out_path)
        save_json({'ticker2id': mapping}, out_path.replace('.pt', '.json'))
        print(f"Graph saved to {out_path}")
