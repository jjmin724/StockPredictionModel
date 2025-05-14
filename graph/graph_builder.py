import pandas as pd, numpy as np, torch
from scipy.stats import pearsonr
from fastdtw import fastdtw
from statsmodels.tsa.stattools import grangercausalitytests
from torch_geometric.data import HeteroData
from core.utils import load_json, save_json, ensure_dir
from tqdm import tqdm

class GraphBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    # ---------- relation builders ----------
    def corr_edges(self, mat):
        edges, attrs = [], []
        thr = self.cfg['graph']['corr_thr']
        for i in range(mat.shape[0]):
            for j in range(i+1, mat.shape[0]):
                r, _ = pearsonr(mat[i], mat[j])
                if abs(r) >= thr:
                    edges.append([i,j]); edges.append([j,i])
                    attrs += [r, r]
        return torch.tensor(edges).t(), torch.tensor(attrs).unsqueeze(-1)

    def dtw_edges(self, series, k):
        N = len(series)
        dist_mat = np.zeros((N,N))
        for i in tqdm(range(N)):
            for j in range(i+1,N):
                dist, _ = fastdtw(series[i], series[j])
                dist_mat[i,j]=dist_mat[j,i]=dist
        edges, attrs = [], []
        for i in range(N):
            idx = np.argsort(dist_mat[i])[:k+1]   # self 포함
            for j in idx:
                if i==j: continue
                edges.append([i,j])
                attrs.append(1.0/(dist_mat[i,j]+1e-6))
        return torch.tensor(edges).t(), torch.tensor(attrs).unsqueeze(-1)

    def granger_edges(self, series, maxlag=5, p_thr=0.05):
        N = len(series)
        edges=[]
        for i in tqdm(range(N)):
            for j in range(N):
                if i==j: continue
                gtest = grangercausalitytests(
                    np.vstack([series[j],series[i]]).T,
                    maxlag=maxlag, verbose=False)
                pvals=[gtest[L][0]['ssr_chi2test'][1] for L in range(1,maxlag+1)]
                if min(pvals) < p_thr:
                    edges.append([i,j])
        e = torch.tensor(edges).t()
        return e, torch.ones(e.size(1),1)

    # ---------- pipeline ----------
    def build(self, feat_path, out_path):
        df = pd.read_parquet(feat_path)
        tickers = df['ticker'].unique().tolist()
        mapping = {t:i for i,t in enumerate(tickers)}
        series = [df[df.ticker==t]['ret_z'].values for t in tickers]

        data = HeteroData()
        data['stock'].x = torch.tensor(
            df.groupby('ticker')[['ret_z','vol_z']].mean().values,
            dtype=torch.float)

        # base corr
        e, w = self.corr_edges(np.stack(series))
        data['stock','corr','stock'].edge_index = e
        data['stock','corr','stock'].edge_attr  = w

        # dtw
        k=self.cfg['graph']['dtw_k']
        e,w = self.dtw_edges(series,k)
        data['stock','dtw','stock'].edge_index=e
        data['stock','dtw','stock'].edge_attr =w

        # granger
        e,w = self.granger_edges(series, p_thr=self.cfg['graph']['granger_p'])
        data['stock','granger','stock'].edge_index=e
        data['stock','granger','stock'].edge_attr =w

        torch.save(data, out_path)
        save_json({'ticker2id':mapping}, out_path.replace('.pt','.json'))
        print(f"Graph saved to {out_path}")
