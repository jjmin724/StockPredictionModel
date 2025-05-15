import json, numpy as np, torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class WikipediaSimilarityModule:
    """
    위키백과 문서 유사도 기반 간선 생성
    edge_feature_type: 'weight' -> 가중치, 'embedding' -> MLP 임베딩
    """
    def __init__(self, wiki_path,
                 sim_threshold=0.8, top_percent=None,
                 edge_feature_type='weight'):
        with open(wiki_path) as f:
            self.wiki = json.load(f)
        self.tickers = list(self.wiki.keys())
        self.sim_thr = sim_threshold
        self.top_pct = top_percent
        self.edge_type = edge_feature_type
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if edge_feature_type == 'embedding':
            dim = self.model.get_sentence_embedding_dimension()
            self.edge_mlp = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(),
                nn.Linear(dim, dim))
        else:
            self.edge_mlp = None

    def _embed(self):
        texts = [self.wiki[t] for t in self.tickers]
        emb = self.model.encode(texts, batch_size=16, show_progress_bar=False)
        emb = np.asarray(emb)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return emb  # [N, dim]

    def build_edges(self):
        emb = self._embed()
        N, dim = emb.shape
        sim = emb @ emb.T
        np.fill_diagonal(sim, 0.0)

        # 임계값 결정
        thr = self.sim_thr
        if self.top_pct is not None:
            k = int(len(sim.flatten()) * (1 - self.top_pct))
            thr = np.partition(sim.flatten(), k)[k]

        edges, attrs = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if sim[i, j] >= thr:
                    edges += [[i, j], [j, i]]
                    if self.edge_type == 'weight':
                        attrs += [sim[i, j], sim[i, j]]
                    else:
                        cat_ij = np.concatenate([emb[i], emb[j]])
                        cat_ji = np.concatenate([emb[j], emb[i]])
                        attrs.append(self.edge_mlp(torch.tensor(cat_ij,
                                                                 dtype=torch.float))
                                     .detach().numpy())
                        attrs.append(self.edge_mlp(torch.tensor(cat_ji,
                                                                 dtype=torch.float))
                                     .detach().numpy())

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        if self.edge_type == 'weight':
            edge_attr = torch.tensor(attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_attr = torch.tensor(np.array(attrs), dtype=torch.float)
        return edge_index, edge_attr
