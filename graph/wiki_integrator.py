import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class WikiGraphIntegrator:
    """
    위키백과 요약 정보(JSON)를 기반으로 TF-IDF 유사도를 계산하여
    주식 노드 간 'wiki' 관계 엣지를 생성
    """
    def __init__(self, wiki_file_path, thr=0.8, top_pct=None):
        self.wiki_file = wiki_file_path
        self.thr = thr
        self.top_pct = top_pct

    def compute_edges(self, tickers):
        # 위키 요약 텍스트 로드
        with open(self.wiki_file, encoding="utf-8") as f:
            wiki_data = json.load(f)
        texts = []
        for t in tickers:
            text = wiki_data.get(t, "")
            if text is None:
                text = ""
            texts.append(text)
        # TF-IDF 벡터화 및 코사인 유사도 계산
        try:
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception as e:
            print(f"[Error] TF-IDF 연산 실패: {e}")
            return [], []
        N = len(tickers)
        edges = []
        weights = []
        # 유사도 임계값 설정
        if self.top_pct:
            sims = sim_matrix.copy()
            np.fill_diagonal(sims, 0)
            flat = sims.flatten()
            cutoff = np.quantile(flat, 1 - self.top_pct) if flat.size > 0 else 1.0
            thr_val = cutoff
        else:
            thr_val = self.thr if self.thr is not None else 1.0
        # 임계값 이상인 기업 쌍에 대해 엣지 생성 (양방향)
        for i in range(N):
            for j in range(i+1, N):
                if sim_matrix[i, j] >= thr_val:
                    edges.append([i, j])
                    edges.append([j, i])
                    weights.append(float(sim_matrix[i, j]))
                    weights.append(float(sim_matrix[i, j]))
        return edges, weights
