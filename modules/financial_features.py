import json, pandas as pd, numpy as np
import torch

class FinancialFeatureModule:
    """
    재무제표 기반 정적 특징 (ROE·부채비율·EPS·유동비율·매출성장률) 전처리
    - 표준화(z-score), 결측치 업종 평균/전체 평균 대체
    - tickers 순서에 맞는 Tensor 반환
    """
    def __init__(self, file_path):
        self.file_path = file_path
        # ---------- load ----------
        if file_path.endswith('.json'):
            with open(file_path) as f:
                raw = json.load(f)
            self.df = pd.DataFrame.from_dict(raw, orient='index')
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            tid = 'ticker' if 'ticker' in df.columns else 'Ticker'
            df = df.set_index(tid)
            self.df = df
        else:
            raise ValueError("지원하지 않는 포맷(.json/.csv만 가능)")
        # 수치형 feature만 선정
        self.features = [c for c in self.df.columns
                         if self.df[c].dtype in (np.float64, np.int64)]
        self.industry_col = 'Industry' if 'Industry' in self.df.columns else None

    def process(self, tickers):
        df = self.df.copy()
        # 누락 ticker 행 추가
        for t in tickers:
            if t not in df.index:
                df.loc[t] = {feat: np.nan for feat in self.features}
                if self.industry_col:
                    df.at[t, self.industry_col] = None
        # 업종 평균으로 결측치 보정
        if self.industry_col:
            for feat in self.features:
                df[feat] = df.groupby(self.industry_col)[feat]\
                             .transform(lambda s: s.fillna(s.mean()))
        df[self.features] = df[self.features].fillna(df[self.features].mean())
        # z-score
        df[self.features] = (df[self.features] - df[self.features].mean())\
                            / df[self.features].std(ddof=0)
        df = df.reindex(tickers)
        return torch.tensor(df[self.features].values, dtype=torch.float)
