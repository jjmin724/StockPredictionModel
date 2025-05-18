import json
import torch

class FinancialFeatureIntegrator:
    """
    재무 특징 JSON을 로드하여 주식 노드의 추가 특징 행렬을 생성
    """
    def __init__(self, fin_file_path):
        self.fin_file = fin_file_path

    def get_features(self, tickers):
        with open(self.fin_file, encoding="utf-8") as f:
            fin_data = json.load(f)
        features = []
        for t in tickers:
            vals = fin_data.get(t, {})
            ta = vals.get('total_assets')
            ni = vals.get('net_income')
            ocf = vals.get('operating_cash_flow')
            # None 값을 0으로 대체
            ta = 0.0 if ta is None else float(ta)
            ni = 0.0 if ni is None else float(ni)
            ocf = 0.0 if ocf is None else float(ocf)
            features.append([ta, ni, ocf])
        feat_tensor = torch.tensor(features, dtype=torch.float)
        return feat_tensor
