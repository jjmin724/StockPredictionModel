# predict/predictor.py
import os
import torch
from core.utils import load_json, save_json, set_seed
from data.data_utils import build_features
from graph.graph_builder import GraphBuilder
from models.model import StockGNN
import pandas as pd

class StockPredictor:
    """
    저장된 모델로부터 다음날 주식 가격을 예측하고 결과를 저장한다.
    """
    def __init__(self, cfg_path="config/config.json"):
        self.cfg = load_json(cfg_path)
        set_seed(self.cfg["seed"])
        # 기존 그래프 로드 (노드 수와 관계 정보 확인)
        graph_pt = f"{self.cfg['data']['processed']}/graph.pt"
        if not os.path.exists(graph_pt):
            raise FileNotFoundError(f"{graph_pt} not found. Preprocess 먼저 실행하세요.")
        data = torch.load(graph_pt)
        rels = [f"{s}__{r}__{d}" for s, r, d in data.edge_types]
        # 모델 초기화 및 가중치 로드
        self.model = StockGNN(self.cfg, rels)
        model_path = f"{self.cfg['artifacts_dir']}/best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found. Train 먼저 실행하세요.")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_next_day(self):
        # 최신 데이터로 특징 및 그래프 재생성
        build_features(
            self.cfg["data"]["root"],
            f"{self.cfg['data']['processed']}/features.parquet"
        )
        gb = GraphBuilder(self.cfg)
        gb.build(
            f"{self.cfg['data']['processed']}/features.parquet",
            f"{self.cfg['data']['processed']}/graph.pt"
        )
        data = torch.load(f"{self.cfg['data']['processed']}/graph.pt")
        # 예측 실행
        with torch.no_grad():
            out, _ = self.model(data)  # [N_stock] 차원의 예측 수익률
        # 노드 ID ↔ 티커 매핑 로드
        mapping = load_json(f"{self.cfg['data']['processed']}/graph.json").get("ticker2id", {})
        inv_map = {v: k for k, v in mapping.items()}
        predictions = {}
        # 각 주식의 마지막 종가에 예측 수익률 반영
        for node_id, ticker in inv_map.items():
            price_file = f"{self.cfg['data']['root']}/{ticker}.csv"  # 수정: '/prices' 제거
            if os.path.exists(price_file):
                df = pd.read_csv(price_file)
                if not df.empty:
                    last_close = df.iloc[-1]['Close']
                    pred_ret = out[node_id].item()
                    pred_price = last_close * (1 + pred_ret)
                    predictions[ticker] = float(pred_price)
        # 예측 결과 저장
        out_file = "./data/next_day_predictions.json"
        save_json(predictions, out_file)
        print(f"[Saved] Predicted prices to {out_file}")
