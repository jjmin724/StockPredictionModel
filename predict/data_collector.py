# predict/data_collector.py
import os
from core.utils import load_json
from data_collect.stock_data_fetcher import StockDataCollector
from data_collect.collect_macro import MacroDataCollector

class PredictDataCollector:
    """
    예측 모듈에서 사용할 데이터를 수집한다.
    - 주가 및 재무제표 최신 데이터
    - 거시경제 지표 최신 데이터
    """
    def __init__(self):
        self.cfg = load_json("config/config.json")
        map_file = f"{self.cfg['data']['processed']}/graph.json"
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"{map_file} not found. Preprocess 먼저 실행하세요.")
        mapping = load_json(map_file).get('ticker2id', {})
        self.tickers = list(mapping.keys())

    def fetch_all(self):
        # 주가 및 재무제표
        stock_col = StockDataCollector(self.tickers)
        stock_col.fetch_prices()
        stock_col.fetch_financials()
        # 거시경제 지표
        macro_col = MacroDataCollector()
        macro_col.fetch()
