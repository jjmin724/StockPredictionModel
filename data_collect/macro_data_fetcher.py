import os
import json
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

class MacroDataCollector:
    """
    config/config.json 의 data.macro_list 항목을 읽어
    FRED 지표를 ./data/raw/macro/{지표}.csv 로 저장
    """
    def __init__(self, start="2018-01-01", end=None):
        self.start = start
        self.end = end
        try:
            with open("config/config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            self.macro_list = cfg["data"]["macro_list"]
        except Exception as e:
            print(f"[Error] config.json 읽기 실패: {e}")
            self.macro_list = []
        self.out_dir = "./data/raw/macro"
        os.makedirs(self.out_dir, exist_ok=True)

    def fetch(self):
        # FRED 코드명 매핑 및 대체 소스 설정
        fred_code_map = {
            "CPI": "CPIAUCSL",
            "FED_FUNDS": "FEDFUNDS",
            "PPI": "PPIACO",
            "GDP": "GDP"
        }
        yahoo_ticker_map = {
            "DXY": "DX-Y.NYB"
        }
        for m in self.macro_list:
            file_path = f"{self.out_dir}/{m}.csv"
            if os.path.exists(file_path):
                print(f"[Skip] {m} 이미 존재")
                continue
            # 실제 코드 및 데이터 소스 결정
            if m in fred_code_map:
                code = fred_code_map[m]
                source = "fred"
            elif m in yahoo_ticker_map:
                code = yahoo_ticker_map[m]
                source = "yahoo"
            else:
                code = m
                source = "fred"
            try:
                if source == "fred":
                    print(f"[Info] Downloading {code} from FRED...")
                    series = pdr.DataReader(code, "fred", start=self.start, end=self.end)
                    if series.empty:
                        print(f"[Warn] {code} 데이터가 비어 있음")
                        continue
                    series.to_csv(file_path)
                elif source == "yahoo":
                    print(f"[Info] Downloading {code} from Yahoo Finance...")
                    df = yf.download(code, start=self.start, end=self.end, progress=False, auto_adjust=False)
                    if df.empty:
                        print(f"[Warn] {m} 데이터가 비어 있음 (Yahoo)")
                        continue
                    # 'Close' 열 기준으로 시계열 저장 (영업일 빈칸 채우기)
                    series_df = df[['Close']].copy()
                    series_df.index = pd.to_datetime(series_df.index)
                    series_df = series_df.resample('B').ffill()
                    series_df.reset_index(inplace=True)
                    series_df.to_csv(file_path, index=False)
                print(f"[Saved] {file_path}")
            except Exception as e:
                print(f"[Error] {m} 수집 실패: {e}")
