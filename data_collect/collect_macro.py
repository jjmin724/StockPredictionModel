import os
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from core.utils import load_json, save_json

class MacroDataCollector:
    """
    config/config.json 의 data.macro_list에 명시된 거시 경제 지표들을 FRED 또는 Yahoo Finance 에서 수집하여
    표준 포맷(JSON)으로 저장한다. 파일 경로: ./data/raw/prices/macro/{지표}.json (이미 존재하는 경우 건너뜀).
    """
    def __init__(self, start="2018-01-01", end=None):
        self.start = start
        self.end = end
        try:
            cfg = load_json("config/config.json")
            self.macro_list = cfg["data"]["macro_list"]
        except Exception as e:
            print(f"[Error] config.json 읽기 실패: {e}")
            self.macro_list = []
        self.out_dir = os.path.join(cfg["data"]["root"], "macro")
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
            file_path = os.path.join(self.out_dir, f"{m}.json")
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
                    data = pdr.DataReader(code, "fred", start=self.start, end=self.end)
                    if data.empty:
                        print(f"[Warn] {code} 데이터가 비어 있음")
                        continue
                    # DataFrame 형식으로 변환 및 인덱스 처리
                    if isinstance(data, pd.Series):
                        df = data.to_frame(name="value")
                    else:
                        df = data.copy()
                    df.index = pd.to_datetime(df.index)
                    df.reset_index(inplace=True)
                    # 컬럼명 정리: 'date', 'value'
                    col0, col1 = df.columns[0], df.columns[1]
                    df.rename(columns={col0: "date", col1: "value"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    records = df.to_dict("records")
                    save_json(records, file_path)
                elif source == "yahoo":
                    print(f"[Info] Downloading {code} from Yahoo Finance...")
                    df = yf.download(code, start=self.start, end=self.end, progress=False, auto_adjust=False)
                    if df.empty:
                        print(f"[Warn] {m} 데이터가 비어 있음 (Yahoo)")
                        continue
                    df = df[["Close"]].copy()
                    df.index = pd.to_datetime(df.index)
                    df = df.resample('B').ffill()
                    df.reset_index(inplace=True)
                    df.rename(columns={"Date": "date", "Close": "value"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    records = df.to_dict("records")
                    save_json(records, file_path)
                print(f"[Saved] {file_path}")
            except Exception as e:
                print(f"[Error] {m} 수집 실패: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)", default="2018-01-01")
    parser.add_argument("--end", help="End date (YYYY-MM-DD, default today)", default=None)
    args = parser.parse_args()
    collector = MacroDataCollector(start=args.start, end=args.end)
    collector.fetch()
