import os
import json
from pandas_datareader import data as pdr


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
        for m in self.macro_list:
            file_path = f"{self.out_dir}/{m}.csv"
            if os.path.exists(file_path):
                print(f"[Skip] {m} 이미 존재")
                continue
            try:
                print(f"[Info] Downloading {m} from FRED...")
                series = pdr.DataReader(
                    m, "fred", start=self.start, end=self.end
                )
                if series.empty:
                    print(f"[Warn] {m} 데이터가 비어 있음")
                    continue
                series.to_csv(file_path)
                print(f"[Saved] {file_path}")
            except Exception as e:
                print(f"[Error] {m} 수집 실패: {e}")
