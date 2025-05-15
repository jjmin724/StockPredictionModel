# data_collect/financial_data_processor.py
import os
import pandas as pd
from core.utils import load_json, save_json

class FinancialDataProcessor:
    """
    재무제표 CSV 파일을 읽어 주요 지표를 추출한 후
    config['graph']['financial_feat_file'] 위치에 JSON으로 저장
    """
    def __init__(self):
        cfg = load_json("config/config.json")
        self.fin_dir = "./data/raw/financials"
        self.out_file = cfg["graph"]["financial_feat_file"]
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)

    def process(self):
        features = {}
        if not os.path.exists(self.fin_dir):
            print(f"[Error] Financial data directory not found: {self.fin_dir}")
            return
        files = os.listdir(self.fin_dir)
        tickers = set(f.split('_')[0] for f in files if f.endswith(".csv"))
        for ticker in tickers:
            ticker_feat = {}
            # 대차대조표: Total Assets
            try:
                bs_file = os.path.join(self.fin_dir, f"{ticker}_balance_sheet.csv")
                if os.path.exists(bs_file):
                    df_bs = pd.read_csv(bs_file, index_col=0)
                    if 'Total Assets' in df_bs.index:
                        ticker_feat['total_assets'] = float(df_bs.loc['Total Assets'].iloc[-1])
                    else:
                        ticker_feat['total_assets'] = None
                else:
                    ticker_feat['total_assets'] = None
            except:
                ticker_feat['total_assets'] = None

            # 손익계산서: Net Income
            try:
                is_file = os.path.join(self.fin_dir, f"{ticker}_income_statement.csv")
                if os.path.exists(is_file):
                    df_is = pd.read_csv(is_file, index_col=0)
                    if 'Net Income' in df_is.index:
                        ticker_feat['net_income'] = float(df_is.loc['Net Income'].iloc[-1])
                    else:
                        ticker_feat['net_income'] = None
                else:
                    ticker_feat['net_income'] = None
            except:
                ticker_feat['net_income'] = None

            # 현금흐름표: Total Cash From Operating Activities
            try:
                cf_file = os.path.join(self.fin_dir, f"{ticker}_cash_flow.csv")
                if os.path.exists(cf_file):
                    df_cf = pd.read_csv(cf_file, index_col=0)
                    if 'Total Cash From Operating Activities' in df_cf.index:
                        ticker_feat['operating_cash_flow'] = float(df_cf.loc['Total Cash From Operating Activities'].iloc[-1])
                    else:
                        ticker_feat['operating_cash_flow'] = None
                else:
                    ticker_feat['operating_cash_flow'] = None
            except:
                ticker_feat['operating_cash_flow'] = None

            features[ticker] = ticker_feat

        # JSON 저장
        save_json(features, self.out_file)
        print(f"[Saved] Financial features to {self.out_file}")
