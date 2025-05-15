# data_collect/stock_data_fetcher.py
import os
import pandas as pd
import yfinance as yf

class StockDataCollector:
    """
    S&P 500 티커 리스트에 대해 주가 CSV와 재무제표 CSV를 저장한다.
    ├─ ./data/raw/prices/TICKER.csv
    └─ ./data/raw/financials/TICKER_{statement}.csv
    이미 존재하는 파일은 건너뛴다.
    """

    def __init__(self, tickers, start="2018-01-01", end=None):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.prices_dir = "./data/raw/prices"
        self.fin_dir = "./data/raw/financials"
        os.makedirs(self.prices_dir, exist_ok=True)
        os.makedirs(self.fin_dir, exist_ok=True)

    # ------------------------------- #
    #           주가 수집             #
    # ------------------------------- #
    def fetch_prices(self):
        for ticker in self.tickers:
            file_path = f"{self.prices_dir}/{ticker}.csv"
            if os.path.exists(file_path):
                print(f"[Skip] Prices for {ticker} already exist.")
                continue
            try:
                print(f"[Info] Downloading prices for {ticker}...")
                df = yf.download(
                    ticker,
                    start=self.start,
                    end=self.end,
                    progress=False,
                    auto_adjust=False,
                )
                if df.empty:
                    print(f"[Warn] No price data fetched for {ticker}.")
                    continue
                # -- 비거래일 처리: 비즈니스데이 기준 리샘플, 결측치는 전일 종가로 보간 --
                df.index = pd.to_datetime(df.index)
                df = df.resample('B').ffill()
                df = df.reset_index()[["Date", "Close", "Volume"]]
                df.to_csv(file_path, index=False)
                print(f"[Saved] {file_path}")
            except Exception as e:
                print(f"[Error] Failed to fetch prices for {ticker}: {e}")

    # ------------------------------- #
    #         재무제표 수집           #
    # ------------------------------- #
    def fetch_financials(self):
        for ticker in self.tickers:
            t = yf.Ticker(ticker)
            statements = {
                "balance_sheet": t.balance_sheet,
                "income_statement": t.financials,
                "cash_flow": t.cashflow,
            }
            for name, df in statements.items():
                file_path = f"{self.fin_dir}/{ticker}_{name}.csv"
                if os.path.exists(file_path):
                    print(f"[Skip] {name} for {ticker} already exist.")
                    continue
                try:
                    print(f"[Info] Downloading {name} for {ticker}...")
                    if df is not None and not df.empty:
                        df.to_csv(file_path)
                        print(f"[Saved] {file_path}")
                    else:
                        print(f"[Warn] {name} for {ticker} is empty.")
                except Exception as e:
                    print(f"[Error] Failed to fetch {name} for {ticker}: {e}")
