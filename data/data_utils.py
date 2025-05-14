"""
- fetch_raw():  Yahoo Finance/API → csv 저장
- build_features():  pct_change · z-score · forward-fill
- build_event_table():  macro shock table
"""
from core.utils import ensure_dir
import pandas as pd, numpy as np, yfinance as yf, datetime as dt

def fetch_raw(tickers, start="2018-01-01", end="2023-12-31", out_dir="./data/raw"):
    ensure_dir(out_dir)
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        df.to_csv(f"{out_dir}/{t}.csv")
    print(f"Downloaded {len(tickers)} tickers.")

def build_features(raw_dir, out_path):
    frames = []
    for csv in os.listdir(raw_dir):
        df = pd.read_csv(f"{raw_dir}/{csv}", parse_dates=['Date'])
        df['ticker'] = csv[:-4]
        frames.append(df[['Date','Close','Volume','ticker']])
    df = pd.concat(frames)
    df['ret'] = df.groupby('ticker')['Close'].pct_change()
    df['ret_z'] = df.groupby('ticker')['ret'].transform(lambda x:(x-x.mean())/x.std())
    df['vol_z'] = df.groupby('ticker')['Volume'].transform(lambda x:(x-x.mean())/x.std())
    df = df.dropna()
    df.to_parquet(out_path)
