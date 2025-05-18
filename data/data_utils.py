from core.utils import ensure_dir, load_json
import pandas as pd, numpy as np, os, yfinance as yf
import torch

def fetch_raw(tickers, start="2018-01-01", end="2023-12-31", out_dir="./data/raw"):
    ensure_dir(out_dir)
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        df.to_csv(f"{out_dir}/{t}.csv")
    print(f"Downloaded {len(tickers)} tickers.")

def build_features(raw_dir, out_path):
    ensure_dir(os.path.dirname(out_path))
    frames = []
    label_map = {}
    for csv in os.listdir(raw_dir):
        df = pd.read_csv(f"{raw_dir}/{csv}", parse_dates=['Date'])
        df.sort_values('Date', inplace=True)
        ticker = csv[:-4]
        if len(df) >= 2:
            # 마지막 행의 종가로 다음날 수익률 레이블 계산
            last_close = df.iloc[-1]['Close']; prev_close = df.iloc[-2]['Close']
            ret_last = (last_close / prev_close - 1)
            label_map[ticker] = float(ret_last)
        # 레이블로 사용된 마지막 행 제거 (학습 입력에서 제외)
        df = df.iloc[:-1]
        df['ticker'] = ticker
        if not df.empty:
            frames.append(df[['Date', 'Close', 'Volume', 'ticker']])
    df_all = pd.concat(frames)
    # 일일 수익률과 표준화된 수익률/거래량 특징 생성
    df_all['ret'] = df_all.groupby('ticker')['Close'].pct_change()
    df_all['ret_z'] = df_all.groupby('ticker')['ret'].transform(lambda x: (x - x.mean())/x.std())
    df_all['vol_z'] = df_all.groupby('ticker')['Volume'].transform(lambda x: (x - x.mean())/x.std())
    df_all = df_all.dropna()
    df_all.to_parquet(out_path)
    # 각 주식 티커의 레이블(다음날 수익률) 저장
    label_path = os.path.join(os.path.dirname(out_path), "labels.json")
    from core.utils import save_json
    save_json(label_map, label_path)

def build_event_table(macro_list, raw_dir, threshold):
    events = []
    macro_series = {}
    for m in macro_list:
        file_path = f"{raw_dir}/{m}.json"
        if not os.path.exists(file_path):
            continue
        data = load_json(file_path)
        df_m = pd.DataFrame(data)
        if len(df_m) < 2:
            continue
        # 데이터 간격 확인 (일간/주간/월간 등)
        df_m['date'] = pd.to_datetime(df_m['date'])
        df_m.sort_values('date', inplace=True)
        median_gap = (df_m['date'].diff().dt.days.dropna().median())
        freq_days = median_gap if pd.notna(median_gap) else None
        if freq_days is not None and freq_days <= 7:
            # 일간 또는 주간 지표 -> 비즈니스 데이 기준 일일 시계열로 변환
            df_idx = df_m.set_index('date')
            if freq_days > 1:
                df_idx = df_idx.resample('B').interpolate()  # 주간 지표 일일 보간
            if 'Close' in df_idx.columns:
                series_val = df_idx['Close']
            else:
                series_val = df_idx[df_idx.columns[0]]
            series_ret = series_val.pct_change().fillna(0)
            macro_series[m] = series_ret
        else:
            # 저주기 이벤트 지표 (월간/분기 발표 등) -> 이벤트 노드 생성
            if 'Close' in df_m.columns:
                values = df_m['Close'].values
            else:
                values = df_m[df_m.columns[1]].values
            for i in range(1, len(values)):
                change = (values[i] - values[i-1]) / (values[i-1] + 1e-9)
                shock_weight = change
                if threshold:
                    shock_weight = shock_weight / threshold
                events.append({"macro": m, "shock": float(shock_weight)})
    return macro_series, events

def load_pretrain_dataset(cfg):
    feat_path = os.path.join(cfg["data"]["processed"], "features.parquet")
    df = pd.read_parquet(feat_path)
    label_path = os.path.join(cfg["data"]["processed"], "labels.json")
    label_map = load_json(label_path)
    # 모든 티커를 정렬된 순서로 가져옴
    tickers = sorted(df['ticker'].unique().tolist())
    seqs = []
    labels = []
    for t in tickers:
        df_t = df[df.ticker == t]
        seq = df_t[['ret_z', 'vol_z']].values
        if len(seq) == 0:
            continue
        seqs.append(seq)
        labels.append(label_map.get(t, 0.0))
    if not seqs:
        return torch.empty(0), torch.empty(0)
    # 길이가 다른 시계열 패딩 (앞쪽을 0으로 채움)
    max_len = max(len(s) for s in seqs)
    padded = []
    for s in seqs:
        if len(s) < max_len:
            pad = np.zeros((max_len - len(s), s.shape[1]))
            s = np.vstack([pad, s])
        padded.append(s)
    X_seq = torch.tensor(np.array(padded), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    return X_seq, y
