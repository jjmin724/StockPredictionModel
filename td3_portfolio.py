
"""
td3_portfolio.py

Standalone example of portfolio optimisation for U.S. equities using TD3 (Twin Delayed DDPG)
------------------------------------------------------------------------------------------

• Downloads OHLC data for Dow‑30 tickers via yfinance
• Builds technical indicators (RSI, MACD, etc.) with `ta`
• Calculates a Mahalanobis‑distance turbulence index
• Adds a naive one‑step EMA price forecast to the state
• Custom OpenAI Gym environment (`PortfolioEnv`) with:
    – state  = [prices, tech‑indicators, forecast, turbulence, holdings, cash]
    – action = continuous vector of target trade sizes (−1 … 1)
    – reward = portfolio daily return minus transaction cost
• Trains `stable_baselines3.TD3` for N timesteps
• Saves model and draws an equity curve of the back‑test

Requirements
------------
pip install yfinance ta stable-baselines3 gym numpy pandas scikit-learn matplotlib
"""

import gym
import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from sklearn.covariance import EmpiricalCovariance
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────
# 1. Parameters
# ─────────────────────────────────────────────────────────
TICKERS = [
    "AAPL","MSFT","JPM","UNH","V","HD","PG","KO","DIS","INTC",
    "IBM","GS","DOW","CAT","MMM","CRM","TRV","NKE","WBA","CVX",
    "XOM","VZ","BA","MCD","WMT","AXP","HON","JNJ","AMGN","MRK"
]
START_DATE   = "2014-01-01"
END_DATE     = "2024-12-31"
INITIAL_CASH = 1_000_000          # initial portfolio value (USD)
TRANSACTION_COST_PCT = 0.001      # proportional cost per trade
MAX_TRADE_PCT = 0.2               # max portion of PV traded per asset per step
LOOKBACK = 1                      # days of history needed before first action
TOTAL_TIMESTEPS = 50_000          # TD3 training steps


# ─────────────────────────────────────────────────────────
# 2. Data utilities
# ─────────────────────────────────────────────────────────
def fetch_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    df = df.dropna(how="all").fillna(method="ffill")
    return df

def add_technical_indicators(price_df):
    """
    Returns:
        close_pivot : (Date × ticker) closing price
        tech_pivot  : (Date × (feature,ticker)) technical indicators
    """
    feats = price_df.copy()
    feats = feats.stack().reset_index().rename(columns={"level_1":"ticker",0:"close"})
    feats["open"] = feats["high"] = feats["low"] = feats["close"]  # dummy OHLC
    feats = add_all_ta_features(
        feats,
        open="open", high="high", low="low", close="close", volume=None, fillna=True
    )
    feats = feats.set_index(["Date","ticker"]).sort_index()
    tech_pivot = feats.drop(columns=["open","high","low","close"]).unstack("ticker")
    return price_df, tech_pivot

def compute_turbulence(price_df, span=252):
    """
    Mahalanobis distance of today's returns to the trailing 'span' trade‑day window.
    """
    returns = price_df.pct_change().dropna()
    turb = pd.Series(index=returns.index, dtype=float)
    for i in range(span, len(returns)):
        hist = returns.iloc[i-span:i]
        cov  = EmpiricalCovariance().fit(hist)
        diff = (returns.iloc[i] - hist.mean()).values.reshape(1, -1)
        turb.iloc[i] = cov.mahalanobis(diff)[0]
    turb = turb.reindex(price_df.index).fillna(method="ffill")
    return turb

def ema_forecast(price_df, span=5):
    """
    One‑step‑ahead forecast with simple EMA; placeholder for any price predictor.
    """
    ema = price_df.ewm(span=span).mean()
    return ema.shift(1).fillna(method="bfill")


# ─────────────────────────────────────────────────────────
# 3. Custom trading environment
# ─────────────────────────────────────────────────────────
class PortfolioEnv(gym.Env):
    """
    Observation = concat( prices, forecast, tech‑indicators, holdings, cash, turbulence )
    Action      = continuous trade size vector in (−1 … 1) for each asset
    Reward      = daily return of portfolio value
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        tech_indicators: pd.DataFrame,
        forecast: pd.DataFrame,
        turbulence: pd.Series,
        initial_cash=INITIAL_CASH,
        max_trade_pct=MAX_TRADE_PCT,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        lookback=LOOKBACK
    ):
        super().__init__()
        self.prices = prices
        self.tech   = tech_indicators
        self.fore   = forecast
        self.turb   = turbulence
        self.assets = prices.columns.tolist()
        self.n_assets = len(self.assets)

        self.initial_cash = float(initial_cash)
        self.max_trade_pct = max_trade_pct
        self.cost_pct = transaction_cost_pct
        self.lookback = lookback

        # observation space size
        self.asset_feat_dim = (
            1 +                     # price
            1 +                     # forecast
            self.tech.columns.get_level_values(0).unique().size
        )
        self.obs_dim = (
            self.n_assets * self.asset_feat_dim +  # per‑asset features
            self.n_assets +                        # holdings (in shares)
            1 +                                    # cash
            1                                      # turbulence
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self._reset_portfolio()

    # Internal helpers -------------------------------------------------
    def _reset_portfolio(self):
        self.day = self.lookback
        self.end_day = len(self.prices) - 1
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=float)
        self.portfolio_value = self.initial_cash

    def _get_obs(self):
        p  = self.prices.iloc[self.day].values                 # prices
        f  = self.fore.iloc[self.day].values                   # forecast
        ti = self.tech.iloc[self.day].values                   # TA indicators
        asset_feats = np.concatenate([p, f, ti]).reshape(-1)
        obs = np.concatenate([
            asset_feats,
            self.holdings,
            [self.cash],
            [self.turb.iloc[self.day]]
        ])
        return obs.astype(np.float32)

    # Gym required methods ---------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_portfolio()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        today_prices = self.prices.iloc[self.day].values

        # translate action into trade sizes (shares)
        trade_value_target = action * self.max_trade_pct * self.portfolio_value
        trade_shares = np.floor(np.abs(trade_value_target) / today_prices) * np.sign(action)

        # Sell first
        sell_idx = trade_shares < 0
        sell_cash = (-trade_shares[sell_idx] * today_prices[sell_idx]).sum()
        self.holdings[sell_idx] += trade_shares[sell_idx]
        self.cash += sell_cash * (1 - self.cost_pct)

        # Buy
        buy_idx = trade_shares > 0
        buy_cost = (trade_shares[buy_idx] * today_prices[buy_idx]).sum()
        if buy_cost > self.cash:
            scale = self.cash / buy_cost
            trade_shares[buy_idx] = np.floor(trade_shares[buy_idx] * scale)
            buy_cost = (trade_shares[buy_idx] * today_prices[buy_idx]).sum()
        self.holdings[buy_idx] += trade_shares[buy_idx]
        self.cash -= buy_cost * (1 + self.cost_pct)

        # next day
        self.day += 1
        next_prices = self.prices.iloc[self.day].values
        prev_value = self.portfolio_value
        self.portfolio_value = self.cash + (self.holdings * next_prices).sum()

        reward = (self.portfolio_value - prev_value) / prev_value

        # Risk‑off rule: clear positions if turbulence extreme
        if self.turb.iloc[self.day] > self.turb.quantile(0.95):
            self.cash += (self.holdings * next_prices).sum() * (1 - self.cost_pct)
            self.holdings[:] = 0.0

        terminated = self.day >= self.end_day
        info = {"portfolio_value": self.portfolio_value}
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        print(f"Day {self.day} | PV: ${self.portfolio_value:,.2f} | Cash: ${self.cash:,.2f}")


# ─────────────────────────────────────────────────────────
# 4. Main – training & back‑testing
# ─────────────────────────────────────────────────────────
def main():
    # Data preparation -------------------------------------------------
    price_df = fetch_prices(TICKERS, START_DATE, END_DATE)
    close_df, tech_df = add_technical_indicators(price_df)
    turb_series = compute_turbulence(close_df)
    pred_df = ema_forecast(close_df)

    # Split train/test by date
    split_date = "2021-01-01"
    train_mask = close_df.index < split_date
    test_mask  = close_df.index >= split_date

    env_train = PortfolioEnv(
        prices=close_df.loc[train_mask],
        tech_indicators=tech_df.loc[train_mask],
        forecast=pred_df.loc[train_mask],
        turbulence=turb_series.loc[train_mask],
    )

    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

    model = TD3(
        policy="MlpPolicy",
        env=env_train,
        action_noise=action_noise,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=512,
        tau=0.02,
        gamma=0.99,
        train_freq=(1,"step"),
        gradient_steps=-1,
        verbose=1,
        policy_kwargs=dict(net_arch=[256,256,128]),
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("td3_portfolio_agent")

    # Back‑test --------------------------------------------------------
    env_test = PortfolioEnv(
        prices=close_df.loc[test_mask],
        tech_indicators=tech_df.loc[test_mask],
        forecast=pred_df.loc[test_mask],
        turbulence=turb_series.loc[test_mask],
    )

    obs, _ = env_test.reset()
    done = False
    values, dates = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_test.step(action)
        values.append(info["portfolio_value"])
        dates.append(env_test.prices.index[env_test.day])

    equity_curve = pd.Series(values, index=dates, name="TD3_Portfolio")
    print("\nFinal portfolio value:", f"${equity_curve.iloc[-1]:,.2f}")

    equity_curve.pct_change().dropna().plot(title="TD3 Portfolio Equity Curve")
    plt.show()


if __name__ == "__main__":
    main()
