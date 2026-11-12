"""Trading signals. Every signal at date t only uses prices up to and including t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(prices: pd.DataFrame, lookbacks: list[int]) -> pd.DataFrame:
    """Volatility-adjusted time-series momentum averaged over several horizons, in [-1, 1]."""
    rets = np.log(prices).diff()
    daily_vol = rets.rolling(63, min_periods=20).std()
    scores = []
    for lb in lookbacks:
        move = np.log(prices / prices.shift(lb))
        z = move / (daily_vol * np.sqrt(lb))
        scores.append(np.tanh(z))
    return sum(scores) / len(scores)


def trend(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Moving-average crossover: +1 above the slow average, -1 below."""
    ma_fast = prices.rolling(fast, min_periods=fast).mean()
    ma_slow = prices.rolling(slow, min_periods=slow).mean()
    return np.sign(ma_fast - ma_slow)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((loss == 0) & (gain > 0), 100.0)   # only gains -> maximally overbought
    out = out.mask((loss == 0) & (gain == 0), 50.0)   # flat -> neutral
    return out.where(prices.notna())


def mean_reversion(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Short-term contrarian signal from RSI: positive when oversold, negative when overbought."""
    return ((50 - rsi(prices, window)) / 50).clip(-1, 1)


def composite(prices: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    s = cfg["strategy"]
    parts = {
        "momentum": momentum(prices, s["momentum_lookbacks"]),
        "trend": trend(prices, s["trend_fast"], s["trend_slow"]),
        "mean_reversion": mean_reversion(prices, s["rsi_window"]),
    }
    w = s["weights"]
    total = sum(w.values())
    parts["composite"] = sum(parts[k] * (w[k] / total) for k in w).clip(-1, 1)
    return parts
