"""Position sizing and risk controls."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def size_positions(
    signal: pd.Series,
    returns_window: pd.DataFrame,
    asset_class: dict[str, str],
    cfg: dict,
) -> pd.Series:
    """Turn signals in [-1, 1] into portfolio weights for one date.

    1. Inverse-volatility sizing so each asset contributes similar risk.
    2. Long-only constraint where shorting is not allowed.
    3. Per-asset cap, then scale the book to the portfolio vol target using the
       recent covariance matrix, capped by maximum gross leverage.
    """
    r = cfg["risk"]
    sig = signal.dropna()
    if sig.empty:
        return pd.Series(0.0, index=signal.index)

    window = returns_window[sig.index].dropna(how="all").fillna(0.0)
    vol = window.std() * np.sqrt(TRADING_DAYS)
    vol = vol.replace(0, np.nan)

    raw = (sig / vol).fillna(0.0)
    for name in raw.index:
        if raw[name] < 0 and not r["allow_short"][asset_class[name]]:
            raw[name] = 0.0
    if raw.abs().sum() == 0:
        return pd.Series(0.0, index=signal.index)

    w = raw / raw.abs().sum()
    w = w.clip(-r["max_weight_per_asset"], r["max_weight_per_asset"])

    cov = window.cov() * TRADING_DAYS
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    w = w * (r["target_vol_annual"] / port_vol)

    gross = w.abs().sum()
    if gross > r["max_gross_leverage"]:
        w = w * (r["max_gross_leverage"] / gross)
    # re-apply the cap after scaling up
    w = w.clip(-r["max_weight_per_asset"], r["max_weight_per_asset"])
    return w.reindex(signal.index).fillna(0.0)


def ex_ante_vol(weights: pd.Series, returns_window: pd.DataFrame) -> float:
    cov = returns_window[weights.index].fillna(0.0).cov() * TRADING_DAYS
    return float(np.sqrt(max(weights @ cov @ weights, 0.0)))


def historical_var(weights: pd.Series, returns_window: pd.DataFrame, level: float = 0.95) -> float:
    """1-day historical VaR as a positive fraction of capital."""
    pnl = returns_window[weights.index].fillna(0.0) @ weights
    return float(-np.quantile(pnl, 1 - level))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1
