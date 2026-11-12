"""Walk-forward backtest: weights decided at the close of t-1 earn the returns of day t."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import TRADING_DAYS, drawdown, size_positions
from .signals import composite


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    costs: pd.Series
    signals: dict[str, pd.DataFrame]
    metrics: dict[str, float]
    benchmark: pd.Series | None = None


def run_backtest(prices: pd.DataFrame, asset_class: dict[str, str], cfg: dict,
                 benchmark: str | None = None, rebalance_every: int = 1) -> BacktestResult:
    r = cfg["risk"]
    rets = prices.pct_change()
    signals = composite(prices, cfg)
    sig = signals["composite"]

    warmup = max(cfg["strategy"]["trend_slow"], max(cfg["strategy"]["momentum_lookbacks"])) + 1
    dates = prices.index
    cost_bps = pd.Series({n: r["transaction_cost_bps"][asset_class[n]] for n in prices.columns}) / 1e4

    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    port_ret = pd.Series(0.0, index=dates)
    costs = pd.Series(0.0, index=dates)
    equity = 1.0
    peak = 1.0
    w_prev = pd.Series(0.0, index=prices.columns)
    target = w_prev.copy()

    for i in range(warmup, len(dates)):
        t = dates[i]
        # 1) P&L today from yesterday's book
        day_ret = float((w_prev * rets.iloc[i].fillna(0.0)).sum())

        # 2) decide tomorrow's book using information up to today
        if (i - warmup) % rebalance_every == 0:
            window = rets.iloc[max(0, i - r["vol_window"] + 1): i + 1]
            target = size_positions(sig.iloc[i], window, asset_class, cfg)
        dd = equity * (1 + day_ret) / max(peak, equity * (1 + day_ret)) - 1
        new_w = target * (0.5 if dd < -r["drawdown_brake"] else 1.0)
        # no-trade band: skip small adjustments that would only generate costs
        band = r.get("rebalance_band", 0.0)
        new_w = new_w.where((new_w - w_prev).abs() >= band, w_prev)
        gross = new_w.abs().sum()
        if gross > r["max_gross_leverage"]:  # mixing old and new weights can breach the limit
            new_w = new_w * (r["max_gross_leverage"] / gross)

        turnover = (new_w - w_prev).abs()
        cost = float((turnover * cost_bps).sum())
        net = day_ret - cost

        equity *= 1 + net
        peak = max(peak, equity)
        port_ret[t] = net
        costs[t] = cost
        weights.loc[t] = new_w
        w_prev = new_w

    port_ret = port_ret.iloc[warmup:]
    eq = (1 + port_ret).cumprod()
    bench = None
    if benchmark and benchmark in prices:
        b = rets[benchmark].iloc[warmup:].fillna(0.0)
        bench = (1 + b).cumprod()

    return BacktestResult(
        equity=eq,
        returns=port_ret,
        weights=weights.iloc[warmup:],
        costs=costs.iloc[warmup:],
        signals=signals,
        metrics=performance_metrics(port_ret, weights.iloc[warmup:], costs.iloc[warmup:]),
        benchmark=bench,
    )


def performance_metrics(returns: pd.Series, weights: pd.DataFrame | None = None,
                        costs: pd.Series | None = None) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {}
    eq = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    mdd = drawdown(eq).min()
    out = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": (returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan,
        "Sortino": (returns.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Hit rate": float((returns > 0).sum() / max((returns != 0).sum(), 1)),
    }
    if weights is not None:
        out["Avg gross leverage"] = float(weights.abs().sum(axis=1).mean())
        out["Annual turnover"] = float(weights.diff().abs().sum(axis=1).mean() * TRADING_DAYS)
    if costs is not None:
        out["Annual cost drag"] = float(costs.mean() * TRADING_DAYS)
    return out
