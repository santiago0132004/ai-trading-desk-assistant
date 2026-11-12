"""Market data loading: Yahoo Finance for live runs, a seeded synthetic generator for demos and tests."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Universe:
    names: list[str]          # display names, e.g. "USDCOP"
    tickers: dict[str, str]   # display name -> Yahoo ticker
    asset_class: dict[str, str]  # display name -> "fx" | "equities"

    @classmethod
    def from_config(cls, cfg: dict) -> "Universe":
        tickers, classes = {}, {}
        for klass, items in cfg["universe"].items():
            for name, ticker in items.items():
                tickers[name] = ticker
                classes[name] = klass
        return cls(list(tickers), tickers, classes)


def load_prices(universe: Universe, start: str, demo: bool = False) -> pd.DataFrame:
    """Daily close prices, one column per instrument (display names)."""
    if demo:
        return synthetic_prices(universe, start)

    import yfinance as yf  # imported lazily so tests and demo mode do not need network

    raw = yf.download(
        list(universe.tickers.values()), start=start, auto_adjust=True, progress=False
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    inverse = {v: k for k, v in universe.tickers.items()}
    close = close.rename(columns=inverse)

    missing = [n for n in universe.names if n not in close or close[n].dropna().empty]
    if missing:
        log.warning("No data for %s; dropping them from the run", missing)
    keep = [n for n in universe.names if n not in missing]
    if not keep:
        raise RuntimeError("No market data downloaded. Check your internet connection or use --demo.")
    # FX trades on days equities do not: forward-fill short gaps only
    return close[keep].sort_index().ffill(limit=3).dropna(how="all")


def synthetic_prices(universe: Universe, start: str, seed: int = 7) -> pd.DataFrame:
    """Regime-switching random walks so the whole pipeline runs offline. Not real market data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    n = len(dates)
    base = {"fx": (0.0, 0.09), "equities": (0.07, 0.20)}
    out = {}
    for name in universe.names:
        mu, vol = base[universe.asset_class[name]]
        # slowly drifting trend component gives momentum something to find
        drift = np.cumsum(rng.normal(0, 0.02, n)) / np.sqrt(n) * 0.5 + mu
        regime = np.where(rng.random(n) < 0.02, 2.0, 1.0)
        regime = pd.Series(regime).rolling(20, min_periods=1).max().to_numpy()
        rets = drift / 252 + rng.standard_t(5, n) * vol / np.sqrt(252) * regime * 0.8
        out[name] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(out, index=dates)
