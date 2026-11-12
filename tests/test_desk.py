from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from desk.agent import DeskAgent
from desk.backtest import performance_metrics, run_backtest
from desk.data import Universe, synthetic_prices
from desk.report import write_report
from desk.risk import size_positions
from desk.signals import composite, mean_reversion, momentum, trend

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())


@pytest.fixture(scope="module")
def universe():
    return Universe.from_config(CFG)


@pytest.fixture(scope="module")
def prices(universe):
    return synthetic_prices(universe, "2018-01-01")


def test_signals_are_bounded(prices):
    sigs = composite(prices, CFG)
    for name, df in sigs.items():
        vals = df.dropna().to_numpy()
        assert np.all(vals <= 1 + 1e-9) and np.all(vals >= -1 - 1e-9), name


def test_trend_sign_on_rising_series():
    p = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=pd.bdate_range("2020-01-01", periods=300))
    assert trend(p, 50, 200)["A"].iloc[-1] == 1
    assert momentum(p * np.exp(np.random.default_rng(0).normal(0, 0.001, (300, 1))), [21, 63])["A"].iloc[-1] > 0


def test_mean_reversion_is_contrarian():
    p = pd.DataFrame({"A": np.linspace(100, 150, 60)}, index=pd.bdate_range("2020-01-01", periods=60))
    assert mean_reversion(p)["A"].iloc[-1] < 0  # straight-line rally -> overbought -> negative


def test_signals_have_no_lookahead(prices):
    """Changing future prices must not change today's signal."""
    cut = prices.index[-50]
    base = composite(prices, CFG)["composite"].loc[cut]
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= 1.5
    after = composite(shocked, CFG)["composite"].loc[cut]
    pd.testing.assert_series_equal(base, after)


def test_position_limits(prices, universe):
    sig = composite(prices, CFG)["composite"].iloc[-1]
    window = prices.pct_change().iloc[-63:]
    w = size_positions(sig, window, universe.asset_class, CFG)
    r = CFG["risk"]
    assert w.abs().max() <= r["max_weight_per_asset"] + 1e-9
    assert w.abs().sum() <= r["max_gross_leverage"] + 1e-9
    for name in w.index:  # equities are long-only
        if universe.asset_class[name] == "equities":
            assert w[name] >= 0


def test_backtest_runs_and_respects_limits(prices, universe):
    res = run_backtest(prices, universe.asset_class, CFG, "SPY")
    assert len(res.equity) > 500
    assert np.isfinite(res.metrics["Sharpe"])
    assert res.weights.abs().sum(axis=1).max() <= CFG["risk"]["max_gross_leverage"] + 1e-6
    assert (res.costs >= 0).all()


def test_metrics_on_known_series():
    r = pd.Series([0.01, -0.005] * 252)
    m = performance_metrics(r)
    assert m["Hit rate"] == pytest.approx(0.5)
    assert m["Max drawdown"] == pytest.approx(-0.005)


def test_full_agent_run_writes_report(tmp_path):
    run = DeskAgent(CFG, demo=True).run()
    assert {s.status for s in run.audit} <= {"PASS", "WARN", "FAIL"}
    assert run.commentary
    path = write_report(run, tmp_path)
    assert path.exists() and (tmp_path / "desk_run.json").exists() and (tmp_path / "morning_note.md").exists()
