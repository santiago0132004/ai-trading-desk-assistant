"""Command line entry point.

    python -m desk run              # full daily workflow with live data -> docs/index.html
    python -m desk run --demo       # same workflow on synthetic data (no internet needed)
    python -m desk backtest --demo  # print backtest metrics only
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .agent import DeskAgent
from .backtest import run_backtest
from .data import Universe, clean_bad_ticks, load_prices
from .report import write_report


def main() -> None:
    p = argparse.ArgumentParser(prog="desk", description="AI Trading Desk Assistant")
    p.add_argument("command", choices=["run", "backtest"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="docs", help="output folder for the report")
    p.add_argument("--demo", action="store_true", help="use synthetic data instead of Yahoo Finance")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(levelname)-7s %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.command == "backtest":
        uni = Universe.from_config(cfg)
        prices = load_prices(uni, cfg["start_date"], demo=a.demo)
        fx_cols = [n for n in prices if uni.asset_class[n] in cfg["agent"].get("clean_asset_classes", ["fx"])]
        prices, _ = clean_bad_ticks(prices, fx_cols, threshold=cfg["agent"].get("spike_sigma", 6.0))
        res = run_backtest(prices, uni.asset_class, cfg, cfg.get("benchmark"))
        print(f"\nBacktest {res.equity.index[0].date()} -> {res.equity.index[-1].date()}")
        for k, v in res.metrics.items():
            pct = k not in {"Sharpe", "Sortino", "Calmar", "Avg gross leverage"}
            print(f"  {k:<20} {v:>9.2%}" if pct else f"  {k:<20} {v:>9.2f}")
        return

    run = DeskAgent(cfg, demo=a.demo).run()
    path = write_report(run, a.out, aum=cfg["agent"]["aum_usd"], author=cfg.get("author"), limits=cfg["risk"])
    print(f"\nReport written to {path}")
    print("\n" + run.commentary)


if __name__ == "__main__":
    main()
