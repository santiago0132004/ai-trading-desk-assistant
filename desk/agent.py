"""The desk agent: runs the daily workflow as a checklist, logs every step, and writes the morning note.

Each step returns a status (PASS / WARN / FAIL) that lands in an audit trail, the way a real desk
documents its controls. The written commentary comes from an LLM when ANTHROPIC_API_KEY is set,
and from deterministic rules otherwise, so the run never depends on an external API.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .data import Universe, load_prices
from .risk import drawdown, ex_ante_vol, historical_var

log = logging.getLogger(__name__)


@dataclass
class Step:
    name: str
    status: str
    detail: str


@dataclass
class DeskRun:
    as_of: pd.Timestamp
    demo: bool
    prices: pd.DataFrame
    universe: Universe
    backtest: BacktestResult
    snapshot: pd.DataFrame
    book: pd.DataFrame
    risk: dict
    commentary: str
    commentary_source: str
    audit: list[Step] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "demo_data": self.demo,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "snapshot": self.snapshot.reset_index().to_dict(orient="records"),
            "book": self.book.reset_index().to_dict(orient="records"),
            "risk": self.risk,
            "backtest_metrics": self.backtest.metrics,
            "commentary": self.commentary,
            "commentary_source": self.commentary_source,
            "audit": [s.__dict__ for s in self.audit],
        }


class DeskAgent:
    def __init__(self, cfg: dict, demo: bool = False):
        self.cfg = cfg
        self.demo = demo
        self.universe = Universe.from_config(cfg)
        self.audit: list[Step] = []

    def _log(self, name: str, status: str, detail: str) -> None:
        self.audit.append(Step(name, status, detail))
        getattr(log, "warning" if status != "PASS" else "info")("[%s] %s: %s", status, name, detail)

    # ---- workflow -----------------------------------------------------------------------------
    def run(self) -> DeskRun:
        prices = self._load_data()
        self._data_quality(prices)
        bt = run_backtest(prices, self.universe.asset_class, self.cfg, self.cfg.get("benchmark"))
        self._log("Backtest", "PASS",
                  f"{len(bt.returns)} days, Sharpe {bt.metrics.get('Sharpe', float('nan')):.2f}, "
                  f"max drawdown {bt.metrics.get('Max drawdown', float('nan')):.1%}")
        snapshot = self._market_snapshot(prices, bt)
        book = self._target_book(prices, bt)
        risk = self._risk_checks(prices, bt, book)
        commentary, source = self._commentary(prices.index[-1], snapshot, book, risk, bt)
        return DeskRun(prices.index[-1], self.demo, prices, self.universe, bt, snapshot, book,
                       risk, commentary, source, self.audit)

    def _load_data(self) -> pd.DataFrame:
        prices = load_prices(self.universe, self.cfg["start_date"], demo=self.demo)
        dropped = set(self.universe.names) - set(prices.columns)
        self.universe = Universe(
            [n for n in self.universe.names if n in prices],
            {n: t for n, t in self.universe.tickers.items() if n in prices},
            {n: c for n, c in self.universe.asset_class.items() if n in prices},
        )
        status = "WARN" if dropped else "PASS"
        src = "synthetic demo data" if self.demo else "Yahoo Finance"
        detail = f"{prices.shape[1]} instruments, {len(prices)} days from {src}"
        if dropped:
            detail += f"; dropped: {', '.join(sorted(dropped))}"
        self._log("Load market data", status, detail)
        return prices

    def _data_quality(self, prices: pd.DataFrame) -> None:
        a = self.cfg["agent"]
        last = prices.index[-1]
        lag = np.busday_count(last.date(), pd.Timestamp.today().date())
        self._log("Data freshness", "WARN" if lag > a["stale_data_days"] else "PASS",
                  f"last close {last.date()} ({lag} business days old)")

        recent = prices.iloc[-252:]
        gaps = recent.isna().sum()
        bad = gaps[gaps > 5]
        self._log("Missing values", "WARN" if len(bad) else "PASS",
                  "no material gaps in the last year" if bad.empty
                  else "gaps in " + ", ".join(f"{k} ({v}d)" for k, v in bad.items()))

        rets = prices.pct_change().iloc[-252:]
        z = (rets - rets.mean()) / rets.std()
        jumps = (z.abs() > a["jump_sigma"]).sum()
        jumps = jumps[jumps > 0]
        self._log("Outlier moves", "WARN" if len(jumps) else "PASS",
                  "no moves beyond {}σ".format(a["jump_sigma"]) if jumps.empty
                  else "check " + ", ".join(f"{k} ({v}x)" for k, v in jumps.items()))

    def _market_snapshot(self, prices: pd.DataFrame, bt: BacktestResult) -> pd.DataFrame:
        rets = prices.pct_change()
        vol_1m = rets.rolling(21).std() * np.sqrt(252)
        rows = {}
        for n in prices.columns:
            p = prices[n].dropna()
            v = vol_1m[n].dropna()
            rows[n] = {
                "Class": self.universe.asset_class[n].upper() if n in self.universe.asset_class else "",
                "Last": p.iloc[-1],
                "1D": p.iloc[-1] / p.iloc[-2] - 1,
                "1W": p.iloc[-1] / p.iloc[-6] - 1,
                "1M": p.iloc[-1] / p.iloc[-22] - 1,
                "YTD": p.iloc[-1] / p[p.index.year < p.index[-1].year].iloc[-1] - 1
                if (p.index.year < p.index[-1].year).any() else np.nan,
                "Vol 1M": v.iloc[-1],
                "Vol pctile": float((v.iloc[-756:] <= v.iloc[-1]).mean()),
                "Momentum": bt.signals["momentum"][n].iloc[-1],
                "Trend": bt.signals["trend"][n].iloc[-1],
                "Mean rev.": bt.signals["mean_reversion"][n].iloc[-1],
                "Signal": bt.signals["composite"][n].iloc[-1],
            }
        df = pd.DataFrame(rows).T
        df.index.name = "Instrument"
        self._log("Market snapshot", "PASS", f"{len(df)} instruments scored")
        return df

    def _target_book(self, prices: pd.DataFrame, bt: BacktestResult) -> pd.DataFrame:
        aum = self.cfg["agent"]["aum_usd"]
        w_new = bt.weights.iloc[-1]
        w_old = bt.weights.iloc[-2] if len(bt.weights) > 1 else w_new * 0
        book = pd.DataFrame({
            "Current weight": w_old,
            "Target weight": w_new,
            "Change": w_new - w_old,
        })
        book["Trade (USD)"] = book["Change"] * aum
        book["Action"] = np.select(
            [book["Change"] > 0.005, book["Change"] < -0.005], ["BUY", "SELL"], "HOLD")
        book["Side"] = np.select([w_new > 0.001, w_new < -0.001], ["LONG", "SHORT"], "FLAT")
        book.index.name = "Instrument"
        n_trades = int((book["Action"] != "HOLD").sum())
        self._log("Target book", "PASS", f"{n_trades} trades proposed for next session")
        return book

    def _risk_checks(self, prices: pd.DataFrame, bt: BacktestResult, book: pd.DataFrame) -> dict:
        r = self.cfg["risk"]
        w = book["Target weight"]
        window = prices.pct_change().iloc[-r["vol_window"]:]
        dd = drawdown(bt.equity)
        risk = {
            "Gross exposure": float(w.abs().sum()),
            "Net exposure": float(w.sum()),
            "Largest position": float(w.abs().max()),
            "Ex-ante vol": ex_ante_vol(w, window),
            "1D VaR 95%": historical_var(w, prices.pct_change().iloc[-500:]),
            "Current drawdown": float(dd.iloc[-1]),
            "Drawdown brake active": bool(dd.iloc[-1] < -r["drawdown_brake"]),
        }
        checks = [
            ("Gross limit", risk["Gross exposure"] <= r["max_gross_leverage"] + 1e-9,
             f"gross {risk['Gross exposure']:.2f}x vs limit {r['max_gross_leverage']:.2f}x"),
            ("Concentration limit", risk["Largest position"] <= r["max_weight_per_asset"] + 1e-9,
             f"largest {risk['Largest position']:.1%} vs limit {r['max_weight_per_asset']:.0%}"),
            ("Volatility target", risk["Ex-ante vol"] <= r["target_vol_annual"] * 1.25,
             f"ex-ante vol {risk['Ex-ante vol']:.1%} vs target {r['target_vol_annual']:.0%}"),
            ("Drawdown brake", not risk["Drawdown brake active"],
             f"drawdown {risk['Current drawdown']:.1%} (brake at -{r['drawdown_brake']:.0%})"),
        ]
        for name, ok, detail in checks:
            self._log(name, "PASS" if ok else "WARN", detail)
        return risk

    # ---- commentary ---------------------------------------------------------------------------
    def _commentary(self, as_of, snapshot, book, risk, bt) -> tuple[str, str]:
        facts = {
            "as_of": str(as_of.date()),
            "snapshot": snapshot[["1D", "1W", "1M", "Vol 1M", "Vol pctile", "Signal"]]
            .round(4).to_dict(orient="index"),
            "book": book[["Target weight", "Change", "Action"]].round(4).to_dict(orient="index"),
            "risk": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in risk.items()},
            "backtest": {k: round(v, 3) for k, v in bt.metrics.items() if np.isfinite(v)},
        }
        if self.cfg["agent"].get("llm_commentary") and os.getenv("ANTHROPIC_API_KEY"):
            try:
                text = self._llm_commentary(facts)
                self._log("Commentary", "PASS", "written by LLM from the run's structured output")
                return text, "LLM"
            except Exception as exc:  # never let the note break the run
                self._log("Commentary", "WARN", f"LLM unavailable ({exc.__class__.__name__}); used rules")
        text = self._rule_commentary(snapshot, book, risk)
        if not self.audit or self.audit[-1].name != "Commentary":
            self._log("Commentary", "PASS", "rule-based note (set ANTHROPIC_API_KEY for LLM commentary)")
        return text, "rules"

    def _llm_commentary(self, facts: dict) -> str:
        import anthropic

        client = anthropic.Anthropic()
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        prompt = (
            "You are a sell-side trading desk analyst writing the morning note for an FX and "
            "equities book. Using ONLY the JSON facts below, write 3 short paragraphs: (1) what "
            "moved and the volatility regime, (2) the proposed trades and why the signals support "
            "them, (3) risk status and anything to watch. Be specific with numbers, no hype, no "
            "recommendations beyond the model's book, under 220 words.\n\n"
            + json.dumps(facts, default=str)
        )
        msg = client.messages.create(model=model, max_tokens=600,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    @staticmethod
    def _rule_commentary(snapshot: pd.DataFrame, book: pd.DataFrame, risk: dict) -> str:
        movers = snapshot["1D"].astype(float).abs().sort_values(ascending=False).index[:3]
        mv = ", ".join(f"{n} {snapshot.loc[n, '1D']:+.2%}" for n in movers)
        hot = snapshot[snapshot["Vol pctile"].astype(float) > 0.8].index.tolist()
        vol_txt = (f"Volatility is elevated (above the 80th percentile of the last 3 years) in "
                   f"{', '.join(hot)}." if hot else "Volatility is within normal ranges across the book.")

        trades = book[book["Action"] != "HOLD"]
        if trades.empty:
            trade_txt = "No rebalancing needed: the target book is unchanged."
        else:
            parts = []
            for n, row in trades.iterrows():
                s = float(snapshot.loc[n, "Signal"])
                parts.append(f"{row['Action']} {n} ({row['Change']:+.1%} to {row['Target weight']:+.1%}, "
                             f"signal {s:+.2f})")
            trade_txt = "Proposed trades: " + "; ".join(parts) + "."

        longs = book[book["Side"] == "LONG"].index.tolist()
        shorts = book[book["Side"] == "SHORT"].index.tolist()
        pos_txt = (f"The book is long {', '.join(longs) or 'nothing'}"
                   + (f" and short {', '.join(shorts)}" if shorts else "") + ".")

        risk_txt = (f"Gross exposure {risk['Gross exposure']:.2f}x, net {risk['Net exposure']:+.2f}x, "
                    f"ex-ante volatility {risk['Ex-ante vol']:.1%} and 1-day 95% VaR "
                    f"{risk['1D VaR 95%']:.2%} of capital. Strategy drawdown is "
                    f"{risk['Current drawdown']:.1%}"
                    + (", so the drawdown brake is ON and exposure is halved." if risk["Drawdown brake active"]
                       else "; all limits are respected."))
        return f"Biggest moves today: {mv}. {vol_txt}\n\n{pos_txt} {trade_txt}\n\n{risk_txt}"
