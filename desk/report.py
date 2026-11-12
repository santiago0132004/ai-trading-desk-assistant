"""Renders the daily desk report as HTML (for GitHub Pages), Markdown and JSON."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from .agent import DeskRun  # noqa: E402
from .risk import drawdown  # noqa: E402

NAVY, TEAL, GREY, RED = "#0b2545", "#1b998b", "#8d99ae", "#d1495b"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def charts(run: DeskRun) -> dict[str, str]:
    bt = run.backtest
    out = {}

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(bt.equity.index, bt.equity, color=NAVY, lw=1.6, label="Strategy (net of costs)")
    if bt.benchmark is not None:
        b = bt.benchmark / bt.benchmark.iloc[0]
        ax.plot(b.index, b, color=GREY, lw=1.1, label="SPY buy & hold")
    ax.set_yscale("log")
    ax.set_title("Growth of $1", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    out["equity"] = _png(fig)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    dd = drawdown(bt.equity)
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.35)
    ax.set_title("Drawdown", fontsize=10, loc="left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax)
    out["drawdown"] = _png(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    w = run.book["Target weight"].astype(float).sort_values()
    ax.barh(w.index, w, color=[TEAL if v >= 0 else RED for v in w])
    ax.axvline(0, color="black", lw=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Target book for next session", fontsize=10, loc="left")
    _style(ax)
    out["book"] = _png(fig)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    exp = bt.weights.iloc[-504:]
    classes = run.universe.asset_class
    for klass, color in (("fx", NAVY), ("equities", TEAL)):
        cols = [c for c in exp.columns if classes.get(c) == klass]
        if cols:
            ax.plot(exp.index, exp[cols].sum(axis=1), color=color, lw=1.2, label=f"Net {klass.upper()}")
    ax.plot(exp.index, exp.abs().sum(axis=1), color=GREY, lw=1, ls="--", label="Gross")
    ax.set_title("Exposure, last 2 years", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    _style(ax)
    out["exposure"] = _png(fig)
    return out


def _pct(v, digits=1, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.{digits}%}" if signed else f"{v:.{digits}%}"


def write_report(run: DeskRun, out_dir: str | Path, aum: float = 1_000_000) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=select_autoescape(["html"]))
    env.filters["pct"] = _pct
    env.filters["num"] = lambda v, d=2: "–" if v is None or not np.isfinite(float(v)) else f"{float(v):,.{d}f}"

    m = run.backtest.metrics
    metrics = [
        ("CAGR", _pct(m.get("CAGR"))), ("Volatility", _pct(m.get("Volatility"))),
        ("Sharpe", f"{m.get('Sharpe', float('nan')):.2f}"), ("Sortino", f"{m.get('Sortino', float('nan')):.2f}"),
        ("Max drawdown", _pct(m.get("Max drawdown"))), ("Calmar", f"{m.get('Calmar', float('nan')):.2f}"),
        ("Hit rate", _pct(m.get("Hit rate"))), ("Cost drag / yr", _pct(m.get("Annual cost drag"), 2)),
    ]
    html = env.get_template("report.html.j2").render(
        run=run, charts=charts(run), metrics=metrics, aum=aum,
        snapshot=run.snapshot, book=run.book, risk=run.risk,
        start=run.backtest.equity.index[0].date(),
        paragraphs=[p for p in run.commentary.split("\n") if p.strip()],
    )
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "desk_run.json").write_text(json.dumps(run.to_json(), indent=2, default=str), encoding="utf-8")
    (out / "morning_note.md").write_text(_markdown(run), encoding="utf-8")
    return out / "index.html"


def _markdown(run: DeskRun) -> str:
    lines = [f"# Morning note – {run.as_of.date()}", ""]
    if run.demo:
        lines += ["> Demo run on synthetic data.", ""]
    lines += [run.commentary, "", "## Trade blotter", "",
              "| Instrument | Action | Target | Change | Trade (USD) |", "|---|---|---|---|---|"]
    for n, r in run.book.iterrows():
        lines.append(f"| {n} | {r['Action']} | {r['Target weight']:+.1%} | {r['Change']:+.1%} | "
                     f"{r['Trade (USD)']:+,.0f} |")
    lines += ["", "## Control log", ""]
    lines += [f"- **{s.status}** {s.name}: {s.detail}" for s in run.audit]
    return "\n".join(lines) + "\n"
