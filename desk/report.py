"""Renders the daily desk report as HTML (for GitHub Pages), Markdown and JSON."""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from .agent import DeskRun  # noqa: E402
from .backtest import performance_metrics  # noqa: E402
from .risk import drawdown  # noqa: E402

NAVY, TEAL, GREY, RED, GOLD = "#0b2545", "#1b998b", "#9aa5b8", "#d1495b", "#c9a227"
DEFAULT_LIMITS = {"max_gross_leverage": 1.5, "target_vol_annual": 0.10, "drawdown_brake": 0.12, "rebalance_band": 0.02}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "axes.edgecolor": "#c9d0dc",
    "axes.labelcolor": "#3b4458",
    "xtick.color": "#5c677d",
    "ytick.color": "#5c677d",
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(length=0)


def charts(run: DeskRun) -> dict[str, str]:
    bt = run.backtest
    out = {}

    # growth of $1, log scale with dollar labels
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(bt.equity.index, bt.equity, color=NAVY, lw=1.8, label="Strategy (net of costs)")
    if bt.benchmark is not None:
        b = bt.benchmark / bt.benchmark.iloc[0]
        ax.plot(b.index, b, color=GREY, lw=1.1, label="SPY buy & hold")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=2))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_title("Growth of $1", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style(ax)
    out["equity"] = _png(fig)

    # drawdowns, strategy vs benchmark
    fig, ax = plt.subplots(figsize=(10, 2.4))
    dd = drawdown(bt.equity)
    if bt.benchmark is not None:
        ax.plot(bt.benchmark.index, drawdown(bt.benchmark), color=GREY, lw=0.8, alpha=0.8, label="SPY")
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.35, label="Strategy")
    ax.set_title("Drawdown", loc="left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=2)
    _style(ax)
    out["drawdown"] = _png(fig)

    # target book
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    w = run.book["Target weight"].astype(float).sort_values()
    bars = ax.barh(w.index, w, color=[TEAL if v >= 0 else RED for v in w], height=0.62)
    for bar, v in zip(bars, w):
        ax.text(v + (0.008 if v >= 0 else -0.008), bar.get_y() + bar.get_height() / 2, f"{v:+.0%}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color="#3b4458")
    lim = max(0.35, w.abs().max() + 0.08)
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color="#3b4458", lw=0.7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    out["book"] = _png(fig)

    # exposure
    fig, ax = plt.subplots(figsize=(7, 3.6))
    exp = bt.weights.iloc[-504:]
    classes = run.universe.asset_class
    for klass, color in (("fx", NAVY), ("equities", TEAL)):
        cols = [c for c in exp.columns if classes.get(c) == klass]
        if cols:
            ax.plot(exp.index, exp[cols].sum(axis=1), color=color, lw=1.4, label=f"Net {klass.upper()}")
    ax.plot(exp.index, exp.abs().sum(axis=1), color=GREY, lw=1, ls="--", label="Gross")
    ax.axhline(0, color="#3b4458", lw=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
    _style(ax)
    out["exposure"] = _png(fig)
    return out


def _pct(v, digits=1, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.{digits}%}" if signed else f"{v:.{digits}%}"


def _num(v, digits=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    return f"{v:,.{digits}f}" if np.isfinite(v) else "–"


def _metric_rows(run: DeskRun) -> list[tuple[str, str, str]]:
    m = run.backtest.metrics
    b = {}
    if run.backtest.benchmark is not None:
        b = performance_metrics(run.backtest.benchmark.pct_change().fillna(0.0))
    rows = [("CAGR", "pct"), ("Volatility", "pct"), ("Sharpe", "num"), ("Sortino", "num"),
            ("Max drawdown", "pct"), ("Calmar", "num"), ("Hit rate", "pct")]
    fmt = {"pct": _pct, "num": _num}
    return [(k, fmt[t](m.get(k)), fmt[t](b.get(k)) if b else "–") for k, t in rows]


def write_report(run: DeskRun, out_dir: str | Path, aum: float = 1_000_000,
                 author: dict | None = None, limits: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=select_autoescape(["html"]))
    env.filters["pct"] = _pct
    env.filters["num"] = _num

    author = author or {}
    initials = "".join(p[0] for p in author.get("name", "").split()[:2]).upper()
    bogota = datetime.now(timezone.utc) - timedelta(hours=5)
    m = run.backtest.metrics
    html = env.get_template("report.html.j2").render(
        run=run, charts=charts(run), metric_rows=_metric_rows(run), aum=aum,
        snapshot=run.snapshot, book=run.book, risk=run.risk,
        start=run.backtest.equity.index[0].date(),
        paragraphs=[p for p in run.commentary.split("\n") if p.strip()],
        author=author, initials=initials, limits={**DEFAULT_LIMITS, **(limits or {})},
        updated=bogota.strftime("%b %d, %Y · %H:%M") + " Bogotá",
        n_trades=int((run.book["Action"] != "HOLD").sum()),
        n_warn=sum(s.status != "PASS" for s in run.audit),
        cost_drag=_pct(m.get("Annual cost drag"), 2),
        turnover=_num(m.get("Annual turnover"), 1),
    )
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "desk_run.json").write_text(json.dumps(run.to_json(), indent=2, default=str), encoding="utf-8")
    (out / "morning_note.md").write_text(_markdown(run, author), encoding="utf-8")
    return out / "index.html"


def _markdown(run: DeskRun, author: dict) -> str:
    lines = [f"# Morning note – {run.as_of.date()}", ""]
    if author.get("name"):
        lines += [f"*AI Trading Desk Assistant, built by {author['name']}*", ""]
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
