"""Baseline: does a simple ML forecast help a portfolio, and does robustness protect against its errors?

Runs on the bundled prices (10 US stocks, 1990-2018) and writes results/baseline.md and results/wealth.png.

    python scripts/run_baseline.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from robustfolio import (HistoricalMean, SklearnForecaster, Strategy, backtest, forecast_scores,  # noqa: E402
                         load_prices, returns_pct, walk_forward_forecasts, wealth)

OUT = Path(__file__).resolve().parents[1] / "results"
START, STEP = 756, 21          # first decision after 3 years of data; then every month


def pct(v):
    return f"{100 * v:.1f}%"


def main():
    t0 = time.time()
    returns = returns_pct(load_prices())
    print(f"{returns.shape[1]} stocks, {returns.index[0].date()} to {returns.index[-1].date()}")

    forecasters = [
        HistoricalMean(),
        SklearnForecaster(make_pipeline(StandardScaler(), Ridge(alpha=10.0)), name="ridge"),
    ]
    forecasts = {f.name: walk_forward_forecasts(f, returns, START, STEP) for f in forecasters}

    strategies = {
        "equal": Strategy("Equal weight", "equal"),
        "minvar": Strategy("Minimum variance", "minvar"),
        "markowitz": Strategy("Markowitz", "markowitz"),
        "robust": Strategy("Robust mean-variance", "robust_mv"),
    }
    runs = [backtest(strategies["equal"], returns, start=START, step=STEP),
            backtest(strategies["minvar"], returns, start=START, step=STEP)]
    for f in forecasts.values():
        for key in ("markowitz", "robust"):
            runs.append(backtest(strategies[key], returns, f, start=START, step=STEP))

    OUT.mkdir(exist_ok=True)
    lines = ["# Baseline results", "",
             f"Bundled prices, {returns.shape[1]} US stocks; out of sample from {runs[0].dates[0].date()} "
             f"to {runs[0].dates[-1].date()}. Monthly rebalance, long-only, 10 bps costs.", "",
             "## 1. Forecast quality (next month's average daily return of each stock)", "",
             "| forecaster | R² vs zero | rank IC | hit rate |", "|---|---|---|---|"]
    for f in forecasts.values():
        s = forecast_scores(f)
        lines.append(f"| {f.name} | {s['R2 vs zero']:.3f} | {s['rank IC']:.3f} | {pct(s['hit rate'])} |")
    lines += ["", "## 2. Portfolios", "",
              "| strategy | annual return | volatility | Sharpe | max drawdown | turnover |",
              "|---|---|---|---|---|---|"]
    for r in runs:
        m = r.metrics
        lines.append(f"| {r.name} | {pct(m['annual return'])} | {pct(m['annual volatility'])} | "
                     f"{m['Sharpe (rf = 0)']:.2f} | {pct(m['max drawdown'])} | {m['turnover / rebalance']:.2f} |")
    (OUT / "baseline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        ax.plot(wealth(r), lw=2.2 if r.name.startswith("Robust") else 1.2, label=r.name)
    ax.set_yscale("log"); ax.set_ylabel("value of $1 (log scale)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Walk-forward backtest, monthly rebalance, 10 bps costs")
    fig.tight_layout(); fig.savefig(OUT / "wealth.png", dpi=130)
    print(f"\nwritten to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
