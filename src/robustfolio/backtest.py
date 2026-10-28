"""Walk-forward evaluation, in two steps.

1. Forecasts (the ML question): on every rebalance day t, the forecaster sees the returns before t and
   predicts the next month. `walk_forward_forecasts` collects the predictions and what really happened;
   `forecast_scores` measures how good they are.
2. Portfolios (the decision question): on the same days, each strategy turns the last `window` days of
   returns, and the forecast, into weights, held for `step` days. `backtest` returns the daily returns
   after costs, and `performance` summarises them.

A forecast can be statistically weak and still useful for decisions, or the reverse: that is why both
are measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .portfolio import Strategy


def rebalance_days(T: int, start: int, step: int) -> list:
    return list(range(start, T, step))


# --------------------------------------------------------------------------- 1. forecasts
@dataclass
class Forecasts:
    name: str
    days: list                    # index of each rebalance day in the returns
    dates: pd.Index
    predicted: np.ndarray         # (k x d) forecast of the next month's average daily return, in %
    realised: np.ndarray          # (k x d) what it really was (the last row may cover fewer days)


def walk_forward_forecasts(forecaster, returns: pd.DataFrame, start: int = 756, step: int = 21) -> Forecasts:
    R = returns.to_numpy(dtype=float)
    days = rebalance_days(len(R), start, step)
    predicted = np.array([forecaster.predict(R[:t]) for t in days])
    realised = np.array([R[t:t + step].mean(0) for t in days])
    return Forecasts(forecaster.name, days, returns.index[days], predicted, realised)


def forecast_scores(f: Forecasts) -> dict:
    """How good are the forecasts?

    R2 vs zero   1 - sum (y - y_hat)^2 / sum y^2: above 0 means better than predicting "nothing happens".
                 Return forecasts are hard: values slightly above 0 are already good.
    rank IC      the average, over rebalance days, of the rank correlation between forecasts and outcomes
                 across assets: does the model order the assets correctly? About 0.05 is already useful.
    hit rate     how often the sign of the forecast is right.
    """
    y, yh = f.realised, f.predicted
    ics = [spearmanr(a, b)[0] for a, b in zip(yh, y) if np.std(a) > 0]
    return {
        "R2 vs zero": float(1 - np.sum((y - yh) ** 2) / np.sum(y ** 2)),
        "rank IC": float(np.nanmean(ics)) if ics else float("nan"),
        "hit rate": float(np.mean(np.sign(yh) == np.sign(y))),
    }


# --------------------------------------------------------------------------- 2. portfolios
@dataclass
class BacktestResult:
    name: str
    dates: pd.Index               # out-of-sample days
    returns: np.ndarray           # daily portfolio returns in %, after costs
    weights: np.ndarray           # weights at each rebalance (k x d)
    turnover: np.ndarray          # sum |x_new - x_old| at each rebalance
    metrics: dict = field(default_factory=dict)


def backtest(strategy: Strategy, returns: pd.DataFrame, forecasts: Forecasts | None = None,
             window: int = 504, start: int = 756, step: int = 21, cost_bps: float = 10.0,
             name: str | None = None) -> BacktestResult:
    """Walk-forward: on each rebalance day t, weights from R[t-window:t] (and the forecast made at t),
    held on R[t:t+step] (rebalanced daily to target). Costs: cost_bps per unit of turnover.
    """
    R = returns.to_numpy(dtype=float)
    days = rebalance_days(len(R), start, step)
    if forecasts is not None and forecasts.days != days:
        raise ValueError("the forecasts were made on different days: use the same start and step")
    d = R.shape[1]
    out, w_hist, turn = [], [], []
    x_prev = np.full(d, 1.0 / d)
    for k, t in enumerate(days):
        mu_hat = forecasts.predicted[k] if forecasts is not None and strategy.uses_forecast else None
        x = strategy.weights(R[t - window:t], mu_hat=mu_hat, x0=x_prev)
        tv = float(np.abs(x - x_prev).sum())
        period = R[t:t + step] @ x
        period[0] -= cost_bps / 100.0 * tv          # bps of the portfolio value, in % units
        out.append(period); w_hist.append(x); turn.append(tv)
        x_prev = x
    rets = np.concatenate(out)
    if name is None:
        name = strategy.name + (f" + {forecasts.name}" if forecasts is not None and strategy.uses_forecast else "")
    res = BacktestResult(name, returns.index[start:start + rets.size], rets, np.array(w_hist), np.array(turn))
    res.metrics = performance(rets, res.turnover)
    return res


def performance(rets_pct, turnover=None, days_per_year=252) -> dict:
    r = np.asarray(rets_pct) / 100.0
    wealth = np.cumprod(1.0 + r)
    years = r.size / days_per_year
    peak = np.maximum.accumulate(wealth)
    m = {
        "annual return": wealth[-1] ** (1.0 / years) - 1.0,
        "annual volatility": r.std() * np.sqrt(days_per_year),
        "Sharpe (rf = 0)": r.mean() / r.std() * np.sqrt(days_per_year),
        "max drawdown": float((wealth / peak - 1.0).min()),
    }
    if turnover is not None:
        m["turnover / rebalance"] = float(np.mean(turnover))
    return m


def wealth(result: BacktestResult) -> pd.Series:
    """Value of $1 invested at the start."""
    return pd.Series(np.cumprod(1.0 + result.returns / 100.0), index=result.dates, name=result.name)
