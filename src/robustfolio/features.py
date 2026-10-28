"""What a forecasting model sees: features of each asset on a given day, and the target it should predict.

Everything is computed from the returns *before* day t only, so a model trained on these features
cannot peek into the future. R is a (T x d) array of daily returns in %.

Starter features (classic, simple, and weak: improving them is part of the project)
  mom_1m     average daily return over the last month (21 trading days)
  mom_3m     average daily return over the last 3 months (63 days)
  mom_12_1   average daily return from 12 months ago to 1 month ago ("momentum")
  vol_3m     daily volatility over the last 3 months

Target
  the average daily return of each asset over the next `horizon` days (default 21: one month).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["mom_1m", "mom_3m", "mom_12_1", "vol_3m"]
LOOKBACK = 252              # days of history needed to compute the features


def features_at(R: np.ndarray, t: int) -> np.ndarray:
    """Features of every asset on day t, from R[t - LOOKBACK : t] only. Shape (d, len(FEATURES))."""
    if t < LOOKBACK:
        raise ValueError(f"need {LOOKBACK} days of history, got {t}")
    W = R[t - LOOKBACK:t]
    return np.column_stack([
        W[-21:].mean(0),
        W[-63:].mean(0),
        W[:-21].mean(0),
        W[-63:].std(0),
    ])


def target_at(R: np.ndarray, t: int, horizon: int = 21) -> np.ndarray:
    """What happened next: average daily return of each asset over R[t : t + horizon]. Shape (d,)."""
    return R[t:t + horizon].mean(0)


def training_set(R: np.ndarray, horizon: int = 21, step: int = 21):
    """All (features, target) pairs whose target is fully known by the end of R. Assets are stacked.

    Samples are taken every `step` days, starting when the features become available. With
    step >= horizon the targets of consecutive samples do not overlap.
    """
    T = len(R)
    days = list(range(LOOKBACK, T - horizon + 1, step))
    if not days:
        return np.empty((0, len(FEATURES))), np.empty(0)
    X = np.vstack([features_at(R, s) for s in days])
    y = np.concatenate([target_at(R, s, horizon) for s in days])
    return X, y


def make_panel(returns: pd.DataFrame, horizon: int = 21, step: int = 21) -> pd.DataFrame:
    """A tidy table for exploration: one row per (date, ticker), the features, and the target.

    The target is NaN for the last dates, where the future is not known yet.
    """
    R = returns.to_numpy(dtype=float)
    rows = []
    for t in range(LOOKBACK, len(R), step):
        F = features_at(R, t)
        y = target_at(R, t, horizon) if t + horizon <= len(R) else np.full(R.shape[1], np.nan)
        for j, ticker in enumerate(returns.columns):
            rows.append([returns.index[t], ticker, *F[j], y[j]])
    return pd.DataFrame(rows, columns=["date", "ticker", *FEATURES, "target"])
