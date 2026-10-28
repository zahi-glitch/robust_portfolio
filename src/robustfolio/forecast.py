"""Forecasters: predict each asset's average daily return over the next month.

Every forecaster has a `name` and a method `predict(R_past)`, where R_past holds only the returns up to
today (a T x d array, in %). It returns one number per asset. Because the past is all it receives, it
cannot use the future by mistake.

  HistoricalMean      the baseline that Markowitz uses: the average of the last two years
  SklearnForecaster   any scikit-learn regressor, retrained every month on the features of
                      features.py, all assets pooled together

Example
-------
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ridge = SklearnForecaster(make_pipeline(StandardScaler(), Ridge(alpha=10.0)), name="ridge")
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone

from .features import features_at, training_set


class HistoricalMean:
    def __init__(self, window: int = 504, name: str = "historical mean"):
        self.window, self.name = window, name

    def predict(self, R_past: np.ndarray) -> np.ndarray:
        return R_past[-self.window:].mean(0)


class SklearnForecaster:
    def __init__(self, model, name: str, horizon: int = 21, step: int = 21, train_days: int | None = None):
        """train_days: use only the last `train_days` days to train (None: all the history so far)."""
        self.model, self.name = model, name
        self.horizon, self.step, self.train_days = horizon, step, train_days

    def predict(self, R_past: np.ndarray) -> np.ndarray:
        R_train = R_past if self.train_days is None else R_past[-self.train_days:]
        X, y = training_set(R_train, self.horizon, self.step)
        model = clone(self.model).fit(X, y)
        return model.predict(features_at(R_past, len(R_past)))
