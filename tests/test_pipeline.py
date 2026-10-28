import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from robustfolio import (HistoricalMean, SklearnForecaster, Strategy, backtest, forecast_scores, load_prices,
                         returns_pct, walk_forward_forecasts)
from robustfolio.features import FEATURES, LOOKBACK, features_at, make_panel, training_set
from robustfolio.portfolio import (markowitz_objective, recentre, robust_mean_objective,
                                   robust_mean_variance_objective)


@pytest.fixture(scope="module")
def returns():
    return returns_pct(load_prices(start="2012-01-01"))


def test_data(returns):
    assert returns.shape[1] == 10 and np.isfinite(returns.to_numpy()).all()


def test_recentre_sets_mean_keeps_covariance(returns):
    R = returns.to_numpy()[:504]
    mu = np.linspace(-0.1, 0.1, R.shape[1])
    Rc = recentre(R, mu)
    assert np.allclose(Rc.mean(0), mu) and np.allclose(np.cov(Rc.T), np.cov(R.T))


def test_features_use_only_the_past(returns):
    """Changing the future must not change today's features or the training set."""
    R = returns.to_numpy().copy()
    t = 600
    F, (X, y) = features_at(R, t), training_set(R[:t])
    R[t:] = 99.0
    assert np.array_equal(F, features_at(R, t))
    X2, y2 = training_set(R[:t])
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert F.shape == (10, len(FEATURES)) and len(X) == len(y) == 10 * len(range(LOOKBACK, t - 21 + 1, 21))


def test_panel(returns):
    panel = make_panel(returns)
    assert list(panel.columns) == ["date", "ticker", *FEATURES, "target"]
    assert panel["target"].isna().sum() <= 10


@pytest.mark.parametrize("objective", [
    lambda R, x: markowitz_objective(R, x, 0.05),
    lambda R, x: robust_mean_objective(R, x, 0.05),
    lambda R, x: robust_mean_variance_objective(R, x, 0.05, 0.05, 0.1),
])
def test_danskin_gradients(returns, objective):
    """Gradients from the worst-case reweighting (Danskin) match finite differences."""
    W, x, e = returns.to_numpy()[:504], np.random.default_rng(0).dirichlet(np.ones(10)), 1e-6
    _, g = objective(W, x)
    fd = [(objective(W, x + e * u)[0] - objective(W, x - e * u)[0]) / (2 * e) for u in np.eye(10)]
    assert np.abs(g - fd).max() < 1e-6


def test_end_to_end(returns):
    ml = SklearnForecaster(LinearRegression(), name="linear")
    for forecaster in (HistoricalMean(), ml):
        f = walk_forward_forecasts(forecaster, returns, start=756, step=63)
        assert f.predicted.shape == f.realised.shape == (len(f.days), 10)
        assert np.isfinite(list(forecast_scores(f).values())).all()
        for kind in ("equal", "minvar", "markowitz", "robust_mv"):
            res = backtest(Strategy(kind, kind), returns, f, start=756, step=63)
            assert np.allclose(res.weights.sum(1), 1) and (res.weights >= -1e-12).all()
            assert res.returns.size == len(returns) - 756 and np.isfinite(list(res.metrics.values())).all()


def test_forecast_moves_markowitz_more_than_robust(returns):
    """The point of the project: the robust rule reacts less to a change in the forecast."""
    R = returns.to_numpy()[:504]
    base, bumped = R.mean(0), R.mean(0).copy()
    bumped[0] += 0.05                                  # the model becomes optimistic about one stock
    move = {}
    for kind in ("markowitz", "robust_mv"):
        s = Strategy(kind, kind)
        move[kind] = np.abs(s.weights(R, bumped) - s.weights(R, base)).sum()
    assert move["robust_mv"] < move["markowitz"]
