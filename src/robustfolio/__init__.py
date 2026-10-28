"""robustfolio: machine-learning return forecasts and KL-robust portfolios.

    data       prices and returns
    features   what a forecasting model sees, and what it should predict
    forecast   forecasters: the historical mean and any scikit-learn model
    portfolio  portfolio rules, from Markowitz (trusts the forecast) to KL-robust (distrusts it by rho)
    backtest   walk-forward evaluation of forecasts and portfolios
    dro        the worst case over a KL ball (engine), with duality/ underneath
"""
from .backtest import backtest, forecast_scores, performance, walk_forward_forecasts, wealth
from .data import STOCKS, download_prices, load_prices, returns_pct
from .forecast import HistoricalMean, SklearnForecaster
from .portfolio import Strategy

__all__ = ["backtest", "forecast_scores", "performance", "walk_forward_forecasts", "wealth",
           "STOCKS", "download_prices", "load_prices", "returns_pct",
           "HistoricalMean", "SklearnForecaster", "Strategy"]
