"""Prices and returns.

Two sources:
  load_prices      the bundled file: daily prices of US stocks, 1990-2018 (works offline)
  download_prices  Yahoo Finance via yfinance (needs internet and `pip install -e ".[live]"`)

Both return a pandas DataFrame: one row per trading day, one column per ticker.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICES = Path(__file__).with_name("data") / "stock_prices.csv"
# The 10 stocks of the bundled file with a full history from 1990 to 2018.
STOCKS = ["AAPL", "GE", "AMD", "WMT", "BAC", "T", "XOM", "BBY", "PFE", "JPM"]


def load_prices(tickers=STOCKS, start=None, end=None, path=None) -> pd.DataFrame:
    """Daily adjusted closing prices from the bundled CSV (or `path`)."""
    df = pd.read_csv(path or PRICES, parse_dates=[0], index_col=0).sort_index()
    df.index.name = "date"
    if tickers is not None:
        df = df[list(tickers)]
    return df.loc[start:end].dropna()


def download_prices(tickers, start="2005-01-01", end=None) -> pd.DataFrame:
    """Daily closing prices adjusted for dividends and splits, from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError('yfinance is not installed: pip install -e ".[live]"') from e
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError("no prices downloaded: check the internet connection")
    df = data["Close"][list(tickers)].dropna(how="all").ffill(limit=2).dropna()
    df.index.name = "date"
    return df


def returns_pct(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns in %: row t is the return from day t-1 to day t."""
    return 100.0 * prices.pct_change().iloc[1:]
