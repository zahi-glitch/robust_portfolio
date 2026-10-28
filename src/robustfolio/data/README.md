# Data

`stock_prices.csv`: daily adjusted closing prices of 20 US stocks, 1989-12-29 to 2018-04-11.
Taken unchanged from the test resources of [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
(`tests/resources/stock_prices.csv`), MIT licence, Copyright (c) 2018 Robert Andrew Martin (full text in `LICENSE-PyPortfolioOpt`).

Caveat: the tickers are well-known companies chosen today, so any backtest on them has
survivorship bias. The backtests compare strategies on the same universe, which is fair
between strategies but flatters absolute returns.
