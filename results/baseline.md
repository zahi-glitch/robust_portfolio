# Baseline results

Bundled prices, 10 US stocks; out of sample from 1992-12-28 to 2018-04-11. Monthly rebalance, long-only, 10 bps costs.

## 1. Forecast quality (next month's average daily return of each stock)

| forecaster | R² vs zero | rank IC | hit rate |
|---|---|---|---|
| historical mean | -0.014 | 0.058 | 56.6% |
| ridge | 0.003 | 0.027 | 58.9% |

## 2. Portfolios

| strategy | annual return | volatility | Sharpe | max drawdown | turnover |
|---|---|---|---|---|---|
| Equal weight | 17.9% | 22.5% | 0.84 | -57.8% | 0.00 |
| Minimum variance | 14.7% | 17.0% | 0.89 | -38.6% | 0.08 |
| Markowitz + historical mean | 19.4% | 21.7% | 0.92 | -45.4% | 0.29 |
| Robust mean-variance + historical mean | 16.5% | 17.6% | 0.96 | -38.3% | 0.14 |
| Markowitz + ridge | 14.0% | 20.6% | 0.74 | -46.3% | 0.57 |
| Robust mean-variance + ridge | 14.8% | 17.2% | 0.89 | -39.0% | 0.18 |
