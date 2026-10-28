"""Portfolio rules: from a window of past returns (and optionally a forecast) to weights.

R is a (T x d) array of daily returns in %, x the weights: long-only and fully invested (x >= 0, sum x = 1).
The daily loss of portfolio x is l_t = -R_t . x.

Strategies
----------
equal             x = 1/d
minvar            min_x Var(l)                                          ignores expected returns
markowitz         min_x E[l] + gamma Var(l)                             trusts expected returns fully
robust_mean       min_x sup { E_nu[l] : KL(nu | mu) <= rho }            entropic value-at-risk (EVaR)
robust_mv         min_x sup { E_nu[l] + gamma Var_nu(l) - tau KL(nu | mu) : KL(nu | mu) <= rho }

mu is the empirical distribution of the window (weight 1/T per day). An adversary may reweight the days
within a relative-entropy (KL) budget rho: the larger rho, the less the past (or the forecast) is trusted.
The inner sup of robust_mv is evaluated through its Parisi-type dual (Kistler-Schmidt-Sebastiani 2023).

Where a forecast enters
-----------------------
A forecast `mu_hat` (expected daily return of each asset, in %) replaces the window's average return:
the window is shifted so that its mean equals mu_hat while its covariance and tails stay the same
(`recentre`). Markowitz then trusts mu_hat fully; the robust rules distrust it by an amount set by rho.
Without a forecast, mu_hat is the historical mean of the window.

Gradients in x come from Danskin's theorem: with nu* the worst-case reweighting,
    d/dx E_nu*[l] = -E_nu*[R],      d/dx Var_nu*(l) = 2 Cov_nu*(R) x.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .dro import worst_case_mean, worst_case_mean_variance


def recentre(R: np.ndarray, mu_hat) -> np.ndarray:
    """Shift each column of R so that its mean is mu_hat (covariance unchanged)."""
    if mu_hat is None:
        return R
    return R - R.mean(0) + np.asarray(mu_hat, dtype=float)


# --------------------------------------------------------------------------- objectives (value, gradient)
def markowitz_objective(R, x, gamma):
    mu, S = R.mean(0), np.cov(R, rowvar=False, bias=True)
    return -mu @ x + gamma * x @ S @ x, -mu + 2 * gamma * S @ x


def robust_mean_objective(R, x, rho):
    r = worst_case_mean(-R @ x, rho)
    return r.value, -r.weights @ R


def robust_mean_variance_objective(R, x, gamma, rho, tau):
    r = worst_case_mean_variance(-R @ x, gamma, tau, rho, check_primal=False)
    w = r.weights
    m = w @ R
    C = (R - m).T @ (w[:, None] * (R - m))
    return r.objective, -m + 2 * gamma * C @ x


def _simplex_minimise(fun_grad, d, x0=None):
    x0 = np.full(d, 1.0 / d) if x0 is None else x0
    res = minimize(fun_grad, x0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * d,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(d)}],
                   options={"ftol": 1e-10, "maxiter": 200})
    x = np.clip(res.x, 0.0, None)
    return x / x.sum()


# --------------------------------------------------------------------------- strategies
@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str                     # "equal", "minvar", "markowitz", "robust_mean", "robust_mv"
    gamma: float = 0.05           # risk aversion (returns in %)
    rho: float = 0.05             # KL budget: how much the data / forecast is distrusted
    tau: float = 0.1              # KL penalty inside robust_mv

    @property
    def uses_forecast(self) -> bool:
        return self.kind in ("markowitz", "robust_mean", "robust_mv")

    def weights(self, R, mu_hat=None, x0=None) -> np.ndarray:
        """Weights from the window R (T x d) and an optional forecast mu_hat (d,)."""
        d = R.shape[1]
        if self.kind == "equal":
            return np.full(d, 1.0 / d)
        if self.kind == "minvar":
            S = np.cov(R, rowvar=False, bias=True)
            return _simplex_minimise(lambda x: (x @ S @ x, 2 * S @ x), d, x0)
        Rc = recentre(R, mu_hat)
        if self.kind == "markowitz":
            return _simplex_minimise(lambda x: markowitz_objective(Rc, x, self.gamma), d, x0)
        if self.kind == "robust_mean":
            return _simplex_minimise(lambda x: robust_mean_objective(Rc, x, self.rho), d, x0)
        if self.kind == "robust_mv":
            return _simplex_minimise(
                lambda x: robust_mean_variance_objective(Rc, x, self.gamma, self.rho, self.tau), d, x0)
        raise ValueError(f"unknown strategy kind {self.kind!r}")
