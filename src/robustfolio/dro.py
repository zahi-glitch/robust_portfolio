"""KL-robust worst cases of a loss distribution (the KL-DRO engine).

Given losses l_1..l_n (for instance l = -return of a portfolio, in %), the
adversary reweights the data within a KL ball around the empirical measure:

    J = sup_nu { E_nu[l] + gamma Var_nu(l) - tau KL(nu | mu_n) :  KL(nu | mu_n) <= rho }.

This is the problem of Kistler-Schmidt-Sebastiani (2023) with a concave non-linearity (the -(E l)^2 in the
variance), so the same Parisi-type dual applies: two scalar parameters
(q, m), with q the worst-case mean loss and m the KL multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .duality.dual import solve_dual
from .duality.gibbs import Quadrature
from .duality.model import mean_variance_model
from .duality.primal import solve_primal


@dataclass(frozen=True)
class WorstCaseMean:
    value: float            # sup { E_nu[l] : KL(nu | mu_n) <= rho }
    nominal: float          # E_mu[l]
    lam: float              # temperature of the tilt (KL multiplier); 0 when frozen
    weights: np.ndarray     # nu* on the data points
    kl: float               # KL(nu* | mu_n)
    frozen: bool            # True when nu* sits on the worst data points only


def _tilt(losses, t):
    a = t * (losses - losses.max())
    w = np.exp(a); w /= w.sum()
    return w


def _kl_uniform(w):
    n = w.size
    nz = w > 0
    return float(np.sum(w[nz] * np.log(w[nz] * n)))


def worst_case_mean(losses, rho: float) -> WorstCaseMean:
    """Linear KL-DRO: the worst expected loss over the KL ball of radius rho around the data.

        sup { E_nu[l] : KL(nu | mu_n) <= rho } = inf_{lam > 0} { lam log E_mu exp(l / lam) + lam rho },

    and the maximiser is the Gibbs tilt nu*(i) ∝ exp(l_i / lam*), lam* fixed by KL(nu* | mu_n) = rho.
    The largest reachable KL is log(n / k), k = number of data points with the maximal loss: beyond it
    nu* is uniform on those points (the 'frozen' case).
    """
    losses = np.asarray(losses, dtype=float).ravel()
    n = losses.size
    if rho < 0:
        raise ValueError("rho must be non-negative")
    top = np.isclose(losses, losses.max())
    if rho <= 0:
        w = np.full(n, 1.0 / n)
        return WorstCaseMean(float(losses.mean()), float(losses.mean()), float("inf"), w, 0.0, False)
    if rho >= np.log(n / top.sum()) - 1e-12:
        w = top / top.sum()
        return WorstCaseMean(float(losses.max()), float(losses.mean()), 0.0, w, _kl_uniform(w), True)
    hi = 1.0
    while _kl_uniform(_tilt(losses, hi)) < rho:
        hi *= 2.0
    t = brentq(lambda t: _kl_uniform(_tilt(losses, t)) - rho, 0.0, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    w = _tilt(losses, t)
    return WorstCaseMean(float(w @ losses), float(losses.mean()), 1.0 / t, w, _kl_uniform(w), False)


def worst_case_mean_dual(losses, lam: float, rho: float) -> float:
    """The dual function lam log E_mu exp(l / lam) + lam rho (its inf over lam is the worst case)."""
    losses = np.asarray(losses, dtype=float).ravel()
    c = losses.max()
    return float(c + lam * np.log(np.mean(np.exp((losses - c) / lam))) + lam * rho)


@dataclass(frozen=True)
class RobustMeanVariance:
    objective: float        # J = sup { risk(nu) - tau KL } over the KL ball
    worst_risk: float       # E_nu*[l] + gamma Var_nu*(l)
    nominal_risk: float     # same under the empirical measure
    worst_mean: float       # q* = E_nu*[l]
    kl: float               # KL(nu* | mu_n)
    m: float                # Parisi parameter m*
    kl_multiplier: float    # total multiplier of KL: tau / m*
    phase: str              # "slack" or "binding"
    weights: np.ndarray     # nu* as probabilities on the data points
    primal_dual_gap: float


def worst_case_mean_variance(losses, gamma: float, tau: float, rho: float,
                             check_primal: bool = True) -> RobustMeanVariance:
    losses = np.asarray(losses, dtype=float).ravel()
    model = mean_variance_model(losses, gamma, tau)
    quad = Quadrature(model)
    d = solve_dual(model, rho)
    weights = quad.weights(d.m * model.dPhi(d.q), d.m)
    mean = float(weights @ losses)
    worst_risk = mean + gamma * float(weights @ (losses - mean) ** 2)
    nominal_risk = losses.mean() + gamma * losses.var()
    gap = abs(solve_primal(model, rho).value - d.value) * tau if check_primal else float("nan")
    return RobustMeanVariance(
        objective=tau * d.value, worst_risk=worst_risk, nominal_risk=float(nominal_risk),
        worst_mean=d.q, kl=d.gibbs.H, m=d.m, kl_multiplier=tau / d.m,
        phase="slack" if d.phase == "high" else "binding", weights=weights,
        primal_dual_gap=gap,
    )
