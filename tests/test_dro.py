import numpy as np
import pytest
from scipy.optimize import brentq, minimize

from robustfolio.data import load_prices, returns_pct
from robustfolio.dro import worst_case_mean, worst_case_mean_dual, worst_case_mean_variance
from robustfolio.duality.dual import fixed_point_q, solve_dual
from robustfolio.duality.gibbs import Quadrature
from robustfolio.duality.model import Model

LOG2 = np.log(2.0)
LOSSES = -returns_pct(load_prices()).mean(axis=1).to_numpy()[-756:]   # equal-weight portfolio, 2015-2018


def test_sample_data():
    assert LOSSES.size == 756 and abs(LOSSES.mean()) < 0.2


@pytest.mark.parametrize("gamma,tau,rho", [(1.0, 0.5, 0.1), (1.0, 0.05, 0.05), (0.5, 10.0, 0.1), (2.0, 1.0, 0.5)])
def test_mean_variance_primal_equals_dual(gamma, tau, rho):
    res = worst_case_mean_variance(LOSSES, gamma, tau, rho)
    assert res.primal_dual_gap < 1e-9
    assert res.kl <= rho + 1e-10
    assert abs(res.weights.sum() - 1) < 1e-12
    assert res.objective >= res.nominal_risk - 1e-12      # nu = mu is feasible
    assert abs(res.objective - (res.worst_risk - tau * res.kl)) < 1e-8


@pytest.mark.parametrize("gamma,tau,rho", [(1.0, 0.3, 0.08), (0.5, 5.0, 0.2)])
def test_mean_variance_against_direct_optimisation(gamma, tau, rho):
    """Independent check: optimise directly over the n weights (SLSQP)."""
    rng = np.random.default_rng(1)
    l = rng.choice(LOSSES, 40, replace=False)
    n = l.size

    def kl(w):
        w = np.clip(w, 1e-300, None)
        return float(np.sum(w * np.log(w * n)))

    def neg_obj(w):
        m = w @ l
        return -(m + gamma * (w @ (l - m) ** 2) - tau * kl(w))

    best = np.inf
    for w0 in (np.full(n, 1 / n), rng.dirichlet(np.ones(n) * 5)):
        r = minimize(neg_obj, w0, method="SLSQP", bounds=[(0, 1)] * n,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                  {"type": "ineq", "fun": lambda w: rho - kl(w)}],
                     options=dict(ftol=1e-14, maxiter=2000))
        best = min(best, r.fun)
    res = worst_case_mean_variance(l, gamma, tau, rho, check_primal=False)
    assert abs(-best - res.objective) < 1e-6


def test_slack_and_binding_phases():
    assert worst_case_mean_variance(LOSSES, 0.5, 30.0, 0.1, check_primal=False).phase == "slack"
    assert worst_case_mean_variance(LOSSES, 1.0, 0.1, 0.1, check_primal=False).phase == "binding"


def test_linear_rem_threshold():
    """Classical REM: Phi = 0, f = beta x, mu = N(0,1). Budget binds at beta = sqrt(2 log 2)."""
    def rem(beta):
        return Model(g=np.tanh, f=lambda x: beta * x, Phi=lambda q: -1e-9 * q * q,
                     dPhi=lambda q: -2e-9 * q)

    def H_at_m1(beta):
        quad = Quadrature(rem(beta))
        return quad.gibbs(fixed_point_q(quad, 1.0), 1.0).H

    beta_c = brentq(lambda b: H_at_m1(b) - LOG2, 0.1, 5.0, xtol=1e-12)
    assert abs(beta_c - np.sqrt(2 * LOG2)) < 1e-8
    d = solve_dual(rem(2.0))                                  # low temperature
    assert abs(d.value - (2.0 * np.sqrt(2 * LOG2) - LOG2)) < 1e-8
    assert abs(d.m - np.sqrt(2 * LOG2) / 2.0) < 1e-8


def test_linear_kl_dro_primal_equals_dual():
    """Worst-case mean over a KL ball: Gibbs tilt (sup) = inf over the temperature lambda."""
    from scipy.optimize import minimize_scalar
    losses = LOSSES
    for rho in (0.01, 0.1, 0.5, 2.0):
        r = worst_case_mean(losses, rho)
        assert abs(r.kl - rho) < 1e-9 and not r.frozen
        dual = minimize_scalar(lambda u: worst_case_mean_dual(losses, np.exp(u), rho),
                               bounds=(-8, 6), method="bounded", options={"xatol": 1e-12})
        assert abs(dual.fun - r.value) < 1e-7
        assert abs(worst_case_mean_dual(losses, r.lam, rho) - r.value) < 1e-9
    frozen = worst_case_mean(losses, 10.0)
    assert frozen.frozen and frozen.value == losses.max()
