"""The DRO side: sup of Omega over measures with a KL budget.

    sup { Phi(E_nu[g]) + E_nu[f] - H(nu | mu) :  H(nu | mu) <= rho }.

The optimiser is a generalised Gibbs measure (KSS23, Prop. 4.5), so it lies
in the two-parameter exponential family

    nu_t ~ exp(t1 g + t2 f) mu.

We maximise over t = (t1, t2) independently of the dual, with a quadratic
penalty on the budget (continuation in the penalty weight) and a final
projection onto H = rho along the ray s -> s t if the result is infeasible,
so the reported value is always attained by a feasible measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

from .gibbs import Quadrature, Tilt
from .model import Model

LOG2 = float(np.log(2.0))


def omega(model: Model, nu: Tilt) -> float:
    """Omega(nu) = Phi(E_nu g) + E_nu f - H(nu | mu)."""
    return model.Phi(nu.E_g) + nu.E_f - nu.H


@dataclass(frozen=True)
class PrimalSolution:
    value: float     # Omega(nu*) at a feasible nu*
    t: tuple         # tilt parameters (t1, t2)
    nu: Tilt


def solve_primal(model: Model, rho: float = LOG2, n_points: int = 2801,
                 starts=None) -> PrimalSolution:
    """Maximise Omega over nu_t ~ exp(t1 g + t2 f) mu with H(nu_t | mu) <= rho.

    The search runs in the coordinates (v, u) with t2 = exp(u), t1 = v t2,
    which keeps the problem well scaled when the optimal t2 is very small
    (a strongly binding budget).
    """
    quad = Quadrature(model, n_points)

    def to_t(z):
        s = np.exp(z[1])
        return np.array([z[0] * s, s])

    def evaluate(t):
        nu = quad.tilt(t[0], t[1])
        return omega(model, nu), nu

    scale = 1.0 + abs(evaluate((0.0, 0.0))[0])

    def penalised(z, weight):
        val, nu = evaluate(to_t(z))
        return -val / scale + weight * max(0.0, nu.H - rho) ** 2

    if starts is None:
        v0 = model.dPhi(0.5 * (quad.q_lo + quad.q_hi))
        starts = [(v0, u0) for u0 in (0.0, -2.0, -4.0, -6.0)] + [(0.0, 0.0)]

    best = None
    for z0 in starts:
        z = np.asarray(z0, dtype=float)
        for weight in (1e1, 1e3, 1e5, 1e7):
            z = minimize(penalised, z, args=(weight,), method="Nelder-Mead",
                         options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000)).x
        t = to_t(z)
        val, nu = evaluate(t)
        if nu.H > rho + 1e-12:  # project onto the budget along the ray
            s = brentq(lambda s: evaluate(s * t)[1].H - rho, 0.0, 1.0, xtol=1e-15)
            t = s * t
            val, nu = evaluate(t)
        if best is None or val > best.value:
            best = PrimalSolution(float(val), (float(t[0]), float(t[1])), nu)
    return best
