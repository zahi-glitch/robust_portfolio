"""The Parisi side: inf over (q, m) of the Parisi function.

Paper notation (KSS23, eq. (34), with the -rho convention):

    P(q, m) = Phi(q) - q Phi'(q) + (1/m) (log Z^{q,m} + rho) - rho,
    Z^{q,m} = E_mu exp( m [Phi'(q) g + f] ),          g = phi^2 in the paper,

for q in the range of g ([0, 1] in the paper), m in (0, 1].
Then  sup_{H <= rho} Omega = inf P.

DRO / convex coordinates: lambda = 1/m is the multiplier of the KL budget and
p = Phi'(q) the multiplier of the linearisation x = E_nu[g]. In (p, lambda)

    P~(p, lambda) = Phi*(p) + lambda log E_mu exp((p g + f)/lambda)
                    + (lambda - 1) rho,      Phi*(p) = sup_x {Phi(x) - p x},

is jointly convex. The solver uses the two KKT conditions:

    stationarity in q:   q = E_{nu^{q,m}}[g]              (unique root)
    slackness in m:      m = 1  or  H(nu^{q,m} | mu) = rho.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .gibbs import Quadrature, Tilt
from .model import Model

LOG2 = float(np.log(2.0))


def parisi_function(quad: Quadrature, q: float, m: float, rho: float = LOG2) -> float:
    """P(q, m) in the paper's coordinates."""
    if not 0.0 < m <= 1.0:
        raise ValueError("m must lie in (0, 1]")
    model = quad.model
    log_Z = quad.gibbs(q, m).log_Z
    return model.Phi(q) - q * model.dPhi(q) + (log_Z + rho) / m - rho


def parisi_function_convex(quad: Quadrature, p: float, lam: float, rho: float = LOG2) -> float:
    """P~(p, lambda) in the jointly convex coordinates p = Phi'(q), lambda = 1/m.

    Phi*(p) is computed as Phi(q) - q p at the q in the range of g with
    Phi'(q) = p, found by bisection (Phi' is strictly decreasing).
    """
    if lam < 1.0:
        raise ValueError("lambda must be >= 1")
    q = _inverse_dPhi(quad, p)
    phi_star = quad.model.Phi(q) - q * p
    log_Z = quad.tilt(p / lam, 1.0 / lam).log_Z
    return phi_star + lam * log_Z + (lam - 1.0) * rho


def _inverse_dPhi(quad: Quadrature, p: float) -> float:
    model, a, b = quad.model, quad.q_lo, quad.q_hi
    lo, hi = model.dPhi(b), model.dPhi(a)  # Phi' decreasing
    if not lo - 1e-12 <= p <= hi + 1e-12:
        raise ValueError(f"p={p} outside Phi'([{a}, {b}]) = [{lo}, {hi}]")
    return brentq(lambda q: model.dPhi(q) - p, a, b, xtol=1e-15)


def fixed_point_q(quad: Quadrature, m: float) -> float:
    """Unique q in the range of g with q = E_{nu^{q,m}}[g].

    q - E_{nu^{q,m}}[g] is strictly increasing in q when Phi'' < 0, so the
    root is bracketed by the range of g and unique.
    """
    a, b = quad.q_lo, quad.q_hi
    gap = lambda q: q - quad.gibbs(q, m).E_g
    if gap(a) >= 0.0:
        return a
    if gap(b) <= 0.0:
        return b
    tol = 1e-14 * max(1.0, abs(a), abs(b))
    return brentq(gap, a, b, xtol=tol, rtol=1e-14)


@dataclass(frozen=True)
class DualSolution:
    value: float        # inf P = sup_{H <= rho} Omega
    q: float            # optimal overlap q* = E_{nu*}[g]
    m: float            # optimal Parisi parameter m*
    lam: float          # KL multiplier lambda* = 1/m*
    p: float            # linearisation multiplier p* = Phi'(q*)
    gibbs: Tilt         # the optimal measure nu^{q*, m*}
    phase: str          # "high" (budget slack, m* = 1) or "low" (binding)


def solve_dual(model: Model, rho: float = LOG2, n_points: int = 2801,
               m_min: float = 1e-6) -> DualSolution:
    """Minimise the Parisi function via its KKT conditions.

    Along the path m -> q*(m), dP/dm = (H(nu) - rho)/m^2 and H decreases as
    m decreases, so either H <= rho at m = 1 (high temperature) or there is
    a unique m* < 1 with H(nu^{q*(m*), m*}) = rho (low temperature).
    """
    quad = Quadrature(model, n_points)

    def entropy_gap(m: float) -> float:
        q = fixed_point_q(quad, m)
        return quad.gibbs(q, m).H - rho

    if entropy_gap(1.0) <= 0.0:
        m_star, phase = 1.0, "high"
    else:
        if entropy_gap(m_min) > 0.0:
            raise RuntimeError("entropy constraint cannot be met; increase rho")
        m_star = brentq(entropy_gap, m_min, 1.0, xtol=1e-14, rtol=1e-13)
        phase = "low"

    q_star = fixed_point_q(quad, m_star)
    nu = quad.gibbs(q_star, m_star)
    value = parisi_function(quad, q_star, m_star, rho)
    return DualSolution(value, q_star, m_star, 1.0 / m_star, model.dPhi(q_star), nu, phase)
