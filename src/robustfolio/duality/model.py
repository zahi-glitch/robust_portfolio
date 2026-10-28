"""Model specification.

A model is a reference measure mu and three functions:

    g   : support -> R   enters the non-linearity (g = phi^2 in the paper)
    f   : support -> R   linear part
    Phi : R -> R         concave non-linearity, Phi'' < 0 on the range of g

The objective over probability measures nu << mu is

    Omega(nu) = Phi( E_nu[g] ) + E_nu[f] - H(nu | mu),

maximised under the KL budget H(nu | mu) <= rho (rho = log 2 in the REM).

The reference measure is either the standard Gaussian N(0, 1) (the spin-glass
setting of the papers) or an empirical measure on data points (the DRO
setting: mu = (1/n) sum_i delta_{x_i}).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Model:
    g: Callable[[Array], Array]
    f: Callable[[Array], Array]
    Phi: Callable[[float], float]
    dPhi: Callable[[float], float]
    name: str = "custom"
    params: dict = field(default_factory=dict)
    support: Optional[Array] = None   # data points; None means mu = N(0, 1)
    q_range: Optional[tuple] = None   # range of E_nu[g]; default: range of g on the support


def sk_model(beta: float, h: float = 0.0) -> Model:
    """REM approximation of the SK TAP free energy (KSS23, Section 3.1), mu = N(0,1).

    Phi(x) = beta^2/4 (1 - x^2)
    g(x)   = phi(x)^2,  phi(x) = tanh(h + beta x)
    f(x)   = -1/2 log(1 - tanh^2(h + beta x)) - beta/2 x tanh(h + beta x)
           = log cosh(h + beta x) - beta/2 x tanh(h + beta x)

    f is bounded below (by -|h|/2) and grows like beta/2 |x|; under the
    Gaussian mu it has exponential moments of all orders.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")

    def g(x):
        return np.tanh(h + beta * x) ** 2

    def f(x):
        y = h + beta * x
        log_cosh = np.logaddexp(y, -y) - np.log(2.0)  # stable for large |y|
        return log_cosh - 0.5 * beta * x * np.tanh(y)

    def Phi(q):
        return 0.25 * beta**2 * (1.0 - q * q)

    def dPhi(q):
        return -0.5 * beta**2 * q

    return Model(g=g, f=f, Phi=Phi, dPhi=dPhi, name="sk",
                 params={"beta": beta, "h": h}, q_range=(0.0, 1.0))


def mean_variance_model(losses, gamma: float, tau: float) -> Model:
    """Worst-case mean-variance risk over a KL ball around the empirical measure.

    For losses l_1..l_n with empirical measure mu, the adversary solves

        sup { E_nu[l] + gamma Var_nu(l) - tau KL(nu | mu) :  KL(nu | mu) <= rho }.

    Since Var_nu(l) = E_nu[l^2] - (E_nu l)^2, dividing by tau gives the form
    Phi(E_nu g) + E_nu f - H(nu | mu) with

        g(l) = l,   f(l) = (l + gamma l^2) / tau,   Phi(x) = -gamma x^2 / tau.

    So the value returned by the solvers is (robust objective) / tau. tau > 0
    weights a KL penalty; as tau -> 0 the problem becomes pure KL-constrained DRO.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    if losses.size < 2:
        raise ValueError("need at least two data points")
    if gamma <= 0 or tau <= 0:
        raise ValueError("gamma and tau must be positive (Phi must be strictly concave)")

    return Model(
        g=lambda l: l,
        f=lambda l: (l + gamma * l * l) / tau,
        Phi=lambda q: -gamma * q * q / tau,
        dPhi=lambda q: -2.0 * gamma * q / tau,
        name="mean_variance",
        params={"gamma": gamma, "tau": tau},
        support=losses,
    )
