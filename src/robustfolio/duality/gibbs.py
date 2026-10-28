"""Expectations under mu and under exponentially tilted measures.

Two kinds of reference measure:

* mu = N(0, 1): the trapezoid rule on a uniform grid of [-L, L] with the
  Gaussian density folded into the weights. For integrands analytic in a
  strip, as here (tanh(h + beta x) has its nearest poles at distance
  pi/(2 beta) from the real axis), the trapezoid rule converges geometrically,
  and it does not lose accuracy when tanh(beta x) is steep (large beta), as
  Gauss-Hermite quadrature does.
* mu = empirical measure on data points: weights 1/n, exact.

All exponentials are evaluated with a log-sum-exp shift, so no overflow
occurs for large tilts. The dashboard (docs/solver.js) uses the same rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .model import Model

DEFAULT_HALF_WIDTH = 14.0
DEFAULT_POINTS = 2801


@lru_cache(maxsize=8)
def gaussian_grid(half_width: float = DEFAULT_HALF_WIDTH,
                  n_points: int = DEFAULT_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Points x_i and weights w_i with sum_i w_i g(x_i) ~ E_mu[g], mu = N(0,1)."""
    x = np.linspace(-half_width, half_width, n_points)
    dx = x[1] - x[0]
    w = np.full(n_points, dx)
    w[[0, -1]] *= 0.5
    w *= np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return x, w


@dataclass(frozen=True)
class Tilt:
    """Summary of nu ~ exp(a g + b f) mu."""

    log_Z: float     # log E_mu exp(a g + b f)
    E_g: float       # E_nu[g]
    E_f: float       # E_nu[f]
    H: float         # H(nu | mu)


class Quadrature:
    """Caches g and f at the integration points of one model."""

    def __init__(self, model: Model, n_points: int = DEFAULT_POINTS,
                 half_width: float = DEFAULT_HALF_WIDTH):
        self.model = model
        if model.support is None:
            self.x, self.w = gaussian_grid(half_width, n_points)
        else:
            self.x = np.asarray(model.support, dtype=float)
            self.w = np.full(self.x.size, 1.0 / self.x.size)
        self.g = np.asarray(model.g(self.x), dtype=float)
        self.f = np.asarray(model.f(self.x), dtype=float)
        lo, hi = model.q_range if model.q_range else (self.g.min(), self.g.max())
        self.q_lo, self.q_hi = float(lo), float(hi)

    def weights(self, a: float, b: float) -> np.ndarray:
        """Probabilities of nu ~ exp(a g + b f) mu at the integration points."""
        expo = a * self.g + b * self.f
        p = self.w * np.exp(expo - expo.max())
        return p / p.sum()

    def tilt(self, a: float, b: float) -> Tilt:
        """Statistics of the tilted measure nu ~ exp(a g + b f) mu."""
        expo = a * self.g + b * self.f
        shift = expo.max()
        weights = self.w * np.exp(expo - shift)
        Z = weights.sum()
        weights /= Z
        log_Z = shift + np.log(Z)
        E_g = float(weights @ self.g)
        E_f = float(weights @ self.f)
        H = a * E_g + b * E_f - log_Z  # Gibbs identity
        return Tilt(float(log_Z), E_g, E_f, float(max(H, 0.0)))

    def gibbs(self, q: float, m: float) -> Tilt:
        """Generalised Gibbs measure nu^{q,m} ~ exp(m [Phi'(q) g + f]) mu."""
        return self.tilt(m * self.model.dPhi(q), m)
