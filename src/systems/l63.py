"""Lorenz-63 system."""
from __future__ import annotations
import numpy as np

from .base import DynamicalSystem, SystemParams


class Lorenz63(DynamicalSystem):
    """Classic 3D chaotic attractor.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x*y - beta*z
    """

    name = "l63"
    dim = 3

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def rhs(self, t, x):
        s, r, b = self.sigma, self.rho, self.beta
        return np.array(
            [
                s * (x[1] - x[0]),
                x[0] * (r - x[2]) - x[1],
                x[0] * x[1] - b * x[2],
            ]
        )

    def jac(self, t, x):
        s, r, b = self.sigma, self.rho, self.beta
        return np.array(
            [
                [-s, s, 0.0],
                [r - x[2], -1.0, -x[0]],
                [x[1], x[0], -b],
            ]
        )

    @property
    def default_params(self) -> SystemParams:
        return SystemParams(dt=0.01, burn_in=20.0, integrator="rk4", name=self.name)

    @property
    def reference_lambda1(self) -> float:
        return 0.866  # numerically estimated; matches the v5 paper's table

    def _generic_ic(self, rng):
        # A point near (but not on) the attractor; long burn-in handles the rest.
        return np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
