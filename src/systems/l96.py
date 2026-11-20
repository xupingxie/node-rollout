"""Lorenz-96 system."""
from __future__ import annotations
import numpy as np

from .base import DynamicalSystem, SystemParams


class Lorenz96(DynamicalSystem):
    """N-variable spatially-extended chaotic system.

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F     for i = 0, ..., N-1
    with periodic indexing.
    """

    name = "l96"

    def __init__(self, N: int = 40, F: float = 8.0):
        self.N = N
        self.F = F
        self.dim = N

    def rhs(self, t, x):
        # Roll-based vectorized form.
        N = self.N
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + self.F

    def jac(self, t, x):
        # df_i/dx_j has nonzero entries at j in {i-2, i-1, i, i+1}.
        N = self.N
        J = np.zeros((N, N))
        for i in range(N):
            ip1 = (i + 1) % N
            im1 = (i - 1) % N
            im2 = (i - 2) % N
            J[i, im2] = -x[im1]
            J[i, im1] = x[ip1] - x[im2]
            J[i, i] = -1.0
            J[i, ip1] = x[im1]
        return J

    @property
    def default_params(self) -> SystemParams:
        return SystemParams(dt=0.01, burn_in=50.0, integrator="rk4", name=f"{self.name}_n{self.N}")

    @property
    def reference_lambda1(self) -> float:
        # Numerically estimated for N=40, F=8.
        if self.N == 40:
            return 1.733
        elif self.N == 8:
            return 1.636
        else:
            # Caller should re-estimate before relying on this.
            return 1.7

    def _generic_ic(self, rng):
        x0 = self.F * np.ones(self.N)
        x0[0] += 0.01  # break symmetry
        return x0 + 1e-3 * rng.standard_normal(self.N)
