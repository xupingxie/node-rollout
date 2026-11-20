"""Rössler system."""
from __future__ import annotations
import numpy as np

from .base import DynamicalSystem, SystemParams


class Rossler(DynamicalSystem):
    """3D chaotic attractor with longer Lyapunov time than L63.

    dx/dt = -y - z
    dy/dt =  x + a*y
    dz/dt =  b + z*(x - c)
    """

    name = "rossler"
    dim = 3

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        self.a = a
        self.b = b
        self.c = c

    def rhs(self, t, x):
        a, b, c = self.a, self.b, self.c
        return np.array(
            [
                -x[1] - x[2],
                x[0] + a * x[1],
                b + x[2] * (x[0] - c),
            ]
        )

    def jac(self, t, x):
        a, _, c = self.a, self.b, self.c
        return np.array(
            [
                [0.0, -1.0, -1.0],
                [1.0, a, 0.0],
                [x[2], 0.0, x[0] - c],
            ]
        )

    @property
    def default_params(self) -> SystemParams:
        # Slower dynamics than L63; small dt still adequate.
        return SystemParams(dt=0.02, burn_in=200.0, integrator="rk4", name=self.name)

    @property
    def reference_lambda1(self) -> float:
        return 0.066  # matches the v5 paper's table

    def _generic_ic(self, rng):
        return np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
