"""Abstract base class for chaotic dynamical systems.

Each system provides:
  - rhs(t, x): the right-hand side of dx/dt = f(x)
  - jac(t, x): the analytical Jacobian d f / d x  (for tangent-linear FTLE)
  - dim: state dimension
  - default_params: integration parameters tuned for the system
  - sample_attractor_ic(n, rng): n initial conditions on the attractor
  - reference_lambda1: estimated largest Lyapunov exponent (numerical)

Conventions:
  - State is a 1D numpy array of shape (dim,)
  - Jacobian returns an array of shape (dim, dim)
  - All time is in the system's natural units; T_lambda = 1 / lambda_1
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SystemParams:
    """Default integration / sampling parameters for a system."""
    dt: float                    # solver step size
    burn_in: float               # burn-in time before sampling attractor states
    integrator: str = "rk4"      # one of {"rk4", "rk45"}
    name: str = ""

    def n_burnin_steps(self) -> int:
        return int(np.ceil(self.burn_in / self.dt))


class DynamicalSystem(ABC):
    """Abstract chaotic dynamical system."""

    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """dx/dt = f(t, x). Returns array of shape (dim,)."""

    @abstractmethod
    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """Jacobian df/dx evaluated at (t, x). Returns shape (dim, dim)."""

    @property
    @abstractmethod
    def default_params(self) -> SystemParams: ...

    @property
    @abstractmethod
    def reference_lambda1(self) -> float:
        """Numerically-estimated largest Lyapunov exponent.

        These values are taken from running our own FTLE estimator on a long
        post-transient trajectory; see scripts/01_compute_ftle_tables.py.
        """

    def sample_attractor_ic(
        self,
        n: int,
        rng: np.random.Generator,
        spacing: float | None = None,
    ) -> np.ndarray:
        """Sample n initial conditions from the attractor.

        Default implementation: integrate from a generic starting point past
        the burn-in time, then sample states along a long trajectory at
        intervals of `spacing` time units. Subclasses can override to provide
        system-specific initial-condition sampling.
        """
        params = self.default_params
        x0 = self._generic_ic(rng)
        n_burn = params.n_burnin_steps()

        # burn in
        x = x0.copy()
        for _ in range(n_burn):
            x = self._step(0.0, x, params.dt)

        # sample at intervals
        if spacing is None:
            spacing = max(10.0 * params.dt, 1.0 / max(self.reference_lambda1, 1e-3))
        n_steps_between = max(1, int(spacing / params.dt))

        out = np.empty((n, self.dim))
        for i in range(n):
            for _ in range(n_steps_between):
                x = self._step(0.0, x, params.dt)
            out[i] = x
        return out

    def _generic_ic(self, rng: np.random.Generator) -> np.ndarray:
        """A generic non-zero initial condition. Override per system if needed."""
        return rng.standard_normal(self.dim) * 0.1

    def _step(self, t: float, x: np.ndarray, dt: float) -> np.ndarray:
        """Single integrator step (RK4 by default)."""
        k1 = self.rhs(t, x)
        k2 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = self.rhs(t + dt, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def tangent_step(
        self,
        x: np.ndarray,
        v: np.ndarray,
        dt: float,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single integrator step of the joint state + tangent system.

        Default implementation: RK4 of
            dx/dt = f(x)
            dv/dt = J(x) v
        This is adequate for non-stiff systems (L63, Rossler, L96). Stiff
        systems with large negative eigenvalues in the linear part (KS) must
        override this with an exponential / IMEX integrator; otherwise
        explicit RK4 on the tangent blows up for any dt above the stability
        threshold.
        """
        def joint_rhs(_t, _x, _v):
            return self.rhs(_t, _x), self.jac(_t, _x) @ _v

        k1x, k1v = joint_rhs(t, x, v)
        k2x, k2v = joint_rhs(t + 0.5 * dt, x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = joint_rhs(t + 0.5 * dt, x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = joint_rhs(t + dt, x + dt * k3x, v + dt * k3v)
        x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return x_new, v_new

    def tangent_step_matrix(
        self,
        x: np.ndarray,
        M: np.ndarray,
        dt: float,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Joint step of the state and a matrix of tangent vectors (columns),
        dM/dt = J(x) M. Default: RK4, matching `tangent_step`. Used to
        propagate the full tangent matrix for the largest singular value.
        """
        def joint_rhs(_t, _x, _M):
            return self.rhs(_t, _x), self.jac(_t, _x) @ _M

        k1x, k1M = joint_rhs(t, x, M)
        k2x, k2M = joint_rhs(t + 0.5 * dt, x + 0.5 * dt * k1x, M + 0.5 * dt * k1M)
        k3x, k3M = joint_rhs(t + 0.5 * dt, x + 0.5 * dt * k2x, M + 0.5 * dt * k2M)
        k4x, k4M = joint_rhs(t + dt, x + dt * k3x, M + dt * k3M)
        return (x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
                M + (dt / 6.0) * (k1M + 2.0 * k2M + 2.0 * k3M + k4M))

    def integrate(
        self,
        x0: np.ndarray,
        T: float,
        dt: float | None = None,
        return_traj: bool = False,
    ) -> np.ndarray:
        """Integrate from x0 for total time T.

        Returns final state if return_traj=False, else full trajectory of
        shape (n_steps + 1, dim).
        """
        dt = dt if dt is not None else self.default_params.dt
        n = int(np.round(T / dt))

        if return_traj:
            traj = np.empty((n + 1, self.dim))
            traj[0] = x0
            x = x0.copy()
            for k in range(n):
                x = self._step(k * dt, x, dt)
                traj[k + 1] = x
            return traj
        else:
            x = x0.copy()
            for k in range(n):
                x = self._step(k * dt, x, dt)
            return x
