"""Kuramoto-Sivashinsky equation, periodic BC, ETDRK4 in Fourier space.

  u_t + u u_x + u_{xx} + u_{xxxx} = 0,    x in [0, L), periodic.

This module provides:
  - rhs/jac: real-space right-hand side and Jacobian (kept for diagnostics)
  - _step: ETDRK4 state integrator (overrides base RK4)
  - tangent_step: JOINT ETDRK4 step for the state + tangent system. This
    overrides the base-class RK4 joint step, which is unconditionally
    unstable for KS because the linear operator (k^2 - k^4) has eigenvalues
    of order -k_max^4 (~7300 at N=64, L=22), far outside RK4's stability
    region at any reasonable dt.

The tangent equation in Fourier space is:
    d v_hat / dt = (k^2 - k^4) v_hat  -  i k * F{ u(x) v(x) }   (dealiased)
which shares the same linear operator as the state. We integrate both with
ETDRK4 using cached coefficients.
"""
from __future__ import annotations
import numpy as np

from .base import DynamicalSystem, SystemParams


class KuramotoSivashinsky(DynamicalSystem):
    """Kuramoto-Sivashinsky in 1D periodic domain."""

    name = "ks"

    def __init__(self, L: float = 22.0, N: int = 64):
        self.L = L
        self.N = N
        self.dim = N

        # Wavenumbers and operators
        self.k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
        self.L_op = self.k ** 2 - self.k ** 4  # Linear part (in Fourier)
        # Dealiasing: 2/3 rule
        self.dealias = (np.abs(self.k) < (2.0 / 3.0) * np.max(np.abs(self.k))).astype(float)

        # Cache for ETDRK4 coefficients, keyed by dt.
        self._etdrk4_cache: dict[float, tuple] = {}

    # ----------------------------------------------------------------------
    # Real-space rhs and Jacobian (for diagnostics; not used in ETDRK4 stepping)
    # ----------------------------------------------------------------------
    def rhs(self, t, x):
        """Real-space RHS: u_t = -u u_x - u_{xx} - u_{xxxx}."""
        u_hat = np.fft.fft(x)
        u_x_hat = 1j * self.k * u_hat
        u_x = np.real(np.fft.ifft(u_x_hat))
        nonlin = -x * u_x
        lin_hat = self.L_op * u_hat
        lin = np.real(np.fft.ifft(lin_hat))
        return nonlin + lin

    def jac(self, t, x):
        """Real-space Jacobian. Used for diagnostics only; the production
        FTLE estimator does NOT call this (it uses the spectral tangent
        nonlinearity directly inside `tangent_step`).
        """
        N = self.N
        F = np.fft.fft(np.eye(N), axis=0)
        Finv = np.fft.ifft(np.eye(N), axis=0)
        D = np.real(Finv @ np.diag(1j * self.k) @ F)
        L_real = np.real(Finv @ np.diag(self.L_op) @ F)
        u_x = D @ x
        return -np.diag(u_x) - np.diag(x) @ D + L_real

    # ----------------------------------------------------------------------
    @property
    def default_params(self) -> SystemParams:
        return SystemParams(dt=0.05, burn_in=400.0, integrator="etdrk4",
                            name=f"{self.name}_l{int(self.L)}")

    @property
    def reference_lambda1(self) -> float:
        return 0.057

    def _generic_ic(self, rng):
        x = np.linspace(0, self.L, self.N, endpoint=False)
        return np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))

    # ----------------------------------------------------------------------
    # ETDRK4 coefficients, cached per dt
    # ----------------------------------------------------------------------
    def _etdrk4_coeffs(self, dt: float):
        """Compute (E, E2, Q, f1, f2, f3) for ETDRK4 at step dt; cached."""
        # Key the cache by a rounded dt to avoid float-equality issues.
        key = round(float(dt), 12)
        if key in self._etdrk4_cache:
            return self._etdrk4_cache[key]

        L_op = self.L_op
        E = np.exp(dt * L_op)
        E2 = np.exp(dt * L_op / 2)
        # Contour-integral coefficients (Kassam-Trefethen 2005)
        M = 16
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * L_op[:, None] + r[None, :]
        Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

        coeffs = (E, E2, Q, f1, f2, f3)
        self._etdrk4_cache[key] = coeffs
        return coeffs

    # ----------------------------------------------------------------------
    # Spectral nonlinearities (state and tangent)
    # ----------------------------------------------------------------------
    def _N_state(self, uhat: np.ndarray) -> np.ndarray:
        """Nonlinearity of the STATE equation in Fourier: -u u_x.

        Computed as -1/2 * i*k * F{u^2}, with dealiasing.
        """
        u = np.real(np.fft.ifft(uhat))
        return -0.5j * self.k * np.fft.fft(u * u) * self.dealias

    def _N_tangent(self, uhat: np.ndarray, vhat: np.ndarray) -> np.ndarray:
        """Linearization of -u u_x in direction v: -(u v)_x.

        In Fourier: -i*k * F{u(x) v(x)}, with dealiasing. `vhat` may be a
        matrix of tangent vectors in its columns (shape (N, m)).
        """
        u = np.real(np.fft.ifft(uhat))
        if vhat.ndim == 2:
            v = np.real(np.fft.ifft(vhat, axis=0))
            return (-1j * self.k * self.dealias)[:, None] * np.fft.fft(u[:, None] * v, axis=0)
        v = np.real(np.fft.ifft(vhat))
        return -1j * self.k * np.fft.fft(u * v) * self.dealias

    def tangent_step_matrix(self, x, M, dt, t=0.0):
        """Joint ETDRK4 step of the state and a matrix of tangent vectors."""
        E, E2, Q, f1, f2, f3 = self._etdrk4_coeffs(dt)
        uhat = np.fft.fft(x)
        Mhat = np.fft.fft(M, axis=0)
        c = lambda a: a[:, None]                       # broadcast spectral coeffs over columns
        Nu0 = self._N_state(uhat); Nv0 = self._N_tangent(uhat, Mhat)
        au = E2 * uhat + Q * Nu0; av = c(E2) * Mhat + c(Q) * Nv0
        Nu1 = self._N_state(au); Nv1 = self._N_tangent(au, av)
        bu = E2 * uhat + Q * Nu1; bv = c(E2) * Mhat + c(Q) * Nv1
        Nu2 = self._N_state(bu); Nv2 = self._N_tangent(bu, bv)
        cu = E2 * au + Q * (2 * Nu2 - Nu0); cv = c(E2) * av + c(Q) * (2 * Nv2 - Nv0)
        Nu3 = self._N_state(cu); Nv3 = self._N_tangent(cu, cv)
        uhat_new = E * uhat + Nu0 * f1 + 2 * (Nu1 + Nu2) * f2 + Nu3 * f3
        Mhat_new = c(E) * Mhat + Nv0 * c(f1) + 2 * (Nv1 + Nv2) * c(f2) + Nv3 * c(f3)
        return np.real(np.fft.ifft(uhat_new)), np.real(np.fft.ifft(Mhat_new, axis=0))

    # ----------------------------------------------------------------------
    # ETDRK4 single state step
    # ----------------------------------------------------------------------
    def _etdrk4_step(self, x, dt):
        """One ETDRK4 step on the state alone; takes & returns real-space x."""
        E, E2, Q, f1, f2, f3 = self._etdrk4_coeffs(dt)
        v = np.fft.fft(x)
        Nv = self._N_state(v)
        a = E2 * v + Q * Nv
        Na = self._N_state(a)
        b = E2 * v + Q * Na
        Nb = self._N_state(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = self._N_state(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        return np.real(np.fft.ifft(v))

    def _step(self, t, x, dt):
        # Override base RK4 with ETDRK4 for stability/speed.
        return self._etdrk4_step(x, dt)

    # ----------------------------------------------------------------------
    # JOINT ETDRK4 step (state + tangent).
    # ----------------------------------------------------------------------
    def tangent_step(self, x, v, dt, t=0.0):
        """JOINT ETDRK4 step on state + tangent.

        The state x and tangent v share the same linear operator (k^2 - k^4),
        so ETDRK4 coefficients are reused. The nonlinearities differ:
            state:    N_u(u_hat) = -1/2 i*k F{u^2}                  (dealiased)
            tangent:  N_v(u_hat, v_hat) = -i*k F{u(x) v(x)}         (dealiased)

        At each ETDRK4 stage we evaluate BOTH nonlinearities at the current
        intermediate (u_hat, v_hat), advancing them together so the tangent
        sees the correct state at each stage.

        Returns (x_new, v_new) in real space.
        """
        E, E2, Q, f1, f2, f3 = self._etdrk4_coeffs(dt)
        uhat = np.fft.fft(x)
        vhat = np.fft.fft(v)

        # Stage 0
        Nu0 = self._N_state(uhat)
        Nv0 = self._N_tangent(uhat, vhat)

        # Stage 1 (predictor at half-step)
        au = E2 * uhat + Q * Nu0
        av = E2 * vhat + Q * Nv0
        Nu1 = self._N_state(au)
        Nv1 = self._N_tangent(au, av)

        # Stage 2 (corrector at half-step)
        bu = E2 * uhat + Q * Nu1
        bv = E2 * vhat + Q * Nv1
        Nu2 = self._N_state(bu)
        Nv2 = self._N_tangent(bu, bv)

        # Stage 3 (predictor at full step)
        cu = E2 * au + Q * (2 * Nu2 - Nu0)
        cv = E2 * av + Q * (2 * Nv2 - Nv0)
        Nu3 = self._N_state(cu)
        Nv3 = self._N_tangent(cu, cv)

        # Combine
        uhat_new = E * uhat + Nu0 * f1 + 2 * (Nu1 + Nu2) * f2 + Nu3 * f3
        vhat_new = E * vhat + Nv0 * f1 + 2 * (Nv1 + Nv2) * f2 + Nv3 * f3

        return np.real(np.fft.ifft(uhat_new)), np.real(np.fft.ifft(vhat_new))
