"""Smoke tests for FTLE and schedules.

Run with:
    pytest tests/

These tests are intentionally cheap; they verify correctness on small
parameter settings rather than reproducing the paper's results.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.systems import get_system
from src.ftle import ftle_largest, ftle_running_integral
from src.schedules import (
    saturating_schedule,
    largest_global_W,
    shrunken_schedule,
    coverage,
    coverage_ratio,
)


def test_l63_jacobian_consistency():
    """Analytical Jacobian matches finite-difference."""
    sys = get_system("l63")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(3)
    J_analytic = sys.jac(0.0, x)
    eps = 1e-6
    J_fd = np.zeros_like(J_analytic)
    f0 = sys.rhs(0.0, x)
    for j in range(3):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (sys.rhs(0.0, xp) - f0) / eps
    assert np.allclose(J_analytic, J_fd, atol=1e-4)


def test_rossler_jacobian_consistency():
    sys = get_system("rossler")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(3) * 0.5
    J_analytic = sys.jac(0.0, x)
    eps = 1e-6
    J_fd = np.zeros_like(J_analytic)
    f0 = sys.rhs(0.0, x)
    for j in range(3):
        xp = x.copy()
        xp[j] += eps
        J_fd[:, j] = (sys.rhs(0.0, xp) - f0) / eps
    assert np.allclose(J_analytic, J_fd, atol=1e-4)


def test_l63_ftle_recovers_lambda1():
    """FTLE estimator on L63 should recover lambda_1 ~ 0.866 within 30%
    on a small sample. Looser tolerance because n is small."""
    sys = get_system("l63")
    rng = np.random.default_rng(0)
    x0s = sys.sample_attractor_ic(20, rng)
    estimates = []
    for i, x0 in enumerate(x0s):
        sub = np.random.default_rng(i + 100)
        res = ftle_largest(sys, x0, tau=10.0, rng=sub)
        estimates.append(res.lambda_ftle)
    mean = np.mean(estimates)
    # Should be within 30% of 0.866.
    assert 0.5 < mean < 1.2, f"FTLE estimate mean {mean} far from 0.866"


def test_running_integral_matches_largest():
    """The terminal value of the running integral equals the FTLE * tau."""
    sys = get_system("l63")
    rng = np.random.default_rng(0)
    x0 = sys.sample_attractor_ic(1, rng)[0]

    tau = 5.0
    rng_t = np.random.default_rng(99)
    res_largest = ftle_largest(sys, x0, tau=tau, rng=rng_t,
                               qr_interval_steps=1)
    rng_r = np.random.default_rng(99)
    t_grid, Lambda = ftle_running_integral(sys, x0, tau=tau, rng=rng_r)

    # Terminal Lambda should match log_stretch from the largest estimator
    # (modulo small differences due to QR-interval bookkeeping).
    assert abs(Lambda[-1] - res_largest.log_stretch) < 0.1 * abs(res_largest.log_stretch + 1)


def test_saturating_basic():
    """Saturating rule is W_i = beta / mu_i."""
    mu = np.array([1.0, 2.0, 4.0])
    W = saturating_schedule(mu, beta=2.0)
    assert np.allclose(W, [2.0, 1.0, 0.5])


def test_largest_global_W():
    """W_g_max = min_i W_i^star."""
    mu = np.array([1.0, 2.0, 4.0])
    Wg = largest_global_W(mu, beta=2.0)
    assert np.isclose(Wg, 0.5)


def test_shrunken_uses_smaller_budget():
    mu = np.array([1.0, 2.0, 4.0])
    W_star = saturating_schedule(mu, beta=2.0)
    W_tilde = shrunken_schedule(mu, beta=2.0, epsilon=0.5)
    # All W_tilde should be smaller than W_star
    assert np.all(W_tilde < W_star)
    # Specifically, W_tilde = 1.5 / mu
    assert np.allclose(W_tilde, 1.5 / mu)


def test_coverage_ratio_homogeneous_is_zero():
    """When all mu are equal, the relative coverage gain is exactly zero."""
    mu = np.full(10, 2.0)
    res = coverage_ratio(mu, beta=4.0)
    assert abs(res["ratio_minus_one"]) < 1e-10


def test_coverage_ratio_heterogeneous_positive():
    mu = np.array([1.0, 2.0, 4.0, 8.0])
    res = coverage_ratio(mu, beta=4.0)
    # W_star = [4, 2, 1, 0.5], W_g_max = 0.5, mean = 1.875
    # ratio_minus_one = (1.875 - 0.5) / 0.5 = 2.75
    assert np.isclose(res["ratio_minus_one"], 2.75)


def test_proposition4_feasibility_two_sided():
    """Two-sided uniform error bound: |Lambda_hat - Lambda| <= eps everywhere
    under the constant-mu approximation translates to bounded relative error
    in mu. The budget-shrunken schedule must remain feasible.
    
    Setup: choose mu_true uniformly bounded below. Inject relative noise so
    that the resulting Lambda(W) = mu * W has uniform error eps for any W
    in the test range. Verify Proposition 4(i): tilde W is feasible.
    """
    rng = np.random.default_rng(0)
    # Bound mu_true below so the resulting Lambda has bounded relative error.
    L_lower = 0.5
    mu_true = rng.uniform(L_lower, 2.0, size=200)

    beta = 1.0
    # Choose a uniform Lambda-error bound eps and back out mu noise.
    eps = 0.05
    # Worst-case W in the schedule is W_max = beta / L_lower = 2.0
    W_max = beta / L_lower
    # |Lambda(W) - Lambda_hat(W)| = |mu - mu_hat| * W <= eps for W <= W_max
    # so we need |mu - mu_hat| <= eps / W_max.
    sigma_mu = eps / W_max
    mu_noisy = mu_true + rng.uniform(-sigma_mu, sigma_mu, size=mu_true.shape)
    mu_noisy = np.maximum(mu_noisy, 1e-6)

    W_tilde = shrunken_schedule(mu_noisy, beta=beta, epsilon=eps)
    Lambda_at_tilde = mu_true * W_tilde
    # Feasibility: Lambda_i(tilde W_i) <= beta for all i
    assert np.all(Lambda_at_tilde <= beta + 1e-9), (
        f"Max Lambda at tildeW = {Lambda_at_tilde.max()}, beta = {beta}"
    )


def test_svd_ftle_is_largest_direction():
    """The defined FTLE (sigma_max) is at least the directional proxy for any
    single tangent, and on L63 at tau = T_lambda/2 it is well above it on average."""
    sys = get_system("l63")
    rng = np.random.default_rng(0)
    x0s = sys.sample_attractor_ic(20, rng)
    tau = 0.5 / sys.reference_lambda1
    svd = np.array([ftle_largest(sys, x, tau, method="svd").lambda_ftle for x in x0s])
    single = np.array([ftle_largest(sys, x, tau, method="single", rng=np.random.default_rng(i)).lambda_ftle
                       for i, x in enumerate(x0s)])
    assert np.all(svd >= single - 1e-9)
    assert svd.mean() > single.mean()
