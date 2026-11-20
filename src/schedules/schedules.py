"""Rollout schedule constructors.

The schedules are characterized by per-segment rollout lengths W_i. Three
families are relevant to the paper:

  - saturating(beta, mu_hat):      W_i^star  = max W : Lambda_i(W) <= beta
                                              (under the constant-mu_hat
                                              approximation, W_i = beta / mu_hat_i)
  - global_lyapunov(beta, mu_hat): W_g^max = min_i W_i^star (constant rule
                                              with the largest window
                                              feasible globally)
  - shrunken(beta, eps, mu_hat):   tilde W_i = max W : hat Lambda_i(W) <= beta - eps
                                              (Proposition 4)

All three accept a per-segment scalar mu_hat array of shape (B,) representing
the local instability estimate (the FTLE estimate at the segment's start).
They return arrays of W_i values clipped to [W_min, W_max].
"""
from __future__ import annotations
import numpy as np


def saturating_schedule(
    mu_hat: np.ndarray,
    beta: float,
    W_min: float = 0.0,
    W_max: float = np.inf,
) -> np.ndarray:
    """W_i^star(beta) under the constant-mu approximation.

    Lambda_i(W) ~= mu_i * W, so W_i = beta / mu_i. Clipped to [W_min, W_max].
    """
    mu_hat = np.asarray(mu_hat, dtype=float)
    # Avoid division by zero on near-laminar segments.
    mu_safe = np.where(mu_hat > 1e-12, mu_hat, 1e-12)
    W = beta / mu_safe
    return np.clip(W, W_min, W_max)


def largest_global_W(
    mu_hat: np.ndarray,
    beta: float,
    W_min: float = 0.0,
    W_max: float = np.inf,
) -> float:
    """W_g^max = min_i W_i^star(beta), the largest constant window feasible
    for every segment under budget beta.
    """
    W_star = saturating_schedule(mu_hat, beta, W_min, W_max)
    return float(W_star.min())


def shrunken_schedule(
    mu_hat: np.ndarray,
    beta: float,
    epsilon: float,
    W_min: float = 0.0,
    W_max: float = np.inf,
) -> np.ndarray:
    """tilde W_i(beta; eps) from Proposition 4.

    Under constant-mu approximation: tilde W_i = (beta - eps) / mu_i.
    """
    if beta <= epsilon:
        raise ValueError(f"Budget {beta} must exceed shrinkage {epsilon}.")
    return saturating_schedule(mu_hat, beta - epsilon, W_min, W_max)


def coverage(W: np.ndarray) -> float:
    """Total coverage C(W) = sum_i W_i."""
    return float(np.sum(W))


def coverage_ratio(
    mu_hat: np.ndarray,
    beta: float,
    W_min: float = 0.0,
    W_max: float = np.inf,
) -> dict:
    """Quantities used to verify Corollary 3.

    Returns:
        {
          "W_star": (B,) saturating schedule,
          "W_g_max": scalar largest-feasible-global window,
          "W_bar_star": scalar mean of W_star,
          "C_star": coverage of W_star,
          "C_g_max": coverage of constant W_g_max schedule,
          "ratio_minus_one": (W_bar_star - W_g_max) / W_g_max,
                             == relative coverage gain (Corollary 3)
        }
    """
    W_star = saturating_schedule(mu_hat, beta, W_min, W_max)
    W_g_max = float(W_star.min())
    W_bar_star = float(W_star.mean())
    C_star = coverage(W_star)
    C_g_max = float(W_g_max * len(W_star))
    return {
        "W_star": W_star,
        "W_g_max": W_g_max,
        "W_bar_star": W_bar_star,
        "C_star": C_star,
        "C_g_max": C_g_max,
        "ratio_minus_one": (W_bar_star - W_g_max) / W_g_max if W_g_max > 0 else float("nan"),
    }
