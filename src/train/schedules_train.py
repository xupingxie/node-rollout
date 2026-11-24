"""Training-time rollout schedules.

Every policy maps (step, teacher rates mu for the batch) -> per-segment
horizons W_i in time units, plus an info dict for logging. The trainer
converts W_i to solver steps. Five families:

  ss       constant window W                        (class baseline, budget sweep)
  const_g  W_g^max(beta_k) = beta_k / max_i mu_i    (the matched-budget baseline
                                                     of Theorem 1, batch-wise)
  sat      W_i = beta_k / mu_i, bucketed            (the theorem's schedule)
  ramp     W ramps ramp_W0 -> W over training       (global-Lyapunov curriculum)
  mpnode   constant nominal W split into K chunks   (multi-window class)

`sat` and `const_g` share the same warm-up beta_k so that pairing them at
equal `beta` is a matched-budget comparison by construction:
    beta_k = beta_min + (beta - beta_min) (1 - exp(-k / k0)).
"""
from __future__ import annotations
import numpy as np


class Policy:
    K = 1
    name = "base"

    def __init__(self, cfg, T_lambda, dt, n_steps):
        self.cfg = cfg
        self.T = T_lambda
        self.dt = dt
        self.n_steps = n_steps
        self.W_min = cfg.W_min_steps * dt
        self.W_max = cfg.W_max * T_lambda

    def beta_k(self, step):
        b = self.cfg.beta
        b0 = self.cfg.beta_min_frac * b
        k0 = max(1.0, self.cfg.warmup_frac * self.n_steps)
        return b0 + (b - b0) * (1.0 - np.exp(-step / k0))

    def __call__(self, step, mu):
        raise NotImplementedError

    def clip(self, W):
        return np.clip(W, self.W_min, self.W_max)


class SingleShoot(Policy):
    name = "ss"

    def __call__(self, step, mu):
        W = np.full(len(mu), self.cfg.W * self.T)
        return self.clip(W), {"beta_k": float("nan")}


class Ramp(Policy):
    name = "ramp"

    def __call__(self, step, mu):
        frac = min(1.0, step / max(1, self.n_steps))
        Wt = (self.cfg.ramp_W0 + (self.cfg.W - self.cfg.ramp_W0) * frac) * self.T
        return self.clip(np.full(len(mu), Wt)), {"beta_k": float("nan")}


class ConstG(Policy):
    name = "const_g"

    def __call__(self, step, mu):
        b = self.beta_k(step)
        Wg = b / np.max(mu)
        return self.clip(np.full(len(mu), Wg)), {"beta_k": b}


class Saturating(Policy):
    name = "sat"

    def __call__(self, step, mu):
        b = self.beta_k(step)
        W = self.clip(b / mu)
        bk = self.cfg.bucketing
        if bk != "none":
            q = int(bk[1:])
            # quantile buckets of the batch; each bucket takes its mean W so
            # that the batch splits into at most q rollout groups.
            edges = np.quantile(W, np.linspace(0, 1, q + 1))
            which = np.clip(np.searchsorted(edges, W, side="right") - 1, 0, q - 1)
            Wb = W.copy()
            agg = np.min if self.cfg.bucket_agg == "min" else np.mean
            for j in range(q):
                m = which == j
                if m.any():
                    # `min` keeps every member of the bucket inside its own
                    # budget (Lambda_i = mu_i W_bucket <= mu_i W_i <= beta);
                    # `mean` gives the high-mu
                    # members more than beta and breaks budget matching.
                    Wb[m] = agg(W[m])
            W = Wb
        return W, {"beta_k": b}


class MPNode(Policy):
    name = "mpnode"

    def __init__(self, cfg, T_lambda, dt, n_steps):
        super().__init__(cfg, T_lambda, dt, n_steps)
        self.K = int(cfg.K)

    def __call__(self, step, mu):
        W = np.full(len(mu), self.cfg.W * self.T)
        return self.clip(W), {"beta_k": float("nan")}


def build_policy(cfg, T_lambda, dt, n_steps) -> Policy:
    table = {"ss": SingleShoot, "ramp": Ramp, "const_g": ConstG,
             "sat": Saturating, "mpnode": MPNode}
    if cfg.name not in table:
        raise ValueError(f"unknown schedule {cfg.name!r}; choose from {list(table)}")
    return table[cfg.name](cfg, T_lambda, dt, n_steps)


def run_name(cfg) -> str:
    """Canonical run directory name from a schedule config."""
    n = cfg.name
    if n == "ss":
        return f"ss_W{cfg.W:g}"
    if n == "ramp":
        return f"ramp_W{cfg.ramp_W0:g}to{cfg.W:g}"
    if n == "const_g":
        wmax = "" if cfg.W_max == 2.0 else f"_Wmax{cfg.W_max:g}"
        return f"const_g_b{cfg.beta:g}{wmax}"
    if n == "sat":
        agg = "" if cfg.bucket_agg == "min" or cfg.bucketing == "none" else f"_{cfg.bucket_agg}"
        wmax = "" if cfg.W_max == 2.0 else f"_Wmax{cfg.W_max:g}"
        return f"sat_b{cfg.beta:g}_{cfg.bucketing}{agg}_{cfg.loss_norm}{wmax}"
    if n == "mpnode":
        return f"mpnode_K{cfg.K}_W{cfg.W:g}"
    raise ValueError(n)
