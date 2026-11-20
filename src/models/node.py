"""Neural ODE with a fixed-step RK4 integrator.

The vector field operates on normalized coordinates internally
(x_n = (x - mean) / std) and returns dx/dt in raw coordinates, so the same
code works for L63 (state scale ~10-40), KS (scale ~1) and L96 (scale ~5)
without per-system tuning.

Two field types:
  - MLPField : dense MLP, for L63 / Rossler / L96 and as a KS fallback
  - ConvField: periodic 1D conv net, the natural choice for KS on a
               periodic grid (translation-equivariant)

NeuralODE.rollout(x0, n_steps) returns the trajectory (B, n_steps+1, dim)
including x0. All rollouts inside one call share n_steps; the trainer
groups segments by step count (bucketing) and calls rollout once per
group.
"""
from __future__ import annotations
import torch
import torch.nn as nn


_ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}


class _Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def norm(self, x):
        return (x - self.mean) / self.std

    def scale_out(self, y):
        return y * self.std


class MLPField(nn.Module):
    """f_theta(x) = std * MLP((x - mean)/std)."""

    def __init__(self, dim, width=128, depth=4, activation="silu", mean=None, std=None):
        super().__init__()
        act = _ACTS[activation]
        layers = []
        d = dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)
        self.normalizer = _Normalizer(mean if mean is not None else torch.zeros(dim),
                                      std if std is not None else torch.ones(dim))
        # Small output init: start near the identity flow, which stabilizes
        # the first rollouts.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.normalizer.scale_out(self.net(self.normalizer.norm(x)))


class ConvField(nn.Module):
    """Periodic 1D conv net for fields on a periodic grid (KS)."""

    def __init__(self, dim, channels=64, depth=5, kernel=5, activation="silu",
                 mean=None, std=None):
        super().__init__()
        act = _ACTS[activation]
        pad = kernel // 2
        layers = [nn.Conv1d(1, channels, kernel, padding=pad, padding_mode="circular"), act()]
        for _ in range(depth - 2):
            layers += [nn.Conv1d(channels, channels, kernel, padding=pad,
                                 padding_mode="circular"), act()]
        layers += [nn.Conv1d(channels, 1, kernel, padding=pad, padding_mode="circular")]
        self.net = nn.Sequential(*layers)
        self.normalizer = _Normalizer(mean if mean is not None else torch.zeros(dim),
                                      std if std is not None else torch.ones(dim))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):
        xn = self.normalizer.norm(x).unsqueeze(1)          # (B, 1, N)
        return self.normalizer.scale_out(self.net(xn).squeeze(1))


def build_field(kind, dim, mean, std, **kw):
    if kind == "mlp":
        return MLPField(dim, mean=mean, std=std, **kw)
    if kind == "conv":
        return ConvField(dim, mean=mean, std=std, **kw)
    raise ValueError(f"unknown field kind {kind!r}")


class NeuralODE(nn.Module):
    """dx/dt = f_theta(x), integrated with fixed-step RK4 at the data dt."""

    def __init__(self, field: nn.Module, dt: float):
        super().__init__()
        self.field = field
        self.dt = float(dt)

    def step(self, x, dt=None):
        dt = self.dt if dt is None else dt
        f = self.field
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout(self, x0, n_steps):
        """(B, dim) -> (B, n_steps + 1, dim), gradients flow through."""
        xs = [x0]
        x = x0
        for _ in range(int(n_steps)):
            x = self.step(x)
            xs.append(x)
        return torch.stack(xs, dim=1)

    @torch.no_grad()
    def rollout_nograd(self, x0, n_steps, chunk=2000, keep=True):
        """Long rollout without autograd, in chunks to bound memory.

        Returns (B, n_steps + 1, dim) if keep else final state (B, dim).
        Stops early and pads with NaN if the state becomes non-finite.
        """
        x = x0
        out = [x0.unsqueeze(1)] if keep else None
        done = 0
        while done < n_steps:
            m = min(chunk, n_steps - done)
            xs = []
            for _ in range(m):
                x = self.step(x)
                xs.append(x)
            block = torch.stack(xs, dim=1)
            if keep:
                out.append(block)
            done += m
            if not torch.isfinite(x).all():
                if keep:
                    pad = torch.full((x0.shape[0], n_steps - done, x0.shape[1]),
                                     float("nan"), device=x0.device, dtype=x0.dtype)
                    out.append(pad)
                break
        return torch.cat(out, dim=1) if keep else x
