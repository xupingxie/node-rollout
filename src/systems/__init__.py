"""Dynamical systems registry.

Usage:
    from src.systems import get_system
    sys = get_system("l63")          # Lorenz-63
    sys = get_system("l96_n40")      # Lorenz-96, N=40
    sys = get_system("rossler")      # Rössler
    sys = get_system("ks_l22")       # Kuramoto-Sivashinsky, L=22, N=64
"""
from __future__ import annotations
from .base import DynamicalSystem, SystemParams
from .l63 import Lorenz63
from .rossler import Rossler
from .l96 import Lorenz96
from .ks import KuramotoSivashinsky


def get_system(name: str) -> DynamicalSystem:
    """Factory for the four benchmark systems."""
    name = name.lower().strip()
    if name in ("l63", "lorenz63", "lorenz_63"):
        return Lorenz63()
    if name in ("rossler", "roessler"):
        return Rossler()
    if name in ("l96", "l96_n40", "lorenz96"):
        return Lorenz96(N=40, F=8.0)
    if name in ("l96_n8",):
        return Lorenz96(N=8, F=8.0)
    if name in ("ks", "ks_l22"):
        return KuramotoSivashinsky(L=22.0, N=64)
    raise ValueError(f"Unknown system: {name!r}. "
                     f"Try one of: l63, rossler, l96_n40, ks_l22.")


__all__ = [
    "DynamicalSystem",
    "SystemParams",
    "Lorenz63",
    "Rossler",
    "Lorenz96",
    "KuramotoSivashinsky",
    "get_system",
]
