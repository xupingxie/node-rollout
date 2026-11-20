"""Generate fig_longtime_psd.png: per-coordinate power spectral density
of L63 long-time rollouts vs ground truth.

Three panels (x, y, z), one per coordinate. The PSD of the reference
trajectory is plotted as a gray line; each method as a colored line.
Methods that collapsed to fixed-point attractors (no oscillatory
content) appear as flat curves at modest amplitudes.

EXPECTED INPUT
--------------
Same JSON schema as plot_longtime_marginals.py:
each input has `arr` of shape (K, T, 3), `dt` (time step), `config`,
and `class`.

USAGE
-----
    python scripts/figures/plot_longtime_psd.py \
        --reference data/l63_reference.json \
        --inputs logs/mp_node.json logs/adaptive.json ... \
        --out plots/fig_longtime_psd.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch


CLASS_COLORS = {
    "reference": "#333333",
    "single_shoot": "#888888",
    "mp_node": "#2ca02c",
    "global_lyapunov": "#1f77b4",
    "adaptive": "#d62728",
}


def infer_class(name: str) -> str:
    c = name.lower()
    if "single" in c or "shoot" in c:
        return "single_shoot"
    if "mp_node" in c or "mpnode" in c:
        return "mp_node"
    if "lyapunov" in c and "global" in c:
        return "global_lyapunov"
    if "ground" in c or "true" in c or "reference" in c:
        return "reference"
    return "adaptive"


def load(path: Path):
    with open(path) as f:
        rec = json.load(f)
    arr = np.asarray(rec["rollouts"])  # (K, T, 3)
    rec["arr"] = arr
    rec["dt"] = float(rec.get("dt", 0.01))
    rec.setdefault("class", infer_class(rec.get("config", "")))
    return rec


def compute_psd(arr: np.ndarray, dt: float, axis_idx: int):
    """Welch PSD averaged over rollouts (K), per coordinate."""
    fs = 1.0 / dt
    freqs = None
    psds = []
    for k in range(arr.shape[0]):
        x = arr[k, :, axis_idx]
        f, p = welch(x, fs=fs, nperseg=min(2048, len(x)))
        freqs = f
        psds.append(p)
    return freqs, np.mean(np.stack(psds, axis=0), axis=0)


def plot(reference, curves, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    axis_names = ["x", "y", "z"]

    for j, name in enumerate(axis_names):
        ax = axes[j]
        f_ref, p_ref = compute_psd(reference["arr"], reference["dt"], j)
        ax.loglog(f_ref, p_ref, color="#333333", lw=1.5,
                  label="True L63" if j == 0 else None)

        for c in curves:
            f, p = compute_psd(c["arr"], c["dt"], j)
            color = CLASS_COLORS.get(c["class"], "black")
            label = c["config"] if j == 0 else None
            ax.loglog(f, p, color=color, lw=1.2, alpha=0.85, label=label)

        ax.set_xlabel("frequency (1/time)", fontsize=10)
        if j == 0:
            ax.set_ylabel("PSD", fontsize=11)
        ax.set_title(name, fontsize=11)
        ax.grid(True, which="both", alpha=0.2)

    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.suptitle("L63 power spectra from long-time rollouts", fontsize=11)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_longtime_psd] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--inputs", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ref = load(args.reference)
    curves = [load(p) for p in args.inputs]
    plot(ref, curves, args.out)


if __name__ == "__main__":
    main()
