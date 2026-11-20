"""Floor sweep for the Corollary 3 verification, computed from the saved FTLE
tables of the v12r1 CPU stage (results_v12r1/corollary3/part_<system>_tau0.5/
ftle_<system>_tau0.5.npz). Same procedure as scripts/test_corollary3.py:
floor L at a percentile of the positive FTLE values, beta = tau * mean FTLE,
W_i* = beta / max(lambda_i, L), W_g^max = min_i W_i*, R = mean W* / W_g^max - 1,
and the bound CV_eff^2 with CV_eff the CV of the floored rate.

    python floor_sweep.py results_v12r1/corollary3
"""
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v12r1/corollary3")
Q = [0.01, 0.05, 0.10, 0.25, 0.50]
print(f"{'system':8s} {'beta':>6s} {'nonpos':>7s}  " + "  ".join(f"{int(q*100):>3d}%: R (CVeff^2)" for q in Q))
for s in ["l63", "rossler", "l96_n40", "ks_l22"]:
    z = np.load(root / f"part_{s}_tau0.5" / f"ftle_{s}_tau0.5.npz")
    lam, tau = z["lambda_ftle"], float(z["tau"])
    beta, pos = tau * lam.mean(), lam[lam > 0]
    cells = []
    for q in Q:
        L = max(float(np.quantile(pos, q)), 1e-3)
        mu = np.maximum(lam, L); W = beta / mu
        cells.append(f"{W.mean()/W.min()-1:6.2f} ({(mu.std()/mu.mean())**2:.3f})")
    print(f"{s:8s} {beta:6.3f} {100*np.mean(lam<=0):6.1f}%  " + "  ".join(cells))
