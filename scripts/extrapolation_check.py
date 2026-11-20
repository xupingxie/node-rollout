"""Profile-extrapolation diagnostic for the Lorenz-63 training study (Section 5.5).

Uses the saved Corollary-3 FTLE tables, whose 4000 sample states are identical
at tau = T/2, T and 2T (checked below). For the nominal saturating schedule
W_i = min(beta / mu_i(tau=T/2), W_max) computed on these states WITHOUT the
training-time bucketing, it reports:
  * the fraction of states whose window exceeds tau and the fraction at W_max;
  * the "nominal" budget mu_i(T/2) * W_i charged by the profile;
  * a horizon-consistent estimate of the stretch over the assigned window,
    mu_i(tau*) * W_i with tau* the saved horizon nearest to W_i from above
    (T/2 for W <= T/2, T for T/2 < W <= T, 2T for W > T), i.e. the FTLE of the
    SAME state measured over a horizon that covers the window.
These are teacher-flow diagnostics on an unbucketed nominal schedule, not a
trace of the bucketed windows actually assigned during training.

    python extrapolation_check.py results_v12r1/corollary3
"""
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v12r1/corollary3")
T = 1.0 / 0.866                                                      # Lorenz-63 reference T_lambda
tab = {}
for tag, name in [("0.5", "tau0.5"), ("1.0", "tau1"), ("2.0", "tau2")]:
    z = np.load(root / f"part_l63_tau{tag}" / f"ftle_l63_{name}.npz")
    tab[tag] = (z["x0"], z["lambda_ftle"])
assert np.array_equal(tab["0.5"][0], tab["1.0"][0]) and np.array_equal(tab["0.5"][0], tab["2.0"][0]), "states differ"
a, c, b = tab["0.5"][1], tab["1.0"][1], tab["2.0"][1]
print(f"n={len(a)}  mean rate: tau=T/2 {a.mean():.3f}, T {c.mean():.3f}, 2T {b.mean():.3f}")
print(f"corr(rate at T/2, rate at 2T) = {np.corrcoef(a, b)[0,1]:.4f};  corr(T/2, T) = {np.corrcoef(a, c)[0,1]:.4f}")
low = a <= np.quantile(a, 0.10)
print(f"bottom decile of the T/2 rate: mean {a[low].mean():.4f}; same states at 2T: {b[low].mean():.4f}; "
      f"stretch over 2T: {(b[low]*2*T).mean():.2f} nats vs nominal {(a[low]*2*T).mean():.2f}")
for beta in (0.5, 1.0, 2.0):
    for Wmax in (2 * T, 0.5 * T):
        W = np.minimum(beta / np.maximum(a, 1e-3), Wmax)
        mu_star = np.where(W > T, b, np.where(W > 0.5 * T, c, a))
        nominal, consistent = a * W, mu_star * W
        cap = W >= Wmax - 1e-12
        print(f"beta {beta:3.1f} Wmax {Wmax/T:.1f}T: W>tau {100*np.mean(W > 0.5*T):5.1f}%  at cap {100*cap.mean():5.1f}%  "
              f"mean W {W.mean()/T:.2f}T | nominal mean {nominal.mean():.2f} | horizon-consistent mean {consistent.mean():.2f} "
              f"p95 {np.quantile(consistent, 0.95):.2f} max {consistent.max():.2f}"
              + (f" | at cap: nominal {nominal[cap].mean():.2f} consistent {consistent[cap].mean():.2f}" if cap.any() else ""))
