"""Job list for the W_max = tau ablation: the saturating rule with its maximum
window capped at the FTLE horizon (0.5 T_lambda), so the constant-rate profile
is never extrapolated beyond the horizon it was measured on.

    python scripts/slurm/make_ablation_wmax.py --system l63 --seeds 5 > scripts/slurm/jobs_l63_wmax.txt
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--W_max", type=float, default=0.5)
a = ap.parse_args()
for seed in range(a.seeds):
    for b in (0.5, 1.0, 2.0):
        print(f"--config configs/{a.system}.yaml --schedule sat --beta {b} --seed {seed} "
              f"--set schedule.W_max={a.W_max}")
