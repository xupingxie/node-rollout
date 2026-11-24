"""Write the job list (one `train.py` argument string per line) for a system.

The default grid is:
  ss      W in {0.1, 0.25, 0.5, 1.0, 2.0}
  const_g beta in {0.5, 1.0, 2.0}
  sat     beta in {0.5, 1.0, 2.0}          (q4 / inv_W)
  ramp    0.1 -> 1.0
  mpnode  K in {2, 3, 5} at W = 1.0
times `--seeds` seeds.

    python scripts/slurm/make_job_list.py --system l63 --seeds 5 > scripts/slurm/jobs_l63.txt
    sbatch --array=1-$(wc -l < scripts/slurm/jobs_l63.txt) scripts/slurm/run_train.sbatch scripts/slurm/jobs_l63.txt
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--extra", default="", help="appended to every line, e.g. '--set train.n_steps=30000'")
grid = ap.add_mutually_exclusive_group()
grid.add_argument("--readiness", action="store_true", help="three L63 pilot comparisons per seed")
grid.add_argument("--minimal", action="store_true",
                help="reduced grid: ss {0.25,0.5,1.0}, const_g/sat {0.5,1.0}, ramp, mpnode K=3 (9 schedules)")
grid.add_argument("--decisive", action="store_true",
                help="the full comparison grid: ss {0.1,0.25,0.5,1,2}, const_g {0.5,1,2}, "
                     "sat capped at tau {0.5,1,2}, sat uncapped (W_max=2 T_lambda) {0.5,1,2}, "
                     "sat capped UNWEIGHTED (per_segment loss) {0.5,1,2}, const_g with the same 0.5 cap {0.5,1,2}, "
                     "ramp, mpnode K {2,3,5} (24 schedules)")
grid.add_argument("--lean", action="store_true",
                help="no window longer than 1 T_lambda: ss {0.1,0.25,0.5,1.0}, const_g {0.5,1,2}, "
                     "const_g and sat {0.5,1,2} with W_max = 0.5 T_lambda (= the FTLE horizon), ramp, mpnode K {3,5}. "
                     "For KS, where 2 T_lambda unrolls are too slow and OOM at 3 procs/GPU.")
a = ap.parse_args()

cfg = f"configs/{a.system}.yaml"
jobs = []
Ws = (0.25, 0.5, 1.0) if a.minimal else (0.1, 0.25, 0.5, 1.0) if a.lean else (0.1, 0.25, 0.5, 1.0, 2.0)
betas = (0.5, 1.0) if a.minimal else (0.5, 1.0, 2.0)
Ks = (3,) if a.minimal else (3, 5) if a.lean else (2, 3, 5)
sat_extra = " --set schedule.W_max=0.5" if a.lean else ""
for W in Ws:
    jobs.append(f"--schedule ss --W {W}")
for b in betas:
    jobs.append(f"--schedule const_g --beta {b}{sat_extra}")
    if a.decisive:
        jobs.append(f"--schedule const_g --beta {b} --set schedule.W_max=0.5")   # same cap as the capped rule
        jobs.append(f"--schedule sat --beta {b} --set schedule.W_max=0.5")
        jobs.append(f"--schedule sat --beta {b}")
        jobs.append(f"--schedule sat --beta {b} --set schedule.W_max=0.5 schedule.loss_norm=per_segment")
    else:
        jobs.append(f"--schedule sat --beta {b}{sat_extra}")
jobs.append("--schedule ramp --W 1.0")
for K in Ks:
    jobs.append(f"--schedule mpnode --W 1.0 --K {K}")
if a.readiness:
    jobs = ["--schedule ss --W 0.5",
            "--schedule const_g --beta 1.0 --set schedule.W_max=0.5",
            "--schedule sat --beta 1.0 --set schedule.W_max=0.5"]
# Cheapest first within each seed so the matched-budget pairs finish early.

for seed in range(a.seeds):
    for j in jobs:
        print(f"--config {cfg} {j} --seed {seed} {a.extra}".strip())
