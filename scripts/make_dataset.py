"""Generate the training dataset (.npz) for one system.

    python scripts/make_dataset.py --config configs/l63.yaml
    python scripts/make_dataset.py --config configs/ks_l22.yaml --set data.T_train=100

Writes datasets/<system>.npz. Run once per system; every training run
loads the same file so all schedules see identical data and teacher FTLE.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train.config import RunConfig
from src.train.data import make_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action="append", default=[], help="overrides, e.g. data.T_train=100")
    ap.add_argument("--out_dir", default="datasets")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. tau2 -> datasets/l63_tau2.npz "
                                            "(use with --set data.tau_factor=2.0 for a longer-horizon profile)")
    args = ap.parse_args()
    args.set = [x for grp in args.set for x in grp]   # several --set groups accumulate
    cfg = RunConfig.load(args.config, args.set)
    d = cfg.data
    out = Path(args.out_dir) / f"{d.system}{('_' + args.tag) if args.tag else ''}.npz"
    make_dataset(d.system, d.T_train, d.T_val, d.n_test, d.T_test, d.T_long,
                 tau_factor=d.tau_factor, ftle_stride=d.ftle_stride,
                 lyap_T=d.lyap_T, seed=d.seed, gap_T=d.gap_T, ftle_method=d.ftle_method,
                 out_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
