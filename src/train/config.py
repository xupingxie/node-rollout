"""Run configuration: YAML per system + command-line overrides.

All horizons in the config are in units of T_lambda unless the key ends
in `_steps`. The trainer converts to solver steps with the dataset dt.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    system: str = "l63"
    T_train: float = 200.0        # in T_lambda
    T_val: float = 20.0
    n_test: int = 50              # test initial conditions
    T_test: float = 15.0          # valid-time horizon per test IC (long enough to avoid censoring)
    T_long: float = 200.0         # long-time reference trajectory
    tau_factor: float = 0.5       # FTLE horizon = tau_factor * T_lambda
    ftle_stride: int = 5          # compute FTLE every k train points
    ftle_method: str = "svd"      # svd = defined largest FTLE (full tangent matrix); single = directional proxy
    gap_T: float = 5.0            # gap (T_lambda) between the disjoint train/val/ref/test windows
    lyap_T: float = 1000.0        # Benettin run length for the true lambda_1
    seed: int = 0


@dataclass
class ModelConfig:
    kind: str = "mlp"             # mlp | conv
    width: int = 128              # mlp width or conv channels
    depth: int = 4
    kernel: int = 5               # conv only
    activation: str = "silu"
    dt_mult: int = 1              # model RK4 step = dt_mult x data dt; loss on every dt_mult-th data point


@dataclass
class ScheduleConfig:
    name: str = "ss"              # ss | const_g | sat | ramp | mpnode
    W: float = 0.5                # nominal horizon (T_lambda) for ss / mpnode
    beta: float = 1.0             # budget (nats) for sat / const_g
    beta_min_frac: float = 0.25   # warm-up starts at beta_min_frac * beta
    warmup_frac: float = 0.10     # k0 = warmup_frac * n_steps in the exp warm-up
    W_min_steps: int = 5
    W_max: float = 2.0            # T_lambda
    bucketing: str = "q4"         # q4 | q16 | none (sat only)
    bucket_agg: str = "min"       # min (budget-feasible) | mean (violates budget)
    loss_norm: str = "inv_W"      # inv_W | per_segment | global
    ramp_W0: float = 0.1          # ramp: from ramp_W0 to W over training
    K: int = 3                    # mpnode sub-windows
    mp_penalty: float = 1.0       # mpnode end-of-chunk penalty weight
    mu_floor_quantile: float = 0.0   # numerical floor on the teacher FTLE: quantile of the positive values (0 = smallest positive); W_max governs long horizons


@dataclass
class TrainConfig:
    n_steps: int = 20000
    batch_size: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-5
    # plateau: halve the lr (plateau_factor) when the validation rollout
    # loss has not improved for plateau_patience validations, floor lr_min;
    # cosine: half-cosine from lr to lr_min over n_steps (not adaptive).
    lr_schedule: str = "plateau"
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = None   # None = no clipping (pre-clip norm always logged)
    log_every: int = 10
    progress_every: int = 50      # heartbeat line (step, loss, steps/s) to stdout
    val_every: int = 100
    segment_log_every: int = 250
    segment_log_n: int = 32
    ckpt_every: int = 2000
    seed: int = 0
    device: str = "auto"
    max_nan_steps: int = 50
    # Early stopping on the validation rollout loss (schedule-independent):
    # stop when `early_stop_metric` has not improved by a relative
    # `early_stop_min_delta` for `early_stop_patience` consecutive
    # validations, but never before `min_steps`. The best checkpoint (by
    # the same metric) is what gets evaluated.
    early_stop_metric: str = "loss_val_halfT"
    early_stop_patience: int = 40
    early_stop_min_delta: float = 0.01
    min_steps: int = 10000
    resume: bool = True           # continue from checkpoint_last.pt if present
    lock_stale_s: int = 1800      # RUNNING.lock older than this is considered dead and taken over


@dataclass
class EvalConfig:
    valid_thresholds: tuple = (0.3, 0.5)
    n_long_rollouts: int = 3
    lyap_T: float = 100.0         # Benettin on the learned model, in T_lambda
    psd_nperseg: int = 1024
    mmd_n: int = 2000


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        cfg = RunConfig()
        for section in ("data", "model", "schedule", "train", "eval"):
            if section in d and d[section]:
                obj = getattr(cfg, section)
                for k, v in d[section].items():
                    if not hasattr(obj, k):
                        raise KeyError(f"unknown config key {section}.{k}")
                    setattr(obj, k, v)
        return cfg

    @staticmethod
    def load(path: str | Path, overrides: list[str] | None = None) -> "RunConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = RunConfig.from_dict(d)
        for ov in overrides or []:
            key, _, val = ov.partition("=")
            section, _, name = key.partition(".")
            obj = getattr(cfg, section)
            cur = getattr(obj, name)
            setattr(obj, name, _coerce(val, cur))
        return cfg

    def save(self, path: str | Path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _coerce(val: str, like):
    if val.lower() in ("none", "null"):
        return None
    if isinstance(like, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(val)
    if isinstance(like, float) or like is None:
        try:
            return float(val)
        except ValueError:
            return val
    if isinstance(like, (tuple, list)):
        return type(like)(float(x) for x in val.split(","))
    return val
