"""Identity checks for resumable experiments; paths do not identify data."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PROTOCOL = "v12r1"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


@lru_cache(maxsize=1)
def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))}
    entry = root.parent / "scripts" / "train.py"
    files["scripts/train.py"] = file_sha256(entry)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def run_identity(cfg, dataset_path, dataset_sha256=None):
    config = cfg.to_dict()
    # Only operational settings may change on restart. Diagnostic sampling
    # consumes RNG, so its settings deliberately remain part of the identity.
    for key in ("device", "resume", "lock_stale_s", "progress_every", "ckpt_every"):
        config["train"].pop(key, None)
    payload = dict(protocol=PROTOCOL, source_sha256=source_identity(),
                   dataset_sha256=dataset_sha256 or file_sha256(dataset_path), config=config)
    payload = json.loads(json.dumps(payload))  # canonicalize tuples to lists
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def ensure_manifest(run_dir, identity):
    """Call under the run lock, before overwriting any config or checkpoint."""
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"{run_dir}: dataset/config/protocol changed; use a fresh --out_dir")
    else:
        existing = [p for p in run_dir.iterdir() if p.name != "RUNNING.lock"]
        if existing:
            raise ValueError(f"{run_dir}: legacy/unverified files; use a fresh --out_dir")
        atomic_json(path, identity)
