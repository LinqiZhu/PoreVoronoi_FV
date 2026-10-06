"""Configuration, hashing and requeue-safe checkpoint I/O.

All paths come from one JSON config (default: config.json next to this file; override with --cfg or the PVFV_CONFIG
environment variable); relative paths in it are resolved against the config file's directory. No module in this
folder carries an absolute path.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import hashlib
import json
import os
import platform
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def thread_env(n=1):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, str(n))


def load_cfg(path=None):
    path = path or os.environ.get("PVFV_CONFIG") or os.path.join(HERE, "config.json")
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = os.path.dirname(path)

    def res(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["_path"] = path
    cfg["_base"] = base
    cfg["in_dir"] = res(cfg["in_dir"])
    cfg["out_dir"] = res(cfg["out_dir"])
    return cfg


def case_paths(cfg, code):
    c = cfg["cases"][code]
    j = lambda name: os.path.join(cfg["in_dir"], name)  # noqa: E731
    return dict(reference=j(c["reference"]), window=j(c["window"]), state=j(c["state"]), manifest=j(c["manifest"]))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)  # requeue-safe: an existing run folder is resumed, never an error
    return path


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return "NaN"
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def save_json(path, obj):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_jsonable(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)  # atomic: a preempted writer leaves no half file under the final name


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npz(path, **arrays):
    import numpy as np
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def done(path, validate=None):
    """True when a checkpoint exists and (optionally) validates; a corrupt checkpoint is renamed aside."""
    if not os.path.exists(path):
        return False
    try:
        if validate is not None:
            validate(path)
        return True
    except Exception as exc:  # noqa: BLE001
        bad = f"{path}.bad{int(time.time())}"
        os.replace(path, bad)
        print(dict(stage="checkpoint_invalid", path=path, moved_to=bad, error=repr(exc)), flush=True)
        return False


def environment():
    import numpy as np
    import scipy
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), node=platform.node(),
                threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                slurm={k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_RESTART_COUNT", "SLURM_CPUS_PER_TASK")},
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def code_hashes():
    names = sorted(n for n in os.listdir(HERE) if n.endswith(".py") or n.endswith(".sh"))
    return {n: sha(os.path.join(HERE, n)) for n in names}


class Beat:
    """Heartbeat: one line every `every` seconds or `each` events, carrying an index and elapsed time."""

    def __init__(self, label, every=60.0, each=None):
        self.label, self.every, self.each = label, float(every), each
        self.t0 = self.last = time.perf_counter()
        self.n = 0

    def __call__(self, *_):
        self.n += 1
        now = time.perf_counter()
        if (self.each and self.n % self.each == 0) or now - self.last >= self.every:
            self.last = now
            print(dict(beat=self.label, i=self.n, elapsed_s=round(now - self.t0, 2)), flush=True)
