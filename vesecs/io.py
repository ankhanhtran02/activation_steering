"""Atomic file writes (every stage output is written to a temporary file and renamed, so a file that exists
is complete and a crashed stage is simply redone on resume) and small helpers."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def to_py(o):
    """JSON-safe copy (numpy scalars / arrays -> Python; NaN -> None). Unlike pilot.metrics.to_py, +-inf are kept
    (written as Infinity / -Infinity, which json.load reads back): thresholds can be infinite."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f else f
    return o


def _atomic(path, write_fn, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, mode) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path, obj, indent=None):
    data = json.dumps(to_py(obj), indent=indent, ensure_ascii=False)
    _atomic(path, lambda f: f.write(data), "w")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    _atomic(path, lambda f: f.writelines(json.dumps(to_py(r), ensure_ascii=False) + "\n" for r in rows), "w")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_py(row), ensure_ascii=False) + "\n")


def save_npz(path, **arrays):
    path = Path(path)
    if not path.name.endswith(".npz"):
        raise ValueError("save_npz needs a .npz path")
    _atomic(path, lambda f: np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()}), "wb")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_npy(path, arr):
    _atomic(path, lambda f: np.save(f, np.asarray(arr)), "wb")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(to_py(obj), sort_keys=True).encode()).hexdigest()


def stable_int(*parts, bits=60):
    """Deterministic non-negative integer from any parts (seeds, seeded mappings)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) & ((1 << bits) - 1)
