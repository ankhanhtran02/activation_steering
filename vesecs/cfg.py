"""Configuration (configs/config.yaml), model specs (pilot/config.py) and the run-folder layout.

Run folder (spec section 12), one per run: outputs_ves/run_<UTC>/
    configs/config.yaml            copy of the config used
    configs/frozen_config.json     written in phase D
    runs/<model>/<split>/          per-question records (JSON / JSONL) + state arrays (float16 .npy / .npz)
    results/summary.json           all metrics with CIs
    results/verdicts.json          VES and ECS verdicts + inputs to each rule
    results/results_report.tex     report, structured like the SQ2 report
    audit/human_check_<model>.csv  100 outputs per model that differ from A1
    logs/timing.jsonl              wall-clock per stage, projected end time
"""
import copy
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ["VES_CONFIG"]) if os.environ.get("VES_CONFIG") else ROOT / "configs" / "config.yaml"
SMOKE = os.environ.get("VES_SMOKE") == "1"
SPLITS = ("dev", "test")


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if SMOKE:
        cfg = _merge(cfg, cfg.get("smoke_overrides", {}))
    cfg["smoke"] = SMOKE
    if os.environ.get("VES_SQ2_RUN_DIR"):
        cfg["paths"]["sq2_run_dir"] = os.environ["VES_SQ2_RUN_DIR"]
    return cfg


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def model_specs(cfg):
    """Specs (hf_id, pinned revision, layer counts) come from the SQ2 pilot config, in config order."""
    from pilot import config as PC
    by_name = {m["name"]: m for m in PC.MODELS}
    if SMOKE:
        return list(PC.MODELS)
    missing = [n for n in cfg["models"] if n not in by_name]
    if missing:
        raise SystemExit(f"models {missing} are not in pilot/config.py MODELS")
    return [by_name[n] for n in cfg["models"]]


def depth_layer(depth, n_layers):
    """SQ2 convention (pilot.config.layer_plan): depth d -> layer round(d * n_layers); layer L is
    model.model.layers[L], whose output residual is hidden_states[L + 1]."""
    return int(round(depth * n_layers))


class RunDir:
    def __init__(self, cfg, out_dir=None):
        if out_dir is None and os.environ.get("VES_OUT_DIR"):
            out_dir = os.environ["VES_OUT_DIR"]
        if out_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = resolve(cfg["paths"]["out_root"]) / f"run_{stamp}"
        self.root = Path(out_dir)
        for sub in ("configs", "runs", "results", "audit", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        dst = self.root / "configs" / "config.yaml"
        if not dst.exists():
            shutil.copyfile(CONFIG_PATH, dst)

    def split_dir(self, model, split):
        d = self.root / "runs" / model / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def frozen_path(self):
        return self.root / "configs" / "frozen_config.json"

    @property
    def results(self):
        return self.root / "results"

    @property
    def logs(self):
        return self.root / "logs"

    @property
    def audit(self):
        return self.root / "audit"
