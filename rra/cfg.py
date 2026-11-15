"""Configuration (configs/rra_config.yaml), model specs and the run-folder layout.

Run folder (spec section 10, "Layout"), one per run: outputs_rra/run_<UTC>/
    configs/config.yaml, configs/frozen_config.json
    runs/<model>/pool/                   A2 for the pool, R (cross-fitted + full), Platt fits, check results
    runs/<model>/{dev_ves,test_ves}/     VES/ECS dev (tuning) and test (replay): re-teacher-forced A1 states
    runs/<model>/{dev_fresh,test_fresh}/ per-question JSON / JSONL + state arrays (float16)
    runs/questions.json                  fresh dev and test questions
    results/summary.json, results/verdicts.json, results/results_report.tex, results/integrity.json
    audit/human_check_<model>.csv        100 routed retries per model
    logs/timing.jsonl                    measured rates, projections, any cuts
"""
import copy
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ["RRA_CONFIG"]) if os.environ.get("RRA_CONFIG") else ROOT / "configs" / "rra_config.yaml"
SMOKE = os.environ.get("RRA_SMOKE") == "1"

# split names: (folder, role)
FRESH = ("dev_fresh", "test_fresh")
VES = ("dev_ves", "test_ves")
TUNING = ("dev_ves", "dev_fresh")         # the tuning set = VES/ECS dev (600) + fresh dev (1,000)


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
    if os.environ.get("RRA_VES_RUN_DIR"):
        cfg["paths"]["ves_run_dir"] = os.environ["RRA_VES_RUN_DIR"]
    if os.environ.get("RRA_SQ2_RUN_DIR"):
        cfg["paths"]["sq2_run_dir"] = os.environ["RRA_SQ2_RUN_DIR"]
    else:
        txt = resolve(cfg["paths"]["ves_run_dir"]) / "sq2_run_dir.txt"
        if txt.exists() and Path(txt.read_text().strip()).exists():
            cfg["paths"]["sq2_run_dir"] = txt.read_text().strip()
    for k in ("E2", "G"):                 # RRA_SKIP_E2=1 / RRA_SKIP_G=1 switch an optional stage off
        if os.environ.get(f"RRA_SKIP_{k}") == "1":
            cfg["optional"][k] = False
    return cfg


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def model_specs(cfg):
    """Specs (hf_id, pinned revision, layer counts) of the two main models come from the SQ2 pilot config."""
    from pilot import config as PC
    if SMOKE:
        return list(PC.MODELS)
    by_name = {m["name"]: m for m in PC.MODELS}
    missing = [n for n in cfg["models"] if n not in by_name]
    if missing:
        raise SystemExit(f"models {missing} are not in pilot/config.py MODELS")
    return [by_name[n] for n in cfg["models"]]


def gemma_spec(cfg):
    g = dict(cfg["gemma"])
    g["family"] = "gemma2"
    return g


def depth_layer(depth, n_layers):
    """SQ2 convention (pilot.config.layer_plan): depth d -> layer round(d * n_layers)."""
    return int(round(depth * n_layers))


def record_layers(cfg, n_layers):
    return sorted({depth_layer(d, n_layers) for d in cfg["depths"]})


class RunDir:
    def __init__(self, cfg, out_dir=None):
        if out_dir is None and os.environ.get("RRA_OUT_DIR"):
            out_dir = os.environ["RRA_OUT_DIR"]
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
