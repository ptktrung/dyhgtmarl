"""Shared helpers: seeding, IO, paths, and light configuration loading."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"

# The replication cohort runs the same pipeline on a second parameterisation.
# Setting DYHGT_CITY=kl redirects every processed artifact and result table to
# its own directory, so the two cohorts never overwrite one another.
CITY = os.environ.get("DYHGT_CITY", "hcmc").lower()
# DYHGT_REGIME=stress selects the revised generator of Section VI-G.
REGIME = os.environ.get("DYHGT_REGIME", "primary").lower()
_sfx = "" if CITY == "hcmc" else f"_{CITY}"
if REGIME != "primary":
    _sfx += f"_{REGIME}"
PROC = ROOT / "data" / f"processed{_sfx}"
TAB = RES / f"tables{_sfx}"
FIG = RES / f"figures{_sfx}"

RAW_FILES = {
    "hcmc": {"traffic": "hcmc_traffic_data.csv",
             "iot": "HCMC_IoT_Graph_Sensor_Data.csv",
             "transactions": "hcmc_mixed_mobility_data.csv"},
    "kl": {"traffic": "kl_traffic_data.csv",
           "iot": "KL_IoT_Graph_Sensor_Data.csv",
           "transactions": "kl_mixed_mobility_data.csv"},
}


def raw_file(kind: str) -> Path:
    """Path of a raw input for the active cohort."""
    return RAW / RAW_FILES[CITY][kind]


def ensure_dirs() -> None:
    for p in (PROC, TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)


def load_config(name: str = "default.yaml") -> dict:
    with open(ROOT / "configs" / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(1)
    except Exception:
        pass


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)


def require(path, hint: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required input not found: {p}. {hint}"
        )
    return p
