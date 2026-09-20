from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    artifacts = Path(cfg["project"]["artifacts_dir"])
    runs = Path(cfg["project"]["runs_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    return artifacts, runs


def year_range(pair: list[int]) -> list[int]:
    return list(range(int(pair[0]), int(pair[1]) + 1))


def split_years(cfg: dict[str, Any]) -> dict[str, list[int]]:
    d = cfg["data"]
    return {
        "train": year_range(d["train_years"]),
        "val": year_range(d["val_years"]),
        "test": year_range(d["test_years"]),
        "ood": year_range(d["ood_years"]),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

