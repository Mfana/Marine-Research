"""Configuration, logging, seeding and resource-estimation helpers."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(settings: dict | None = None, name: str = "marine") -> logging.Logger:
    settings = settings or load_settings()
    log_dir = PROJECT_ROOT / settings["project"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(settings["project"]["log_level"])
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        fh = logging.FileHandler(log_dir / f"{datetime.now():%Y%m%d}.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def estimate_uncompressed_mb(n_time: int, n_lat: int, n_lon: int, n_depth: int = 1,
                             n_vars: int = 1, bytes_per_value: int = 4) -> float:
    """Upper-bound size of the selected values in memory (float32 by default).

    The toolbox delivers float64, so use bytes_per_value=8 for in-memory size.
    Network transfer can be lower (compression) or higher (whole Zarr chunks are
    fetched even when only part of a chunk is selected).
    """
    return n_time * n_lat * n_lon * n_depth * n_vars * bytes_per_value / 1e6


def grid_cells(min_v: float, max_v: float, step: float = 1 / 12) -> int:
    return int(round((max_v - min_v) / step)) + 1


def months_between(start: str, end: str) -> int:
    s, e = np.datetime64(start, "M"), np.datetime64(end, "M")
    return int((e - s).astype(int)) + 1
