"""Configuration loading, path resolution, seeding and run metadata.

All experiment choices live in ``configs/*.yaml``.  Everything is resolved
relative to the repository root so no machine-specific paths are ever needed.
"""
from __future__ import annotations

import argparse
import copy
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stream ids keep the random streams of different tasks independent while
# still deriving everything from the single master seed.
STREAM_SIMULATION = 0
STREAM_TASK3 = 1
STREAM_TASK4 = 2


@dataclass(frozen=True)
class Paths:
    """Resolved output/input locations (all inside the repository by default)."""

    root: Path
    raw_data: Path
    outputs: Path
    figures: Path
    tables: Path
    predictions: Path
    logs: Path

    def rel(self, p: Path | str) -> str:
        """Path relative to the repository root when possible (keeps logs machine-independent)."""
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return Path(p).name

    def make_dirs(self) -> None:
        for p in (self.figures, self.tables, self.predictions, self.logs):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration (path relative to the repo root or cwd)."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_paths(cfg: dict[str, Any], output_dir: str | Path | None = None,
                  raw_dir: str | Path | None = None) -> Paths:
    """Resolve input/output directories; CLI overrides win over the config."""
    out = Path(output_dir) if output_dir else Path(cfg["paths"]["outputs"])
    raw = Path(raw_dir) if raw_dir else Path(cfg["paths"]["raw_data"])
    out = out if out.is_absolute() else REPO_ROOT / out
    raw = raw if raw.is_absolute() else REPO_ROOT / raw
    paths = Paths(root=REPO_ROOT, raw_data=raw, outputs=out, figures=out / "figures",
                  tables=out / "tables", predictions=out / "predictions", logs=out / "logs")
    paths.make_dirs()
    return paths


def child_rng(seed: int, stream: int, index: int = 0) -> np.random.Generator:
    """Deterministic child generator: depends only on (seed, stream, index).

    Replication ``i`` therefore produces identical data regardless of how many
    replications are requested, which is what lets Task 2 reuse Task 1's data.
    """
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(stream, index))
    return np.random.default_rng(ss)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Common CLI shared by all task scripts."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    ap.add_argument("--output-dir", default=None, help="override outputs directory")
    return ap


def setup_logging(log_dir: Path, name: str) -> logging.Logger:
    """Log to console and to ``<log_dir>/<name>.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dsa5205")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log", mode="w")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def save_run_metadata(cfg: dict[str, Any], paths: Paths) -> None:
    """Persist the exact config used plus package/runtime versions."""
    import joblib, matplotlib, pandas, scipy, sklearn  # noqa: E401

    cfg_out = {k: v for k, v in copy.deepcopy(cfg).items() if not k.startswith("_")}
    with open(paths.logs / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_out, fh, sort_keys=False)
    versions = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }
    with open(paths.logs / "environment.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(versions, fh, sort_keys=False)
