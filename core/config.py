"""Loads config.yaml once and resolves data paths relative to the repo root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Read the config file. Set CAD_MATCH_CONFIG to point at a different one."""
    cfg_path = Path(os.environ.get("CAD_MATCH_CONFIG", ROOT / "config.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # CAD_MATCH_DATA_DIR relocates all data (used by tests and Docker volumes).
    data_override = os.environ.get("CAD_MATCH_DATA_DIR")
    paths = cfg.setdefault("paths", {})
    if data_override:
        base = Path(data_override)
        paths["data_dir"] = str(base)
        paths["library_dir"] = str(base / "library")
        paths["cache_dir"] = str(base / "cache")
        paths["db_file"] = str(base / "cad_match.db")
        paths["uploads_dir"] = str(base / "uploads")
        paths["results_dir"] = str(base / "results")

    # Quick backend switches without editing the file
    if os.environ.get("CAD_MATCH_SEGMENTATION"):
        cfg.setdefault("photo", {})["segmentation_backend"] = os.environ["CAD_MATCH_SEGMENTATION"]
    if os.environ.get("CAD_MATCH_EMBEDDING"):
        cfg.setdefault("matching", {})["embedding_backend"] = os.environ["CAD_MATCH_EMBEDDING"]

    for key, value in list(paths.items()):
        p = Path(value)
        paths[key] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def path(key: str) -> Path:
    """Convenience accessor, e.g. path('library_dir')."""
    return Path(get_config()["paths"][key])


def reset_config_cache() -> None:
    """Forget the cached config (tests use this after changing env vars)."""
    get_config.cache_clear()
