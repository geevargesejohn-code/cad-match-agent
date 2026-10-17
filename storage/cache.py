"""Where precomputed data for each library item lives on disk.

data/cache/<item_id>/
    normalized.ply   unit-scale mesh (Phase 1)
    thumbnail.png    preview image  (Phase 2)
    views.npz        renders, silhouettes, camera poses (Phase 2)
    features.npz     embeddings + shape descriptors   (Phase 4)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from core.config import path


def item_dir(item_id: str) -> Path:
    d = path("cache_dir") / item_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalized_mesh_path(item_id: str) -> Path:
    return item_dir(item_id) / "normalized.ply"


def thumbnail_path(item_id: str) -> Path:
    return item_dir(item_id) / "thumbnail.png"


def views_path(item_id: str) -> Path:
    return item_dir(item_id) / "views.npz"


def features_path(item_id: str) -> Path:
    return item_dir(item_id) / "features.npz"


def remove(item_id: str) -> None:
    shutil.rmtree(path("cache_dir") / item_id, ignore_errors=True)
