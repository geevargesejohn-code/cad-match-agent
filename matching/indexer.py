"""Builds and loads the per-model view index, and hooks it into the CAD library."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import trimesh

from cad_processing import renderer
from core.config import get_config
from matching.embeddings import get_embedder
from matching.features import ViewFeatures, compute_view_features
from storage import cache, db, library

log = logging.getLogger(__name__)


def _refined_path(item_id: str):
    return cache.item_dir(item_id) / "features_refined.npz"


def index_model(item_id: str, mesh: trimesh.Trimesh) -> ViewFeatures:
    """Render the default views, store features and a thumbnail. Runs on every upload."""
    n = int(get_config()["rendering"]["num_views"])
    feats = compute_view_features(mesh, renderer.fibonacci_sphere(n))
    feats.save(cache.features_path(item_id))
    _refined_path(item_id).unlink(missing_ok=True)
    thumb = renderer.thumbnail(mesh)
    cv2.imwrite(str(cache.thumbnail_path(item_id)), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    return feats


def load_features(item_id: str, include_refined: bool = True) -> ViewFeatures:
    """Load a model's view features, re-indexing if missing or built with another embedder."""
    feats = ViewFeatures.load(cache.features_path(item_id))
    if feats is None or feats.backend != get_embedder().name:
        log.info("Re-indexing %s", item_id)
        feats = index_model(item_id, library.load_normalized(item_id))
        db.update_item(item_id, status="ready", error=None)
    if include_refined:
        refined = ViewFeatures.load(_refined_path(item_id))
        if refined is not None and refined.backend == feats.backend:
            feats = feats.concat(refined)
    return feats


def refine_model(item_id: str) -> ViewFeatures:
    """Add a denser set of viewpoints for one model (used when the agent is unsure)."""
    existing = ViewFeatures.load(_refined_path(item_id))
    if existing is None or existing.backend != get_embedder().name:
        n = int(get_config()["rendering"]["refine_views"])
        existing = compute_view_features(library.load_normalized(item_id), renderer.refine_directions(n))
        existing.save(_refined_path(item_id))
    return load_features(item_id, include_refined=True)


def is_refined(item_id: str) -> bool:
    return _refined_path(item_id).exists()


def render_view(item_id: str, direction: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    mesh = renderer.prepare_mesh(library.load_normalized(item_id))
    return renderer.render(mesh, direction, size=size, color=(120, 170, 225))


def reindex_all() -> int:
    count = 0
    for item in library.list_items():
        try:
            index_model(item["id"], library.load_normalized(item["id"]))
            db.update_item(item["id"], status="ready", error=None)
            count += 1
        except Exception as e:
            db.update_item(item["id"], status="error", error=str(e))
    return count


def install() -> None:
    """Make every new library upload get indexed automatically."""
    library.register_processor(index_model)
