"""The CAD library: add, list, rename and delete CAD files.

Adding a file:
  1. checks the format and skips exact duplicates (same file content)
  2. copies the original into data/library/<id>/
  3. loads it into a mesh (STEP is converted) and records its shape stats
  4. saves a scale-normalized copy in data/cache/<id>/
  5. runs any registered processors (rendering in Phase 2, features in Phase 4)
"""
from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import trimesh

from cad_processing.loader import CADLoadError, load_cad, supported_formats
from cad_processing.normalize import normalize_mesh, shape_stats
from core.config import path
from storage import cache, db

# Later phases register extra steps that run on each new model: fn(item_id, normalized_mesh)
Processor = Callable[[str, trimesh.Trimesh], None]
_PROCESSORS: list[Processor] = []


def register_processor(fn: Processor) -> None:
    if fn not in _PROCESSORS:
        _PROCESSORS.append(fn)


@dataclass
class AddResult:
    filename: str
    ok: bool
    item: dict | None = None
    duplicate: bool = False
    error: str | None = None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_file(source: str | Path, name: str | None = None, filename: str | None = None) -> AddResult:
    """Add one CAD file. `filename` lets callers keep the user's original name
    when the file arrives at a temporary upload path."""
    src = Path(source)
    filename = filename or src.name
    ext = Path(filename).suffix.lower()

    if ext not in supported_formats():
        return AddResult(filename, ok=False, error=f"Unsupported format '{ext}'")
    if not src.exists():
        return AddResult(filename, ok=False, error="File not found")

    digest = _sha256(src)
    existing = db.find_by_hash(digest)
    if existing:
        return AddResult(filename, ok=True, item=existing, duplicate=True)

    item_id = uuid.uuid4().hex[:12]
    dest_dir = path("library_dir") / item_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    shutil.copy2(src, dest)

    try:
        mesh = load_cad(dest)
        stats = shape_stats(mesh)
        normalized = normalize_mesh(mesh)
        normalized.export(cache.normalized_mesh_path(item_id))
    except (CADLoadError, ValueError) as e:
        shutil.rmtree(dest_dir, ignore_errors=True)
        cache.remove(item_id)
        return AddResult(filename, ok=False, error=str(e))

    db.insert_item({
        "id": item_id,
        "name": name or Path(filename).stem,
        "filename": dest.name,
        "format": ext.lstrip("."),
        "sha256": digest,
        "uploaded_at": db.now_iso(),
        "n_vertices": stats.n_vertices,
        "n_faces": stats.n_faces,
        "watertight": int(stats.watertight),
        "proportions": list(stats.proportions),
        "original_extents": list(stats.original_extents),
        "status": "loaded",
        "error": None,
    })

    try:
        for processor in _PROCESSORS:
            processor(item_id, normalized)
        db.update_item(item_id, status="ready" if _PROCESSORS else "loaded")
    except Exception as e:  # keep the item, but flag it so it can be reprocessed
        db.update_item(item_id, status="error", error=f"Processing failed: {e}")

    return AddResult(filename, ok=True, item=db.get_item(item_id))


def add_many(sources: list[str | Path]) -> list[AddResult]:
    """Bulk add. One bad file never stops the rest."""
    return [add_file(s) for s in sources]


def list_items() -> list[dict]:
    return db.list_items()


def get_item(item_id: str) -> dict | None:
    return db.get_item(item_id)


def rename(item_id: str, new_name: str) -> dict:
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("Name cannot be empty")
    if not db.get_item(item_id):
        raise KeyError(f"No library item with id {item_id}")
    db.update_item(item_id, name=new_name)
    return db.get_item(item_id)


def delete(item_id: str) -> bool:
    removed = db.delete_item(item_id)
    shutil.rmtree(path("library_dir") / item_id, ignore_errors=True)
    cache.remove(item_id)
    return removed


def original_file(item_id: str) -> Path | None:
    item = db.get_item(item_id)
    if not item:
        return None
    return path("library_dir") / item_id / item["filename"]


def load_normalized(item_id: str) -> trimesh.Trimesh:
    return trimesh.load(cache.normalized_mesh_path(item_id), force="mesh")
