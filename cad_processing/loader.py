"""Load any supported CAD file into a single clean triangle mesh.

STL / OBJ / PLY are read directly by trimesh.
STEP / STP are tessellated with gmsh (OpenCascade kernel) and then read by trimesh.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
import trimesh

from core.config import get_config

MESH_FORMATS = {".stl", ".obj", ".ply"}
STEP_FORMATS = {".step", ".stp"}

# gmsh keeps global state and is not thread-safe, so STEP conversion is serialized.
_GMSH_LOCK = threading.Lock()


class CADLoadError(Exception):
    """Raised when a file cannot be turned into a usable mesh."""


def supported_formats() -> set[str]:
    return set(get_config()["cad"]["supported_formats"])


def load_cad(file_path: str | Path) -> trimesh.Trimesh:
    """Return a cleaned, single-body trimesh for any supported CAD file."""
    p = Path(file_path)
    if not p.exists():
        raise CADLoadError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext not in supported_formats():
        raise CADLoadError(
            f"Unsupported format '{ext}'. Supported: {', '.join(sorted(supported_formats()))}"
        )

    max_mb = get_config()["cad"]["max_file_mb"]
    if p.stat().st_size > max_mb * 1024 * 1024:
        raise CADLoadError(f"File is larger than the {max_mb} MB limit")

    mesh = _load_step(p) if ext in STEP_FORMATS else _load_mesh(p)
    return _clean(mesh, p.name)


def _load_mesh(p: Path) -> trimesh.Trimesh:
    try:
        # force="mesh" flattens multi-part scenes (e.g. OBJ groups) into one mesh
        loaded = trimesh.load(p, force="mesh", process=False)
    except Exception as e:  # trimesh raises many different error types
        raise CADLoadError(f"Could not read {p.name}: {e}") from e
    if not isinstance(loaded, trimesh.Trimesh):
        raise CADLoadError(f"{p.name} did not contain a triangle mesh")
    return loaded


def _load_step(p: Path) -> trimesh.Trimesh:
    try:
        import gmsh
    except Exception as e:
        raise CADLoadError(
            "STEP support needs gmsh (pip install gmsh; on Linux also libglu1-mesa)."
        ) from e

    cad_cfg = get_config()["cad"]
    with _GMSH_LOCK, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "converted.stl"
        gmsh.initialize(interruptible=False)  # no signal handler: works in server threads
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFactor", cad_cfg["step_mesh_size_factor"])
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", cad_cfg["step_curvature_points"])
            gmsh.option.setNumber("Mesh.Binary", 1)
            gmsh.model.add("step_import")
            gmsh.model.occ.importShapes(str(p))
            gmsh.model.occ.synchronize()
            if not gmsh.model.getEntities(2):
                raise CADLoadError(f"{p.name} has no surfaces to mesh")

            # Cap element size relative to the part, so tiny and huge parts mesh alike.
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 30.0)

            gmsh.model.mesh.generate(2)  # surface mesh is all we need
            gmsh.write(str(out))
        except CADLoadError:
            raise
        except Exception as e:
            raise CADLoadError(f"Could not convert STEP file {p.name}: {e}") from e
        finally:
            gmsh.finalize()
        return _load_mesh(out)


def _clean(mesh: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise CADLoadError(f"{name} has no valid faces after cleaning")
    if not np.all(np.isfinite(mesh.vertices)):
        raise CADLoadError(f"{name} contains invalid (NaN/inf) coordinates")
    if mesh.extents.max() <= 0:
        raise CADLoadError(f"{name} has zero size")
    return mesh
