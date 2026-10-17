"""Remove scale and position from a mesh, and describe its shape proportions.

This is what makes size irrelevant: every model is centered at the origin and
scaled so it fits exactly inside a sphere of radius 1. A 10 mm part and a
1 m part with the same design end up identical after this step.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import trimesh


@dataclass
class ShapeStats:
    n_vertices: int
    n_faces: int
    watertight: bool
    # Principal extents divided by the largest one, sorted: (1.0, b, c).
    # Scale-invariant, so it is directly comparable between a model and a photo.
    proportions: tuple[float, float, float]
    original_extents: tuple[float, float, float]  # in the file's own units, for display only

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy centered at the origin and scaled to a unit bounding sphere."""
    m = mesh.copy()
    center = m.bounding_box.centroid
    m.apply_translation(-center)
    radius = float(np.linalg.norm(m.vertices, axis=1).max())
    if radius <= 0:
        raise ValueError("Cannot normalize a mesh with zero size")
    m.apply_scale(1.0 / radius)
    return m


def principal_proportions(mesh: trimesh.Trimesh, n_samples: int = 20000) -> tuple[float, float, float]:
    """Shape proportions along the part's own principal axes, independent of scale and rotation.

    Points are sampled evenly over the surface (so dense mesh regions don't bias the
    result), then PCA finds the part's natural axes and we measure its length along each.
    """
    points, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    extents = projected.max(axis=0) - projected.min(axis=0)
    extents = np.sort(extents)[::-1]
    extents = extents / extents[0]
    return tuple(round(float(x), 4) for x in extents)


def shape_stats(original: trimesh.Trimesh) -> ShapeStats:
    return ShapeStats(
        n_vertices=int(len(original.vertices)),
        n_faces=int(len(original.faces)),
        watertight=bool(original.is_watertight),
        proportions=principal_proportions(original),
        original_extents=tuple(round(float(x), 4) for x in original.extents),
    )
