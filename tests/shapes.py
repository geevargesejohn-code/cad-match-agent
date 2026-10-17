"""Builds small test parts in every supported format, at any scale."""
from __future__ import annotations

from pathlib import Path

import trimesh


def bracket_step(out: Path, scale: float = 1.0) -> Path:
    """A flat plate with two holes, written as a real STEP file through gmsh/OpenCascade."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("bracket")
        occ = gmsh.model.occ
        plate = occ.addBox(0, 0, 0, 40 * scale, 20 * scale, 5 * scale)
        hole1 = occ.addCylinder(10 * scale, 10 * scale, -1, 0, 0, 7 * scale + 2, 3 * scale)
        hole2 = occ.addCylinder(30 * scale, 10 * scale, -1, 0, 0, 7 * scale + 2, 3 * scale)
        occ.cut([(3, plate)], [(3, hole1), (3, hole2)])
        occ.synchronize()
        gmsh.write(str(out))
    finally:
        gmsh.finalize()
    return out


def l_bracket_mesh(scale: float = 1.0) -> trimesh.Trimesh:
    """An L-shaped part built from two boxes."""
    base = trimesh.creation.box(extents=[40, 20, 5])
    base.apply_translation([20, 10, 2.5])
    wall = trimesh.creation.box(extents=[5, 20, 30])
    wall.apply_translation([2.5, 10, 15])
    mesh = trimesh.util.concatenate([base, wall])
    mesh.apply_scale(scale)
    return mesh


def cylinder_mesh(scale: float = 1.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=10, height=50, sections=48)
    mesh.apply_scale(scale)
    return mesh


def export(mesh: trimesh.Trimesh, out: Path) -> Path:
    mesh.export(out)
    return out
