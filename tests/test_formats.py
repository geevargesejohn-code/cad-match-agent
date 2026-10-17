"""Every supported format must load into a usable mesh; bad input must fail cleanly."""
from __future__ import annotations

import pytest

from cad_processing.loader import CADLoadError, load_cad
from tests import shapes


@pytest.mark.parametrize("ext", ["stl", "obj", "ply"])
def test_mesh_formats_load(cad_dir, ext):
    path = shapes.export(shapes.l_bracket_mesh(), cad_dir / f"part.{ext}")
    mesh = load_cad(path)
    assert len(mesh.faces) > 0
    assert mesh.extents.max() == pytest.approx(40, rel=0.01)


@pytest.mark.parametrize("ext", ["step", "stp"])
def test_step_formats_load(cad_dir, ext):
    path = shapes.bracket_step(cad_dir / f"bracket.{ext}")
    mesh = load_cad(path)
    assert len(mesh.faces) > 100          # holes need a reasonably fine mesh
    assert mesh.extents == pytest.approx([40, 20, 5], rel=0.02)


def test_uppercase_extension_loads(cad_dir):
    path = shapes.export(shapes.cylinder_mesh(), cad_dir / "PART.STL")
    assert len(load_cad(path).faces) > 0


def test_unsupported_format_rejected(cad_dir):
    bad = cad_dir / "drawing.dwg"
    bad.write_bytes(b"not a mesh")
    with pytest.raises(CADLoadError, match="Unsupported format"):
        load_cad(bad)


def test_corrupt_file_rejected(cad_dir):
    bad = cad_dir / "broken.stl"
    bad.write_bytes(b"\x00garbage" * 10)
    with pytest.raises(CADLoadError):
        load_cad(bad)


def test_missing_file_rejected(cad_dir):
    with pytest.raises(CADLoadError, match="not found"):
        load_cad(cad_dir / "nope.stl")
