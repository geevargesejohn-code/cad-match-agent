"""The CAD library: add, dedupe, bulk add, rename, delete."""
from __future__ import annotations

import pytest

from core.config import path
from storage import cache, library
from tests import shapes


def test_add_and_list(bracket_step):
    result = library.add_file(bracket_step)
    assert result.ok and not result.duplicate
    item = result.item
    assert item["format"] == "step"
    assert item["name"] == "bracket"
    assert cache.normalized_mesh_path(item["id"]).exists()
    assert library.original_file(item["id"]).exists()
    assert [i["id"] for i in library.list_items()] == [item["id"]]


def test_same_file_is_not_added_twice(bracket_step):
    first = library.add_file(bracket_step)
    second = library.add_file(bracket_step)
    assert second.duplicate and second.item["id"] == first.item["id"]
    assert len(library.list_items()) == 1


def test_bulk_add_survives_bad_files(cad_dir, bracket_step):
    good = shapes.export(shapes.cylinder_mesh(), cad_dir / "cyl.obj")
    bad = cad_dir / "corrupt.stl"
    bad.write_bytes(b"\x00junk" * 20)
    unsupported = cad_dir / "notes.txt"
    unsupported.write_text("hi")

    results = library.add_many([bracket_step, bad, good, unsupported])
    assert [r.ok for r in results] == [True, False, True, False]
    assert len(library.list_items()) == 2
    # a failed file leaves nothing behind
    assert len([d for d in path("library_dir").iterdir() if d.is_dir()]) == 2


def test_upload_keeps_original_filename(cad_dir):
    tmp_upload = shapes.export(shapes.l_bracket_mesh(), cad_dir / "tmp8f3a.stl")
    item = library.add_file(tmp_upload, filename="mounting_bracket.stl").item
    assert item["name"] == "mounting_bracket"
    assert item["filename"] == "mounting_bracket.stl"


def test_rename(bracket_step):
    item = library.add_file(bracket_step).item
    assert library.rename(item["id"], "  Motor mount  ")["name"] == "Motor mount"
    with pytest.raises(ValueError):
        library.rename(item["id"], "   ")
    with pytest.raises(KeyError):
        library.rename("missing", "x")


def test_delete_removes_everything(bracket_step):
    item = library.add_file(bracket_step).item
    assert library.delete(item["id"])
    assert library.list_items() == []
    assert not (path("library_dir") / item["id"]).exists()
    assert not (path("cache_dir") / item["id"]).exists()


def test_processors_run_on_new_items(cad_dir):
    seen = []
    library.register_processor(lambda item_id, mesh: seen.append((item_id, len(mesh.faces))))
    try:
        item = library.add_file(shapes.export(shapes.cylinder_mesh(), cad_dir / "c.ply")).item
        assert seen and seen[0][0] == item["id"]
        assert item["status"] == "ready"
    finally:
        library._PROCESSORS.clear()


def test_failing_processor_flags_item(cad_dir):
    def boom(item_id, mesh):
        raise RuntimeError("renderer crashed")

    library.register_processor(boom)
    try:
        item = library.add_file(shapes.export(shapes.cylinder_mesh(), cad_dir / "c.stl")).item
        assert item["status"] == "error"
        assert "renderer crashed" in item["error"]
    finally:
        library._PROCESSORS.clear()
