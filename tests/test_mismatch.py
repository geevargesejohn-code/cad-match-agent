"""A different design is NOT A MATCH, even when it has the same size; and the right
design is found when searching the library."""
from __future__ import annotations

import pytest

from agent.loop import run_test
from matching import fusion


def test_same_size_different_design_is_not_a_match(shared_library, photos):
    """3-hole plate photographed, 2-hole plate as reference: same size, same outline, different design."""
    r = run_test([photos / "plate_3holes_photo.jpg"], "reference", shared_library["plate_2holes.step"])
    assert r["verdict"] == fusion.NOT_A_MATCH, r["explanation"]


def test_completely_different_design_is_not_a_match(shared_library, photos):
    r = run_test([photos / "l_bracket_photo.jpg"], "reference", shared_library["sprocket.stp"])
    assert r["verdict"] == fusion.NOT_A_MATCH, r["explanation"]


def test_matching_photo_and_cad(shared_library, photos):
    r = run_test([photos / "plate_2holes_photo.jpg"], "reference", shared_library["plate_2holes.step"])
    assert r["verdict"] == fusion.MATCH, r["explanation"]
    assert r["comparison_image"] and r["best_match"]["name"] == "plate_2holes"


@pytest.mark.parametrize("photo,design", [
    ("t_bracket_photo.jpg", "t_bracket.obj"),
    ("sprocket_photo.jpg", "sprocket.stp"),
    ("hex_nut_photo.jpg", "hex_nut.ply"),
    ("l_bracket_photo.jpg", "l_bracket.stl"),
    ("plate_3holes_photo.jpg", "plate_3holes.step"),
])
def test_library_search_finds_the_right_design(shared_library, photos, photo, design):
    r = run_test([photos / photo], "library_search")
    assert r["best_match"]["id"] == shared_library[design]
    assert r["verdict"] == fusion.MATCH, r["explanation"]


def test_library_search_reports_no_match(shared_library, photos, tmp_path):
    """With the true design deleted, the search must not claim a match."""
    from storage import library
    from scripts.make_samples import DESIGNS
    from tests.conftest import SAMPLES

    tid = shared_library["t_bracket.obj"]
    library.delete(tid)
    try:
        r = run_test([photos / "t_bracket_photo.jpg"], "library_search")
        assert r["verdict"] != fusion.MATCH, r["explanation"]
    finally:
        new = library.add_file(SAMPLES / "cad" / "t_bracket.obj").item
        shared_library["t_bracket.obj"] = new["id"]
        assert "t_bracket.obj" in DESIGNS
