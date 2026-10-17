"""Size must never matter: the same design at a different scale is a MATCH."""
from __future__ import annotations

import cv2

from agent.loop import run_test
from cad_processing.loader import load_cad
from matching import fusion
from matching.features import compute_photo_features, compute_view_features
from cad_processing import renderer
from cad_processing.normalize import normalize_mesh
from matching.photo import analyze_photo
from matching.scorer import score_photo
from scripts import make_samples
from tests.conftest import SAMPLES


def test_photo_of_10x_larger_object_matches_small_cad(shared_library, photos):
    """The library holds a 50 mm flange; the photo shows a 500 mm one."""
    r = run_test([photos / "flange_500mm_photo.jpg"], "reference", shared_library["flange_50mm.step"])
    assert r["verdict"] == fusion.MATCH, r["explanation"]


def test_photo_of_small_object_matches_10x_larger_cad(shared_library, tmp_path):
    """Reverse direction: the reference CAD is 10x larger than the photographed part."""
    from storage import library

    big = library.add_file(SAMPLES / "cad" / "flange_500mm.step").item
    try:
        img = make_samples.synth_photo(load_cad(SAMPLES / "cad" / "flange_50mm.step"), seed=31)
        p = tmp_path / "small_flange.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        r = run_test([p], "reference", big["id"])
        # the small and the 10x flange are the same design, so either one fitting is a match
        assert r["verdict"] == fusion.MATCH, r["explanation"]
    finally:
        library.delete(big["id"])


def test_scale_gives_identical_scores(cad_dir, photos):
    """Scores for a 1x and a 40x copy of the same CAD design are the same."""
    pf = compute_photo_features(analyze_photo(photos / "plate_2holes_photo.jpg"))
    dirs = renderer.fibonacci_sphere(24)
    scores = []
    for s in (1.0, 40.0):
        path = make_samples._occ_part(make_samples.plate_holes(2, s), cad_dir / f"p{s}.step")
        views = compute_view_features(normalize_mesh(load_cad(path)), dirs)
        scores.append(score_photo(pf, views).score)
    assert abs(scores[0] - scores[1]) < 0.02
