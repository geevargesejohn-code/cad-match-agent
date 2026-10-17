"""The CPU renderer: silhouettes, holes, viewpoint coverage, size independence."""
from __future__ import annotations

import cv2
import numpy as np

from cad_processing import renderer
from cad_processing.loader import load_cad
from cad_processing.normalize import normalize_mesh
from matching.silhouette import shape_features
from tests import shapes


def test_directions_cover_the_sphere():
    d = renderer.fibonacci_sphere(60)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    assert d[:, 2].min() < -0.9 and d[:, 2].max() > 0.9   # sees top and bottom
    assert not np.allclose(renderer.refine_directions(72)[:5], renderer.fibonacci_sphere(72)[:5])


def test_render_produces_object_and_mask():
    m = normalize_mesh(shapes.l_bracket_mesh())
    rgb, mask = renderer.render(m, [1, 1, 1], size=128)
    assert rgb.shape == (128, 128, 3) and mask.shape == (128, 128)
    assert 0.05 < mask.mean() < 0.8
    far = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0  # away from anti-aliased edges
    assert (rgb[far] == 255).all()     # background stays white


def test_top_view_shows_both_holes(bracket_step):
    m = normalize_mesh(load_cad(bracket_step))
    _, mask = renderer.render(m, [0, 0, 1], size=224)
    assert shape_features(mask)[2] == 2


def test_size_does_not_change_the_render(cad_dir):
    small = normalize_mesh(load_cad(shapes.bracket_step(cad_dir / "s.step", scale=1)))
    big = normalize_mesh(load_cad(shapes.bracket_step(cad_dir / "b.step", scale=25)))
    _, m1 = renderer.render(small, [0.3, -0.5, 0.8], size=160)
    _, m2 = renderer.render(big, [0.3, -0.5, 0.8], size=160)
    iou = (m1 & m2).sum() / (m1 | m2).sum()
    assert iou > 0.97


def test_perspective_render_with_background():
    m = normalize_mesh(shapes.cylinder_mesh())
    bg = np.zeros((100, 200, 3), np.uint8)
    rgb, mask = renderer.render(m, [0.2, 0.3, 0.9], size=(200, 100), fov_deg=35, background=bg)
    assert rgb.shape == (100, 200, 3)
    far = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    assert (rgb[far] < 10).all()
