"""Normalization must erase size and position, and proportions must ignore size and rotation."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cad_processing.normalize import normalize_mesh, principal_proportions
from tests import shapes


@pytest.mark.parametrize("scale", [0.01, 0.5, 3.0, 250.0])
def test_scaled_copies_normalize_identically(scale):
    reference = normalize_mesh(shapes.l_bracket_mesh(1.0))
    scaled = normalize_mesh(shapes.l_bracket_mesh(scale))
    np.testing.assert_allclose(scaled.vertices, reference.vertices, atol=1e-6)


def test_normalized_mesh_fits_unit_sphere():
    m = normalize_mesh(shapes.l_bracket_mesh(37.0))
    radius = np.linalg.norm(m.vertices, axis=1).max()
    assert radius == pytest.approx(1.0)
    np.testing.assert_allclose(m.bounding_box.centroid, 0, atol=1e-9)


def test_proportions_ignore_scale():
    a = principal_proportions(shapes.l_bracket_mesh(1.0))
    b = principal_proportions(shapes.l_bracket_mesh(80.0))
    np.testing.assert_allclose(a, b, atol=0.02)


def test_proportions_ignore_rotation():
    mesh = shapes.l_bracket_mesh()
    rotated = mesh.copy()
    rotated.apply_transform(trimesh.transformations.random_rotation_matrix(seed=7))
    np.testing.assert_allclose(principal_proportions(mesh), principal_proportions(rotated), atol=0.02)


def test_different_designs_have_different_proportions():
    bracket = principal_proportions(shapes.l_bracket_mesh())
    cylinder = principal_proportions(shapes.cylinder_mesh())
    assert np.abs(np.subtract(bracket, cylinder)).max() > 0.1
