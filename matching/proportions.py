"""Proportion and feature agreement between two silhouettes.

Inputs are shape_features() vectors: [aspect, solidity, holes].
"""
from __future__ import annotations

import numpy as np


def proportion_similarity(photo: np.ndarray, views: np.ndarray) -> np.ndarray:
    """Similarity in [0, 1] between one photo feature vector and V view vectors -> (V,)."""
    views = np.atleast_2d(views)
    aspect_diff = np.abs(views[:, 0] - photo[0])
    solidity_diff = np.abs(views[:, 1] - photo[1])
    aspect_sim = np.clip(1 - aspect_diff / 0.35, 0, 1)
    solidity_sim = np.clip(1 - solidity_diff / 0.25, 0, 1)

    # Hole count is a hard design feature: 2 holes vs 3 holes is a different part.
    # Asymmetric: a photo can easily MISS a hole (not see-through from that angle, poor
    # light), but it cannot show a hole the design does not have.
    extra = photo[2] - views[:, 2]          # > 0: photo shows more holes than the model
    hole_factor = np.select(
        [extra == 0, extra == -1, extra <= -2, extra == 1],
        [1.0, 0.8, 0.55, 0.35], 0.12)
    return (0.5 * aspect_sim + 0.5 * solidity_sim) * hole_factor


def describe(photo: np.ndarray, view: np.ndarray) -> str:
    return (f"aspect {photo[0]:.2f} vs {view[0]:.2f}, solidity {photo[1]:.2f} vs {view[1]:.2f}, "
            f"visible holes {int(photo[2])} vs {int(view[2])}")
