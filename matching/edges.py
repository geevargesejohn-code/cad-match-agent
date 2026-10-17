"""Edge chamfer matching: compares internal structure (hole rims, face boundaries,
steps) and outline at the same pose. Robust to lighting and color, because only
edge *positions* matter, not their brightness.
"""
from __future__ import annotations

import cv2
import numpy as np

EDGE_SIZE = 96
DT_CAP = 8.0   # pixels; distances beyond this count as "no corresponding edge"


def edge_map(canonical_gray: np.ndarray) -> np.ndarray:
    """Edges of a canonical (contrast-normalized, object-on-white) crop -> bool (96,96)."""
    small = cv2.resize(canonical_gray, (EDGE_SIZE, EDGE_SIZE), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    return cv2.Canny(small, 35, 100) > 0


def inner_edge_map(canonical_gray: np.ndarray, edges: np.ndarray, margin: int = 3) -> np.ndarray:
    """Only the edges inside the outline: hole rims, face boundaries, steps.
    These carry the design features that the outline alone cannot show."""
    obj = cv2.resize((canonical_gray < 250).astype(np.uint8), (EDGE_SIZE, EDGE_SIZE),
                     interpolation=cv2.INTER_NEAREST)
    # Fill through-holes so their rims count as inner features; only the outer outline is dropped.
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(obj)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return edges & (cv2.erode(filled, k) > 0)


def distance_maps(edges: np.ndarray) -> np.ndarray:
    """Truncated distance-to-nearest-edge for each map in (N,H,W)."""
    out = np.empty(edges.shape, np.float32)
    for i, e in enumerate(edges):
        if not e.any():
            out[i] = DT_CAP
            continue
        out[i] = np.minimum(cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3), DT_CAP)
    return out


def chamfer_matrix(photo_edges: np.ndarray, view_edges: np.ndarray) -> np.ndarray:
    """Symmetric mean chamfer distance (pixels) between every photo rotation (R) and view (V)."""
    photo_dt = distance_maps(photo_edges)
    view_dt = distance_maps(view_edges)
    R, V = len(photo_edges), len(view_edges)
    p2v = np.full((R, V), DT_CAP, np.float32)
    v2p = np.full((R, V), DT_CAP, np.float32)
    # Neither side shows any such edges: that agrees, but proves little -> neutral score.
    photo_empty = ~photo_edges.reshape(R, -1).any(1)
    view_empty = ~view_edges.reshape(V, -1).any(1)
    for r in range(R):
        ys, xs = np.nonzero(photo_edges[r])
        if len(ys):
            p2v[r] = view_dt[:, ys, xs].mean(axis=1)
    for v in range(V):
        ys, xs = np.nonzero(view_edges[v])
        if len(ys):
            v2p[:, v] = photo_dt[:, ys, xs].mean(axis=1)
    out = 0.5 * (p2v + v2p)
    out[np.outer(photo_empty, view_empty)] = 0.35 * DT_CAP
    return out


def edge_similarity(chamfer: np.ndarray) -> np.ndarray:
    """Map chamfer distance to a 0..1 similarity (1 = identical edges)."""
    return 1.0 - np.clip(chamfer / DT_CAP, 0, 1)
