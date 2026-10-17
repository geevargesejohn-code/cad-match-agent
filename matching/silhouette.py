"""Scale-invariant silhouette tools.

Every silhouette (from a photo or a CAD render) is put into a *canonical frame*:
centered on its centroid and scaled so its RMS radius is fixed. After that, two
silhouettes of the same design line up regardless of how big the object was or
how far the camera stood. Rotation in the image plane is searched explicitly.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

IOU_SIZE = 64          # resolution for the fast IoU comparison
FEATURE_SIZE = 192     # resolution for hole/solidity measurements
RMS_FRACTION = 0.21    # RMS radius as a fraction of the canvas size


def canonical_affine(mask: np.ndarray, out_size: int, rotation_deg: float = 0.0) -> np.ndarray | None:
    """2x3 affine that maps `mask` into the canonical frame (optionally rotated)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    cx, cy = xs.mean(), ys.mean()
    rms = math.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean())
    if rms <= 0:
        return None
    s = (RMS_FRACTION * out_size) / rms
    a = math.radians(rotation_deg)
    cos, sin = math.cos(a) * s, math.sin(a) * s
    # rotate+scale about the centroid, then move centroid to the canvas center
    return np.array([
        [cos, -sin, out_size / 2 - (cos * cx - sin * cy)],
        [sin, cos, out_size / 2 - (sin * cx + cos * cy)],
    ], dtype=np.float64)


def warp_mask(mask: np.ndarray, M: np.ndarray, out_size: int) -> np.ndarray:
    warped = cv2.warpAffine(mask.astype(np.float32), M, (out_size, out_size),
                            flags=cv2.INTER_LINEAR, borderValue=0)
    return warped > 0.5


def canonical_mask(mask: np.ndarray, out_size: int = IOU_SIZE, rotation_deg: float = 0.0) -> np.ndarray:
    M = canonical_affine(mask, out_size, rotation_deg)
    if M is None:
        return np.zeros((out_size, out_size), bool)
    return warp_mask(mask, M, out_size)


def rotated_canonical_masks(mask: np.ndarray, rotations: np.ndarray, out_size: int = IOU_SIZE) -> np.ndarray:
    return np.stack([canonical_mask(mask, out_size, r) for r in rotations])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every mask in a (N,H,W) and every mask in b (M,H,W) -> (N,M)."""
    A = a.reshape(len(a), -1).astype(np.float32)
    B = b.reshape(len(b), -1).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def hu_descriptor(mask: np.ndarray) -> np.ndarray:
    """Log-scaled Hu moments 1-6 (the 7th only flips sign under mirroring, so it is skipped)."""
    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8), binaryImage=True)).ravel()[:6]
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def hu_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between Hu descriptors; broadcasts over leading axes. Weighted toward low orders."""
    w = np.array([1.0, 0.7, 0.4, 0.4, 0.15, 0.15])
    return (np.abs(np.clip(a, 0, 30) - np.clip(b, 0, 30)) * w).sum(-1) / w.sum()


def shape_features(mask: np.ndarray) -> np.ndarray:
    """[aspect, solidity, holes] of a silhouette, all independent of scale and rotation.

    aspect   = minor/major principal axis ratio (0..1]
    solidity = area / convex hull area (catches L-shapes, notches, slots)
    holes    = number of significant through-holes visible in this view
    """
    m = canonical_mask(mask, FEATURE_SIZE).astype(np.uint8)
    if m.sum() < 20:
        return np.array([0.0, 0.0, 0.0])

    ys, xs = np.nonzero(m)
    cov = np.cov(np.stack([xs, ys]).astype(np.float64))
    ev = np.sort(np.linalg.eigvalsh(cov))
    aspect = float(math.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))

    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
    main = max(outer, key=lambda i: cv2.contourArea(contours[i]))
    area = float(m.sum())
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack([contours[i] for i in outer])))
    solidity = min(1.0, area / hull_area) if hull_area > 0 else 0.0

    min_hole = 0.0015 * area
    holes = sum(
        1 for i, h in enumerate(hierarchy[0])
        if h[3] == main and cv2.contourArea(contours[i]) > min_hole
    )
    return np.array([aspect, solidity, float(holes)])
