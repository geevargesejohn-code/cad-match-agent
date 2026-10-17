"""Photo loading and background removal.

Backends:
  rembg     - U^2-Net deep segmentation for the outline (robust to dark faces and
              busy backgrounds; downloads ~170 MB once), plus background-color
              analysis to restore through-holes that U^2-Net fills in
  classical - learns the background color from the photo border, then refines with
              GrabCut. No download. Works well on a plain, contrasting background.
  auto      - rembg if installed, otherwise classical
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from core.config import get_config

log = logging.getLogger(__name__)

WORK_SIZE = 640


@dataclass
class PhotoAnalysis:
    path: str
    rgb: np.ndarray                    # resized RGB image
    mask: np.ndarray                   # bool object mask
    backend: str
    area_fraction: float
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.area_fraction >= 0.01 and not any(w.startswith("No object") for w in self.warnings)

    def summary(self) -> dict:
        return {
            "photo": Path(self.path).name,
            "segmentation": self.backend,
            "object_area_percent": round(100 * self.area_fraction, 1),
            "usable": self.usable,
            "warnings": self.warnings,
        }


def load_image(path: str | Path, max_side: int = WORK_SIZE) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img).copy()


# ---------------- classical segmentation ----------------

def _border_pixels(lab: np.ndarray, width: int) -> np.ndarray:
    return np.concatenate([
        lab[:width].reshape(-1, 3), lab[-width:].reshape(-1, 3),
        lab[:, :width].reshape(-1, 3), lab[:, -width:].reshape(-1, 3),
    ]).astype(np.float32)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    obj = labels == largest
    # fill only tiny holes (noise); real through-holes are kept
    inv = (~obj).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    min_hole = max(8, int(0.0004 * obj.sum()))
    for i in range(1, n2):
        x, y, w, h, area = st2[i]
        touches_border = x == 0 or y == 0 or x + w == obj.shape[1] or y + h == obj.shape[0]
        if not touches_border and area < min_hole:
            obj[lab2 == i] = True
    return obj


def _background_model(rgb: np.ndarray):
    """Background colors learned from the photo border.

    Returns (per-pixel distance to the nearest background color, noise floor of
    that distance on the border itself)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    border = _border_pixels(lab, max(4, min(h, w) // 40))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(border, 4, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dist = np.min(np.linalg.norm(lab[:, :, None, :] - centers[None, None], axis=-1), axis=-1)
    border_dist = np.min(np.linalg.norm(border[:, None] - centers[None], axis=-1), axis=-1)
    return dist, float(np.percentile(border_dist, 99))


def recover_holes(rgb: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Punch through-holes back into a filled outline: regions inside the object
    whose color matches the background (you can see the table through them)."""
    if outer.sum() < 50:
        return outer
    dist, floor = _background_model(rgb)
    bg_like = (dist <= max(floor * 1.15, 4.0)) & outer
    bg_like = cv2.morphologyEx(bg_like.astype(np.uint8), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg_like, connectivity=8)
    area = outer.sum()
    inner = cv2.erode(outer.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    result = outer.copy()
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        region = labels == i
        # a real hole: not tiny, not huge, and surrounded by object (not a notch in the edge)
        if 0.0004 * area <= a <= 0.25 * area and (region & inner).sum() >= 0.6 * a:
            result[region] = False
    return result


def segment_classical(rgb: np.ndarray) -> np.ndarray:
    dist, floor = _background_model(rgb)
    h, w = dist.shape

    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(otsu / 255 * dist.max(), floor * 1.3, 6.0)
    rough = _clean_mask(dist > thresh)
    if rough.sum() < 50:
        return rough

    # GrabCut refinement seeded from the rough mask
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[rough] = cv2.GC_PR_FGD
    k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    gc[cv2.erode(rough.astype(np.uint8), k_er) > 0] = cv2.GC_FGD
    gc[cv2.dilate(rough.astype(np.uint8), k_er, iterations=3) == 0] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        refined = _clean_mask((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
        # trust GrabCut only if it didn't wildly change the object
        if 0.6 < refined.sum() / rough.sum() < 1.5:
            return refined
    except cv2.error:
        pass
    return rough


# ---------------- rembg segmentation ----------------

_REMBG_SESSION = None
_REMBG_LOCK = threading.Lock()


def _rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except Exception:
        return False


def segment_rembg(rgb: np.ndarray) -> np.ndarray:
    global _REMBG_SESSION
    from rembg import new_session, remove

    with _REMBG_LOCK:
        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session(get_config()["photo"].get("rembg_model", "u2net"))
    out = remove(Image.fromarray(rgb), session=_REMBG_SESSION, only_mask=True)
    outer = _clean_mask(np.asarray(out) > 127)
    # U^2-Net outlines objects very well but tends to fill through-holes; restore them.
    return recover_holes(rgb, outer)


# ---------------- entry point ----------------

def analyze_photo(path: str | Path) -> PhotoAnalysis:
    rgb = load_image(path)
    backend = get_config()["photo"].get("segmentation_backend", "auto")
    if backend == "auto":
        backend = "rembg" if _rembg_available() else "classical"

    try:
        mask = segment_rembg(rgb) if backend == "rembg" else segment_classical(rgb)
    except Exception as e:
        log.warning("%s segmentation failed (%s); falling back to classical", backend, e)
        backend, mask = "classical", segment_classical(rgb)

    # Guard against "objects" that are really just background (blank photo, empty table)
    if mask.sum() >= 50:
        dist, floor = _background_model(rgb)
        if float(np.median(dist[mask])) <= max(floor * 1.2, 3.0):
            mask = np.zeros_like(mask)

    area = float(mask.mean())
    warnings: list[str] = []
    if mask.sum() < 50:
        warnings.append("No object found. Use a plain background that contrasts with the object.")
    else:
        if area < 0.03:
            warnings.append("Object is very small in the frame; move closer for more detail.")
        if area > 0.85:
            warnings.append("Object fills almost the whole frame; background may not have been separated.")
        edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        if edge.mean() > 0.08:
            warnings.append("Object touches the photo edge and may be cut off.")
    return PhotoAnalysis(str(path), rgb, mask, backend, area, warnings)
