"""Side-by-side comparison images: the photo next to the best-matching CAD view."""
from __future__ import annotations

import cv2
import numpy as np

from matching.embeddings import canonical_crop
from matching.indexer import render_view
from matching.photo import PhotoAnalysis
from matching.scorer import PhotoScore
from matching.silhouette import canonical_mask

PANEL = 300
COLORS = {"MATCH": (34, 139, 34), "NOT A MATCH": (200, 40, 40), "UNCERTAIN": (210, 140, 0)}


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, img.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _fit(img: np.ndarray, size: int = PANEL) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def comparison_image(analysis: PhotoAnalysis, item_id: str, model_name: str, ps: PhotoScore,
                     verdict: str | None = None, score: float | None = None) -> np.ndarray:
    # 1) the photo with the detected outline
    photo = analysis.rgb.copy()
    contours, _ = cv2.findContours(analysis.mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(photo, contours, -1, (255, 60, 60), 2)

    # 2) the photo aligned into the matched pose, 3) the CAD model from the matched viewpoint
    aligned = canonical_crop(analysis.rgb, analysis.mask, PANEL, rotation_deg=ps.rotation, gray=False)
    cad_rgb, cad_mask = render_view(item_id, np.array(ps.direction), size=PANEL)
    cad = canonical_crop(cad_rgb, cad_mask, PANEL, gray=False)

    # 4) silhouette overlay: red = photo only, blue = CAD only, gray = both
    pm = canonical_mask(analysis.mask, PANEL, ps.rotation)
    cm = canonical_mask(cad_mask, PANEL)
    overlay = np.full((PANEL, PANEL, 3), 255, np.uint8)
    overlay[pm & ~cm] = (230, 70, 70)
    overlay[cm & ~pm] = (70, 120, 230)
    overlay[pm & cm] = (150, 150, 150)

    panels = [
        _label(_fit(photo), "Your photo"),
        _label(aligned, "Photo, aligned"),
        _label(cad, f"CAD: {model_name[:26]}"),
        _label(overlay, f"Overlap IoU {ps.raw['iou']:.2f}"),
    ]
    gap = np.full((PANEL + 34, 8, 3), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, gap, p])

    if verdict:
        color = COLORS.get(verdict, (90, 90, 90))
        banner = np.full((46, row.shape[1], 3), color, np.uint8)
        text = verdict + (f"   score {score:.2f}" if score is not None else "")
        cv2.putText(banner, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        row = np.vstack([banner, row])
    return row
