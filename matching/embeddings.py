"""Visual embeddings for overall appearance similarity.

Backends:
  dinov2 - facebook/dinov2-small through Hugging Face transformers (best quality;
           needs `pip install -r requirements-ml.txt`, downloads ~90 MB once)
  hog    - Histogram of Oriented Gradients via OpenCV (no download, always available)
  auto   - dinov2 if it can be loaded, otherwise hog
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from core.config import get_config
from matching.silhouette import canonical_affine

log = logging.getLogger(__name__)

CROP_SIZE = 224
CROP_RMS_FRACTION = 0.21


def canonical_crop(rgb: np.ndarray, mask: np.ndarray, size: int = CROP_SIZE,
                   rotation_deg: float = 0.0, gray: bool = True) -> np.ndarray:
    """Object cut out on white, in the same canonical frame as the silhouettes.

    Grayscale + per-object contrast normalization removes most of the gap between
    a real photo (color, texture, lighting) and a flat-shaded CAD render.
    """
    M = canonical_affine(mask, size, rotation_deg)
    if M is None:
        return np.full((size, size) if gray else (size, size, 3), 255, np.uint8)
    img = rgb.copy()
    img[~mask] = 255
    warped = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    wmask = cv2.warpAffine(mask.astype(np.float32), M, (size, size), borderValue=0) > 0.5
    if not gray:
        return warped
    g = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if wmask.sum() > 20:
        vals = g[wmask]
        g[wmask] = np.clip((vals - vals.mean()) / (vals.std() + 1e-6) * 38 + 140, 10, 245)
    g[~wmask] = 255
    return g.astype(np.uint8)


def hog_descriptor(gray: np.ndarray, size: int = 128, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Histogram of Oriented Gradients (Dalal-Triggs style), implemented in NumPy so it
    does not depend on any particular OpenCV build."""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0  # unsigned orientation

    n = size // cell
    pos = ang / (180.0 / bins)
    lo = np.floor(pos).astype(int) % bins
    hi = (lo + 1) % bins
    w_hi = pos - np.floor(pos)
    cy, cx = np.divmod(np.arange(size * size), size)
    cell_idx = ((cy // cell) * n + (cx // cell))
    m = mag.ravel()
    hist = (np.bincount(cell_idx * bins + lo.ravel(), m * (1 - w_hi.ravel()), n * n * bins)
            + np.bincount(cell_idx * bins + hi.ravel(), m * w_hi.ravel(), n * n * bins))
    hist = hist.reshape(n, n, bins).astype(np.float32)

    blocks = []
    for y in range(n - 1):
        for x in range(n - 1):
            b = hist[y:y + 2, x:x + 2].ravel()
            b = b / (np.linalg.norm(b) + 1e-6)
            blocks.append(np.minimum(b, 0.2))
    f = np.concatenate(blocks)
    return f / (np.linalg.norm(f) + 1e-8)


class HOGEmbedder:
    name = "hog"

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        return np.stack([hog_descriptor(c) for c in crops]).astype(np.float32)


class DinoV2Embedder:
    name = "dinov2"

    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, crops: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch_size):
            batch = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) if c.ndim == 2 else c for c in crops[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            # CLS token + mean of patch tokens: robust global descriptor
            feat = self._torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], dim=1)
            feat = self._torch.nn.functional.normalize(feat, dim=1)
            out.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(out)


_EMBEDDER = None
_LOCK = threading.Lock()


def get_embedder():
    """Return the configured embedder, created once per process."""
    global _EMBEDDER
    with _LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        cfg = get_config()["matching"]
        backend = cfg.get("embedding_backend", "auto")
        if backend in ("auto", "dinov2"):
            try:
                _EMBEDDER = DinoV2Embedder(cfg.get("dinov2_model", "facebook/dinov2-small"))
                log.info("Using DINOv2 embeddings")
                return _EMBEDDER
            except Exception as e:
                if backend == "dinov2":
                    raise RuntimeError(f"DINOv2 requested but could not be loaded: {e}") from e
                log.warning("DINOv2 unavailable (%s); using HOG embeddings", e.__class__.__name__)
        _EMBEDDER = HOGEmbedder()
        return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER
    with _LOCK:
        _EMBEDDER = None
