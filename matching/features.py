"""Precomputed features for CAD views and for photos.

For each CAD model we render many views once (at upload time) and store, per view:
  - a canonical 64x64 silhouette (for fast IoU against photos)
  - Hu moments and [aspect, solidity, holes] shape features
  - a visual embedding of the shaded render
For a photo we compute the same things, with the silhouette and embedding taken
at 24 in-plane rotations so the photo can be held at any angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from cad_processing import renderer
from matching import silhouette as sil
from matching.edges import EDGE_SIZE, edge_map, inner_edge_map
from matching.embeddings import canonical_crop, get_embedder
from matching.photo import PhotoAnalysis

FEATURE_VERSION = 4
ROTATIONS = np.arange(0, 360, 15, dtype=np.float64)
RENDER_SIZE = 224


@dataclass
class ViewFeatures:
    directions: np.ndarray   # (V,3)
    iou_masks: np.ndarray    # (V,64,64) bool
    hu: np.ndarray           # (V,6)
    shape: np.ndarray        # (V,3)
    emb: np.ndarray          # (V,D)
    edges: np.ndarray        # (V,96,96) bool
    inner_edges: np.ndarray  # (V,96,96) bool
    backend: str

    def __len__(self) -> int:
        return len(self.directions)

    def concat(self, other: "ViewFeatures") -> "ViewFeatures":
        return ViewFeatures(
            np.concatenate([self.directions, other.directions]),
            np.concatenate([self.iou_masks, other.iou_masks]),
            np.concatenate([self.hu, other.hu]),
            np.concatenate([self.shape, other.shape]),
            np.concatenate([self.emb, other.emb]),
            np.concatenate([self.edges, other.edges]),
            np.concatenate([self.inner_edges, other.inner_edges]),
            self.backend,
        )

    def save(self, path) -> None:
        np.savez_compressed(
            path, version=FEATURE_VERSION, backend=self.backend, directions=self.directions,
            iou_bits=np.packbits(self.iou_masks.reshape(len(self), -1), axis=1),
            hu=self.hu, shape=self.shape, emb=self.emb.astype(np.float16),
            edge_bits=np.packbits(self.edges.reshape(len(self), -1), axis=1),
            inner_bits=np.packbits(self.inner_edges.reshape(len(self), -1), axis=1),
        )

    @classmethod
    def load(cls, path) -> "ViewFeatures | None":
        try:
            d = np.load(path, allow_pickle=False)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if int(d["version"]) != FEATURE_VERSION:
            return None
        n = len(d["directions"])
        size = sil.IOU_SIZE
        masks = np.unpackbits(d["iou_bits"], axis=1)[:, : size * size].reshape(n, size, size).astype(bool)
        def unpack(key):
            return np.unpackbits(d[key], axis=1)[:, : EDGE_SIZE * EDGE_SIZE].reshape(n, EDGE_SIZE, EDGE_SIZE).astype(bool)
        return cls(d["directions"], masks, d["hu"], d["shape"], d["emb"].astype(np.float32),
                   unpack("edge_bits"), unpack("inner_bits"), str(d["backend"]))


def compute_view_features(mesh: trimesh.Trimesh, directions: np.ndarray) -> ViewFeatures:
    mesh = renderer.prepare_mesh(mesh)
    masks, hus, shapes, crops = [], [], [], []
    for d in directions:
        rgb, mask = renderer.render(mesh, d, size=RENDER_SIZE)
        masks.append(sil.canonical_mask(mask))
        hus.append(sil.hu_descriptor(sil.canonical_mask(mask, sil.FEATURE_SIZE)))
        shapes.append(sil.shape_features(mask))
        crops.append(canonical_crop(rgb, mask))
    embedder = get_embedder()
    edges = [edge_map(c) for c in crops]
    return ViewFeatures(
        np.asarray(directions, float), np.stack(masks), np.stack(hus), np.stack(shapes),
        embedder.embed(crops), np.stack(edges),
        np.stack([inner_edge_map(c, e) for c, e in zip(crops, edges)]), embedder.name,
    )


@dataclass
class PhotoFeatures:
    analysis: PhotoAnalysis
    rot_masks: np.ndarray    # (R,64,64)
    rot_emb: np.ndarray      # (R,D)
    rot_edges: np.ndarray    # (R,96,96)
    rot_inner: np.ndarray    # (R,96,96)
    hu: np.ndarray           # (6,)
    shape: np.ndarray        # (3,)
    backend: str


def compute_photo_features(analysis: PhotoAnalysis) -> PhotoFeatures:
    mask, rgb = analysis.mask, analysis.rgb
    embedder = get_embedder()
    crops = [canonical_crop(rgb, mask, rotation_deg=r) for r in ROTATIONS]
    edges = [edge_map(c) for c in crops]
    return PhotoFeatures(
        analysis=analysis,
        rot_masks=sil.rotated_canonical_masks(mask, ROTATIONS),
        rot_emb=embedder.embed(crops),
        rot_edges=np.stack(edges),
        rot_inner=np.stack([inner_edge_map(c, e) for c, e in zip(crops, edges)]),
        hu=sil.hu_descriptor(sil.canonical_mask(mask, sil.FEATURE_SIZE)),
        shape=sil.shape_features(mask),
        backend=embedder.name,
    )
