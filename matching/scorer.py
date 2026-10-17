"""Scores photos against CAD models.

For every (photo rotation r, CAD view v) pair we compute four signals:
  embedding   - visual similarity of the shaded object
  iou         - silhouette overlap after scale/position normalization
  hu          - Hu-moment shape similarity (rotation/scale invariant)
  proportions - aspect, solidity and visible hole count agreement
They are normalized and combined; the best (r, v) pose is the model's score for
that photo. With several photos, each is scored independently and averaged:
every photo must look like the model for the result to be a match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from matching import fusion
from matching.features import ROTATIONS, PhotoFeatures, ViewFeatures
from matching.edges import chamfer_matrix, edge_similarity
from matching.proportions import describe, proportion_similarity
from matching.silhouette import hu_distance, iou_matrix
from storage import library


@dataclass
class PhotoScore:
    photo: str
    score: float
    view_index: int
    rotation: float
    direction: list[float]
    raw: dict[str, float]
    normalized: dict[str, float]
    feature_note: str


@dataclass
class ModelScore:
    item_id: str
    name: str
    score: float
    verdict: str
    confidence: float
    per_photo: list[PhotoScore] = field(default_factory=list)
    num_views: int = 0
    refined: bool = False

    @property
    def best_photo(self) -> PhotoScore:
        return max(self.per_photo, key=lambda p: p.score)

    def summary(self) -> dict:
        return {
            "model_id": self.item_id,
            "model_name": self.name,
            "score": round(self.score, 3),
            "numeric_verdict": self.verdict,
            "views_compared": self.num_views,
            "refined": self.refined,
            "per_photo": [
                {
                    "photo": p.photo,
                    "score": round(p.score, 3),
                    "signals": {k: round(v, 3) for k, v in p.normalized.items()},
                    "raw": {k: round(v, 3) for k, v in p.raw.items()},
                    "features": p.feature_note,
                }
                for p in self.per_photo
            ],
        }

    def explanation(self) -> str:
        """Plain-English reasons, strongest evidence first."""
        p = self.best_photo
        n = p.normalized

        def level(x: float) -> str:
            return "strong" if x >= 0.75 else "partial" if x >= 0.4 else "weak"

        parts = [f"outline {level(n['iou'])} (overlap {p.raw['iou']:.2f})"]
        if "inner_edges" in n:
            parts.append(f"internal features such as holes and steps {level(n['inner_edges'])}")
        parts.append(f"appearance {level(n['embedding'])}")
        parts.append(f"proportions {level(n['proportions'])}")
        holes_photo, holes_model = _holes(p.feature_note)
        hole_txt = ""
        if holes_photo is not None and holes_photo != holes_model and self.verdict != fusion.MATCH:
            hole_txt = f" The photo shows {holes_photo} through-hole(s) where this view of the model shows {holes_model}."
        hi, lo = fusion.thresholds()
        photos = f" over {len(self.per_photo)} photos" if len(self.per_photo) > 1 else ""
        return (f"Compared with '{self.name}' at the best-matching viewpoint: " + ", ".join(parts) + "."
                + hole_txt + f" Overall score {self.score:.2f}{photos} (match at {hi:.2f} or above, "
                f"no match at {lo:.2f} or below).")


def _holes(note: str) -> tuple[int | None, int | None]:
    import re
    m = re.search(r"visible holes (\d+) vs (\d+)", note)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


EDGE_SIGNALS = ("edges", "inner_edges")


def score_photo(photo: PhotoFeatures, views: ViewFeatures, quick: bool = False) -> PhotoScore:
    """Best pose and fused score for one photo against one model's views.

    quick=True skips the edge-chamfer signals (the slowest part); used to pre-filter
    large libraries before the full comparison."""
    backend = views.backend
    emb_raw = photo.rot_emb @ views.emb.T                        # (R,V)
    iou_raw = iou_matrix(photo.rot_masks, views.iou_masks)       # (R,V)
    hu_raw = np.exp(-hu_distance(photo.hu[None], views.hu))      # (V,)
    prop_raw = proportion_similarity(photo.shape, views.shape)   # (V,)

    norm = {
        "embedding": fusion.normalize("embedding", emb_raw, backend),
        "iou": fusion.normalize("iou", iou_raw),
        "hu": np.broadcast_to(fusion.normalize("hu", hu_raw)[None], emb_raw.shape),
        "proportions": np.broadcast_to(fusion.normalize("proportions", prop_raw)[None], emb_raw.shape),
    }
    raw_maps = {"embedding": emb_raw, "iou": iou_raw}
    if not quick:
        raw_maps["edges"] = edge_similarity(chamfer_matrix(photo.rot_edges, views.edges))
        raw_maps["inner_edges"] = edge_similarity(chamfer_matrix(photo.rot_inner, views.inner_edges))
        for k in EDGE_SIGNALS:
            norm[k] = fusion.normalize(k, raw_maps[k])
    total = fusion.combine(norm, skip=EDGE_SIGNALS if quick else ())
    r, v = np.unravel_index(int(np.argmax(total)), total.shape)

    raw = {k: float(m[r, v]) for k, m in raw_maps.items()}
    raw["hu"] = float(hu_raw[v])
    raw["proportions"] = float(prop_raw[v])
    return PhotoScore(
        photo=photo.analysis.path,
        score=float(total[r, v]),
        view_index=int(v),
        rotation=float(ROTATIONS[r]),
        direction=views.directions[v].tolist(),
        raw=raw,
        normalized={k: float(a[r, v]) for k, a in norm.items()},
        feature_note=describe(photo.shape, views.shape[v]),
    )


def score_model(photos: list[PhotoFeatures], item_id: str, views: ViewFeatures, refined: bool = False,
                quick: bool = False) -> ModelScore:
    item = library.get_item(item_id) or {"name": item_id}
    per_photo = [score_photo(p, views, quick=quick) for p in photos]
    score = float(np.mean([p.score for p in per_photo]))
    verdict, conf = fusion.decide(score)
    return ModelScore(item_id, item["name"], score, verdict, conf, per_photo, len(views), refined)
