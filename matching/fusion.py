"""Turns raw signals into one 0-1 score and a verdict. All numbers come from config.yaml."""
from __future__ import annotations

import numpy as np

from core.config import get_config

MATCH = "MATCH"
NOT_A_MATCH = "NOT A MATCH"
UNCERTAIN = "UNCERTAIN"


def _cfg() -> dict:
    return get_config()["matching"]


def normalize(name: str, raw: np.ndarray, backend: str | None = None) -> np.ndarray:
    cfg = _cfg()
    if name == "embedding":
        lo, hi = cfg["embedding_ranges"][backend]
    else:
        lo, hi = cfg["signal_ranges"][name]
    return np.clip((np.asarray(raw, float) - lo) / (hi - lo), 0.0, 1.0)


def combine(norm: dict[str, np.ndarray], skip: tuple[str, ...] = ()) -> np.ndarray:
    w = {k: v for k, v in _cfg()["weights"].items() if k not in skip and v > 0}
    total = sum(w.values())
    return sum(w[k] * norm[k] for k in w) / total


def thresholds() -> tuple[float, float]:
    cfg = _cfg()
    return float(cfg["match_threshold"]), float(cfg["no_match_threshold"])


def decide(score: float) -> tuple[str, float]:
    """Verdict and a confidence in [0.5, 1] from a fused score."""
    hi, lo = thresholds()
    if score >= hi:
        return MATCH, round(0.6 + 0.4 * min(1.0, (score - hi) / max(1e-6, 1 - hi) * 2), 3)
    if score <= lo:
        return NOT_A_MATCH, round(0.6 + 0.4 * min(1.0, (lo - score) / max(1e-6, lo) * 3), 3)
    return UNCERTAIN, 0.5
