"""State of one test run: inputs, what the agent has learned so far, and the step log."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from matching.features import PhotoFeatures
from matching.photo import PhotoAnalysis
from matching.scorer import ModelScore

REFERENCE = "reference"
LIBRARY_SEARCH = "library_search"


@dataclass
class Verdict:
    verdict: str                 # MATCH | NOT A MATCH | UNCERTAIN
    model_id: str | None
    confidence: float
    explanation: str
    decided_by: str              # "llm:<provider>" or "rules"
    needs_more_photos: bool = False


@dataclass
class AgentSession:
    mode: str
    photo_paths: list[str]
    reference_id: str | None = None
    analyses: list[PhotoAnalysis] = field(default_factory=list)
    photo_features: list[PhotoFeatures] = field(default_factory=list)
    scores: dict[str, ModelScore] = field(default_factory=dict)      # full comparisons
    quick_scores: dict[str, float] = field(default_factory=dict)     # pre-filter scores
    refined: set[str] = field(default_factory=set)
    alternatives_checked: bool = False
    library_searched: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)
    verdict: Verdict | None = None
    _t0: float = field(default_factory=time.time)

    def add_log(self, kind: str, title: str, detail: str = "", data: Any = None) -> None:
        self.log.append({
            "step": len(self.log) + 1,
            "t": round(time.time() - self._t0, 2),
            "kind": kind,            # plan | tool | thought | decision | warning | error
            "title": title,
            "detail": detail,
            "data": data,
        })

    @property
    def usable_photos(self) -> list[PhotoFeatures]:
        return [p for p in self.photo_features if p.analysis.usable]

    def ranked(self) -> list[ModelScore]:
        return sorted(self.scores.values(), key=lambda s: s.score, reverse=True)
