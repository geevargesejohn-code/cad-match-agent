"""The numeric decision rules. The rule-based agent uses them directly; the LLM agent
sees their output as advice and makes the final call."""
from __future__ import annotations

from dataclasses import dataclass

from core.config import get_config
from matching import fusion
from agent.session import LIBRARY_SEARCH, REFERENCE, AgentSession


@dataclass
class Decision:
    verdict: str
    model_id: str | None
    confidence: float
    explanation: str
    # set when more evidence could help: which models to look at more closely
    investigate: list[str]

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "model_id": self.model_id,
                "confidence": round(self.confidence, 3), "explanation": self.explanation,
                "worth_investigating": self.investigate}


def _margin() -> float:
    return float(get_config()["matching"].get("alternative_margin", 0.05))


def decide(session: AgentSession) -> Decision:
    hi, lo = fusion.thresholds()
    margin = _margin()

    if session.mode == REFERENCE:
        ref = session.scores.get(session.reference_id)
        if ref is None:
            return Decision(fusion.UNCERTAIN, session.reference_id, 0.5, "Reference not compared yet.", [])
        s = ref.score
        others = [m for m in session.ranked() if m.item_id != ref.item_id]
        alt = others[0] if others else None

        if alt and alt.score >= s + margin and alt.score > lo:
            conf = min(0.95, 0.65 + (alt.score - s))
            return Decision(
                fusion.NOT_A_MATCH, ref.item_id, conf,
                f"The object fits '{alt.name}' better than the reference '{ref.name}' "
                f"({alt.score:.2f} vs {s:.2f}), so it is a different design. " + ref.explanation(), [])
        if s >= hi:
            close = alt is not None and alt.score > s - margin and alt.score >= hi
            note = (f" Note: '{alt.name}' is also similar ({alt.score:.2f}); the reference is still the closest."
                    if close else "")
            _, conf = fusion.decide(s)
            return Decision(fusion.MATCH, ref.item_id, conf, ref.explanation() + note, [])
        if s <= lo:
            _, conf = fusion.decide(s)
            return Decision(fusion.NOT_A_MATCH, ref.item_id, conf, ref.explanation(), [])
        todo = [ref.item_id] + ([alt.item_id] if alt and alt.score > lo else [])
        return Decision(fusion.UNCERTAIN, ref.item_id, 0.5, ref.explanation(), todo)

    # library search
    ranked = session.ranked()
    if not ranked:
        return Decision(fusion.NOT_A_MATCH, None, 0.6, "The library has no models to compare.", [])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if best.score >= hi:
        if second and second.score >= hi and best.score - second.score < margin:
            return Decision(
                fusion.UNCERTAIN, best.item_id, 0.5,
                f"Two library models fit almost equally well: '{best.name}' ({best.score:.2f}) and "
                f"'{second.name}' ({second.score:.2f}).", [best.item_id, second.item_id])
        _, conf = fusion.decide(best.score)
        return Decision(fusion.MATCH, best.item_id, conf, best.explanation(), [])
    if best.score <= lo:
        _, conf = fusion.decide(best.score)
        return Decision(fusion.NOT_A_MATCH, best.item_id, conf,
                        f"No library model matches. Closest was '{best.name}': " + best.explanation(), [])
    return Decision(fusion.UNCERTAIN, best.item_id, 0.5, best.explanation(),
                    [best.item_id] + ([second.item_id] if second and second.score > lo else []))


def needs_investigation(d: Decision, session: AgentSession) -> list[str]:
    """Models from d.investigate that have not been refined yet."""
    return [m for m in d.investigate if m not in session.refined]


__all__ = ["Decision", "decide", "needs_investigation", "LIBRARY_SEARCH", "REFERENCE"]
