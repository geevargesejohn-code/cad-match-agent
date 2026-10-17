"""Agent tools: compare photos with CAD models, search the library, finish the test."""
from __future__ import annotations

from agent import policy
from agent.session import REFERENCE, AgentSession, Verdict
from core.config import get_config
from matching import fusion, indexer
from matching.scorer import score_model
from storage import library
from tools.registry import ToolResult, tool

PREFILTER_KEEP = 20


def ready_model_ids() -> list[str]:
    return [i["id"] for i in library.list_items() if i["status"] != "error"]


def _require_photos(session: AgentSession) -> None:
    if not session.usable_photos:
        raise RuntimeError("No usable photos. Run analyze_photos first, or ask for better photos.")


def full_score(session: AgentSession, model_id: str):
    _require_photos(session)
    if not library.get_item(model_id):
        raise KeyError(f"No library model with id '{model_id}'")
    views = indexer.load_features(model_id)
    ms = score_model(session.usable_photos, model_id, views, refined=indexer.is_refined(model_id))
    session.scores[model_id] = ms
    return ms


def _rank(session: AgentSession, ids: list[str], top_k: int) -> list:
    """Quick pre-filter over all ids, then the full comparison on the most promising ones."""
    _require_photos(session)
    todo = [i for i in ids if i not in session.scores and i not in session.quick_scores]
    for i in todo:
        quick = score_model(session.usable_photos, i, indexer.load_features(i), quick=True)
        session.quick_scores[i] = quick.score
    pool = sorted(ids, key=lambda i: session.scores[i].score if i in session.scores else session.quick_scores[i],
                  reverse=True)
    for i in pool[: max(PREFILTER_KEEP, top_k)]:
        if i not in session.scores:
            full_score(session, i)
    return sorted((session.scores[i] for i in ids if i in session.scores), key=lambda m: m.score, reverse=True)


@tool(
    "list_library", "List CAD library",
    "List the CAD models available in the library (id, name, file format, 3D proportions).",
)
def list_library(session: AgentSession) -> ToolResult:
    items = [
        {"id": i["id"], "name": i["name"], "format": i["format"], "proportions_3d": i["proportions"]}
        for i in library.list_items() if i["status"] != "error"
    ]
    return ToolResult(f"{len(items)} model(s) in the library", {"models": items})


@tool(
    "compare_with_model", "Compare with model",
    "Compare the photos with one CAD model across all its rendered viewpoints. Returns the fused "
    "score (0-1), each signal at the best pose, and the current numeric decision.",
    {"type": "object", "properties": {"model_id": {"type": "string"}}, "required": ["model_id"]},
)
def compare_with_model(session: AgentSession, model_id: str) -> ToolResult:
    ms = full_score(session, model_id)
    d = policy.decide(session)
    return ToolResult(
        f"'{ms.name}' scored {ms.score:.2f} ({ms.verdict.lower()} on its own)",
        {"comparison": ms.summary(), "numeric_decision": d.as_dict()},
    )


@tool(
    "search_library", "Search library",
    "Compare the photos with every model in the library and return the best candidates.",
    {"type": "object", "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
     "required": []},
)
def search_library(session: AgentSession, top_k: int | None = None) -> ToolResult:
    top_k = top_k or int(get_config()["matching"]["top_k_candidates"])
    ranked = _rank(session, ready_model_ids(), top_k)
    session.library_searched = True
    d = policy.decide(session)
    top = ranked[:top_k]
    line = ", ".join(f"{m.name} {m.score:.2f}" for m in top) or "library is empty"
    return ToolResult(f"Top candidates: {line}",
                      {"candidates": [m.summary() for m in top], "numeric_decision": d.as_dict()})


@tool(
    "check_alternatives", "Check other library models",
    "Reference test only: compare the photos with the OTHER library models too. If another design "
    "fits clearly better than the reference, the object is probably that design instead.",
    {"type": "object", "properties": {"model_id": {"type": "string", "description": "the reference model id"}},
     "required": ["model_id"]},
)
def check_alternatives(session: AgentSession, model_id: str) -> ToolResult:
    others = [i for i in ready_model_ids() if i != model_id]
    ranked = _rank(session, others, 3)
    session.alternatives_checked = True
    ref = session.scores.get(model_id) or full_score(session, model_id)
    d = policy.decide(session)
    if not ranked:
        return ToolResult("No other models to compare against",
                          {"alternatives": [], "numeric_decision": d.as_dict()})
    best = ranked[0]
    summary = (f"Best alternative '{best.name}' {best.score:.2f} vs reference {ref.score:.2f}")
    return ToolResult(summary, {"reference_score": round(ref.score, 3),
                                "alternatives": [m.summary() for m in ranked[:3]],
                                "numeric_decision": d.as_dict()})


@tool(
    "refine_views", "Render more viewpoints",
    "Render a denser set of viewpoints for one model and re-compare. Use when a score is close to "
    "a threshold or two models are close: a better-aligned view can settle it.",
    {"type": "object", "properties": {"model_id": {"type": "string"}}, "required": ["model_id"]},
)
def refine_views(session: AgentSession, model_id: str) -> ToolResult:
    _require_photos(session)
    before = session.scores[model_id].score if model_id in session.scores else None
    indexer.refine_model(model_id)
    session.refined.add(model_id)
    ms = full_score(session, model_id)
    d = policy.decide(session)
    change = f" (was {before:.2f})" if before is not None else ""
    return ToolResult(f"'{ms.name}' re-scored with {ms.num_views} views: {ms.score:.2f}{change}",
                      {"comparison": ms.summary(), "numeric_decision": d.as_dict()})


@tool(
    "view_comparison", "Look at side-by-side image",
    "Get an image: the photo, the photo aligned to the best pose, the CAD model rendered from the "
    "matched viewpoint, and a silhouette overlay. Use it to check features (holes, slots, steps) visually.",
    {"type": "object", "properties": {"model_id": {"type": "string"}}, "required": ["model_id"]},
)
def view_comparison(session: AgentSession, model_id: str) -> ToolResult:
    from matching.visuals import comparison_image

    ms = session.scores.get(model_id) or full_score(session, model_id)
    best = ms.best_photo
    analysis = next(p.analysis for p in session.usable_photos if p.analysis.path == best.photo)
    img = comparison_image(analysis, model_id, ms.name, best)
    return ToolResult(f"Side-by-side image for '{ms.name}'",
                      {"model_id": model_id, "model_name": ms.name, "score": round(ms.score, 3),
                       "image": "attached: [photo | photo aligned | CAD view | overlay red=photo only, "
                                "blue=CAD only, gray=both]"},
                      images=[img])


@tool(
    "request_more_photos", "Ask for another photo",
    "End the test without a final verdict and ask the user for more photos from other angles. "
    "Use only when the evidence is genuinely ambiguous after refining.",
    {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
)
def request_more_photos(session: AgentSession, reason: str) -> ToolResult:
    d = policy.decide(session)
    session.verdict = Verdict(fusion.UNCERTAIN, d.model_id, 0.5,
                              reason.strip() + " Please upload another photo from a different angle.",
                              decided_by="", needs_more_photos=True)
    return ToolResult("Asked for another photo", {"ok": True}, terminal=True)


@tool(
    "submit_verdict", "Submit verdict",
    "Finish the test with the final verdict. model_id is the reference model (reference test) or "
    "the best-matching model (library search; may be null if nothing matches).",
    {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": [fusion.MATCH, fusion.NOT_A_MATCH]},
        "model_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string", "description": "2-4 plain-English sentences for the user"},
    }, "required": ["verdict", "confidence", "explanation"]},
)
def submit_verdict(session: AgentSession, verdict: str, explanation: str, confidence: float,
                   model_id: str | None = None) -> ToolResult:
    verdict = verdict.strip().upper()
    if verdict not in (fusion.MATCH, fusion.NOT_A_MATCH):
        raise ValueError(f"verdict must be '{fusion.MATCH}' or '{fusion.NOT_A_MATCH}'")
    if session.mode == REFERENCE:
        model_id = session.reference_id
    if model_id is not None and not library.get_item(model_id):
        raise KeyError(f"Unknown model_id '{model_id}'")
    if verdict == fusion.MATCH and model_id is None:
        raise ValueError("A MATCH verdict needs a model_id")
    session.verdict = Verdict(verdict, model_id, float(max(0.0, min(1.0, confidence))), explanation.strip(), "")
    return ToolResult(f"Verdict: {verdict}", {"ok": True}, terminal=True)
