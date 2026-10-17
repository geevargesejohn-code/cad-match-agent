"""Rule-based agent: the same tools and the same step log as the LLM agent, driven by
the numeric policy. Used when no LLM is configured, and as a safety net if the LLM fails."""
from __future__ import annotations

from agent import policy
from agent.session import REFERENCE, AgentSession, Verdict
from matching import fusion
from tools.match_tools import ready_model_ids
from tools.registry import execute


def run_rules(session: AgentSession) -> None:
    if session.verdict is not None:
        return
    plan = ("Plan: segment the photos, compare them with the reference model, check whether another "
            "library design fits better, render extra viewpoints if the evidence is close, then decide."
            if session.mode == REFERENCE else
            "Plan: segment the photos, search the whole library, render extra viewpoints for close "
            "candidates if needed, then decide.")
    session.add_log("plan", "Plan", plan)

    if not session.analyses:
        execute(session, "analyze_photos")
    if not session.usable_photos:
        session.verdict = Verdict(
            fusion.UNCERTAIN, session.reference_id, 0.5,
            "No object could be isolated in the photos. Photograph the part on a plain background that "
            "contrasts with it, filling about a third of the frame.", "rules", needs_more_photos=True)
        session.add_log("decision", "Need better photos", session.verdict.explanation)
        return

    if session.mode == REFERENCE:
        if session.reference_id not in session.scores:
            execute(session, "compare_with_model", {"model_id": session.reference_id})
        if not session.alternatives_checked and len(ready_model_ids()) > 1:
            execute(session, "check_alternatives", {"model_id": session.reference_id})
    elif not session.library_searched:
        execute(session, "search_library")

    d = policy.decide(session)
    todo = policy.needs_investigation(d, session)
    if d.verdict == fusion.UNCERTAIN and todo:
        session.add_log("thought", "Evidence is close",
                        "The score is between the thresholds, so render more viewpoints before deciding.")
        for model_id in todo[:2]:
            execute(session, "refine_views", {"model_id": model_id})
        d = policy.decide(session)

    if d.verdict == fusion.UNCERTAIN:
        execute(session, "request_more_photos", {
            "reason": "The photo is consistent with the design but not conclusive. " + d.explanation})
        session.verdict.decided_by = "rules"
        return

    execute(session, "submit_verdict", {
        "verdict": d.verdict, "model_id": d.model_id, "confidence": d.confidence, "explanation": d.explanation})
    session.verdict.decided_by = "rules"
