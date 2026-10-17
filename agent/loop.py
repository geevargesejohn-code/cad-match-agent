"""The agent loop and the end-to-end `run_test` entry point used by the UI, API and tests.

LLM mode: the model plans and calls tools in a loop (observe -> think -> act) until it
submits a verdict. Rule mode: the same tools are driven by the numeric policy. If the
LLM is unavailable or misbehaves, the rules finish the job, so a test always completes.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2

from agent import policy
from agent.fallback import run_rules
from agent.llm import ProviderError, ProviderUnavailable, get_provider, png_b64
from agent.prompts import system_prompt, task_message
from agent.session import LIBRARY_SEARCH, REFERENCE, AgentSession
from core.config import get_config, path
from matching import fusion, indexer
from storage import db, library
from tools import registry

log = logging.getLogger(__name__)

AGENT_TOOLS_REFERENCE = ["analyze_photos", "compare_with_model", "check_alternatives", "refine_views",
                         "view_comparison", "list_library", "request_more_photos", "submit_verdict"]
AGENT_TOOLS_SEARCH = ["analyze_photos", "search_library", "compare_with_model", "refine_views",
                      "view_comparison", "list_library", "request_more_photos", "submit_verdict"]


def run_llm(session: AgentSession, provider) -> None:
    names = AGENT_TOOLS_REFERENCE if session.mode == REFERENCE else AGENT_TOOLS_SEARCH
    tools = registry.schemas(names)
    ref = library.get_item(session.reference_id) if session.reference_id else None
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": task_message(session.mode, ref, len(session.photo_paths))}]}
    ]
    max_steps = int(get_config()["agent"]["max_steps"])
    nudged = False
    session.add_log("plan", f"Agent started ({provider.name}, model {getattr(provider, 'model', '?')})",
                    "The language model plans the steps and calls the tools below.")

    for _ in range(max_steps):
        resp = provider.chat(system_prompt(), messages, tools)
        if resp.text.strip():
            session.add_log("thought", "Agent", resp.text.strip())
        messages.append({"role": "assistant", "text": resp.text,
                         "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                                        for c in resp.tool_calls]})
        if not resp.tool_calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": [{"type": "text", "text":
                             "Please continue with tool calls, and finish with submit_verdict or request_more_photos."}]})
            continue

        for call in resp.tool_calls:
            if call.name not in names:
                result = registry.ToolResult(f"Tool {call.name} not allowed", {"error": "tool not available"})
            else:
                result = registry.execute(session, call.name, call.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": result.as_text(), "images": [png_b64(i) for i in result.images]})
            if session.verdict is not None:
                session.verdict.decided_by = f"llm:{provider.name}"
                return
    session.add_log("warning", "Agent did not finish",
                    f"No verdict after {max_steps} steps; the rule-based policy completes the test.")


def run_agent(session: AgentSession, provider_name: str | None = None) -> None:
    registry.load_all()
    try:
        provider = get_provider(provider_name)
    except ProviderUnavailable as e:
        session.add_log("warning", "LLM not available", f"{e}. Using the rule-based agent.")
        provider = None

    if provider is not None:
        try:
            run_llm(session, provider)
        except ProviderError as e:
            session.add_log("error", "LLM call failed", f"{e}. Continuing with the rule-based agent.")
    if session.verdict is None:
        run_rules(session)


def _best_model_id(session: AgentSession) -> str | None:
    v = session.verdict
    if v and v.model_id and v.model_id in session.scores:
        return v.model_id
    if session.mode == REFERENCE and session.reference_id in session.scores:
        return session.reference_id
    ranked = session.ranked()
    return ranked[0].item_id if ranked else None


def run_test(photo_paths: list[str | Path], mode: str = LIBRARY_SEARCH, reference_id: str | None = None,
             provider: str | None = None, photo_names: list[str] | None = None,
             on_step: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Run a full test and save it to history. Returns the result dictionary."""
    indexer.install()
    if mode not in (REFERENCE, LIBRARY_SEARCH):
        raise ValueError(f"mode must be '{REFERENCE}' or '{LIBRARY_SEARCH}'")
    if not photo_paths:
        raise ValueError("Upload at least one photo of the object")
    if mode == REFERENCE:
        if not reference_id or not library.get_item(reference_id):
            raise ValueError("Pick a reference CAD model from the library")
    elif not library.list_items():
        raise ValueError("The CAD library is empty. Add CAD files first.")

    test_id = uuid.uuid4().hex[:12]
    up_dir = path("uploads_dir") / test_id
    up_dir.mkdir(parents=True, exist_ok=True)
    stored = []
    for i, p in enumerate(photo_paths):
        src = Path(p)
        name = Path(photo_names[i]).name if photo_names else src.name
        dest = up_dir / f"{i + 1:02d}_{name}"
        shutil.copy2(src, dest)
        stored.append(str(dest))

    session = AgentSession(mode=mode, photo_paths=stored, reference_id=reference_id)
    if on_step:  # stream steps to a UI as they happen
        original = session.add_log

        def add_and_emit(*a, **k):
            original(*a, **k)
            on_step(session.log[-1])
        session.add_log = add_and_emit  # type: ignore[method-assign]

    run_agent(session, provider)
    return _finish(session, test_id)


def _finish(session: AgentSession, test_id: str) -> dict[str, Any]:
    v = session.verdict
    numeric = policy.decide(session) if session.scores else None
    model_id = _best_model_id(session)
    best = session.scores.get(model_id) if model_id else None

    image_path = None
    if best is not None:
        from matching.visuals import comparison_image
        bp = best.best_photo
        analysis = next(p.analysis for p in session.usable_photos if p.analysis.path == bp.photo)
        img = comparison_image(analysis, best.item_id, best.name, bp, v.verdict, best.score)
        res_dir = path("results_dir")
        res_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(res_dir / f"{test_id}.png")
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ref = library.get_item(session.reference_id) if session.reference_id else None
    result = {
        "test_id": test_id,
        "created_at": db.now_iso(),
        "mode": session.mode,
        "reference": {"id": ref["id"], "name": ref["name"]} if ref else None,
        "verdict": v.verdict,
        "confidence": round(v.confidence, 3),
        "needs_more_photos": v.needs_more_photos,
        "explanation": v.explanation,
        "decided_by": v.decided_by,
        "numeric_verdict": numeric.verdict if numeric else None,
        "best_match": {"id": best.item_id, "name": best.name, "score": round(best.score, 3)} if best else None,
        "candidates": [m.summary() for m in session.ranked()[:5]],
        "photos": [a.summary() for a in session.analyses],
        "comparison_image": image_path,
        "thresholds": {"match": fusion.thresholds()[0], "no_match": fusion.thresholds()[1]},
        "log": session.log,
    }
    db.insert_test({
        "id": test_id, "created_at": result["created_at"], "mode": session.mode,
        "reference_id": session.reference_id, "verdict": v.verdict, "confidence": v.confidence,
        "best_match_id": best.item_id if best else None, "explanation": v.explanation,
        "photo_paths": session.photo_paths, "log": session.log, "result": result,
    })
    return result
