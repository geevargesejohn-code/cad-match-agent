"""The agent loop: rule mode, LLM mode with a scripted model, and every fallback path."""
from __future__ import annotations

import itertools

import pytest

from agent import loop
from agent.llm import LLMResponse, ProviderError, ProviderUnavailable, ToolCall
from matching import fusion
from storage import db


class ScriptedProvider:
    """Plays back a fixed list of tool calls, like an LLM would, and records what it was sent."""
    name = "scripted"
    model = "test-model"

    def __init__(self, script):
        self.script = iter(script)
        self.received = []
        self._ids = itertools.count()

    def chat(self, system, messages, tools):
        self.received.append({"system": system, "messages": list(messages), "tools": tools})
        step = next(self.script, None)
        if step is None:
            return LLMResponse("I am done thinking.")
        text, calls = step
        return LLMResponse(text, [ToolCall(f"c{next(self._ids)}", n, a) for n, a in calls])


def _use(monkeypatch, provider):
    monkeypatch.setattr(loop, "get_provider", lambda name=None: provider)


def test_rule_agent_logs_every_step(shared_library, photos):
    r = loop.run_test([photos / "plate_2holes_photo.jpg"], "reference", shared_library["plate_2holes.step"])
    kinds = [s["kind"] for s in r["log"]]
    titles = " | ".join(s["title"] for s in r["log"])
    assert kinds[0] == "plan"
    assert "Segment photos" in titles and "Compare with model" in titles and "Submit verdict" in titles
    assert r["decided_by"] == "rules"
    saved = db.get_test(r["test_id"])
    assert saved["verdict"] == r["verdict"] and saved["result"]["log"] == r["log"]


def test_llm_agent_drives_tools_and_decides(shared_library, photos, monkeypatch):
    ref = shared_library["plate_2holes.step"]
    provider = ScriptedProvider([
        ("Let me look at the photo first.", [("analyze_photos", {})]),
        ("Now compare with the reference.", [("compare_with_model", {"model_id": ref})]),
        ("Check other designs.", [("check_alternatives", {"model_id": ref})]),
        ("Look at it.", [("view_comparison", {"model_id": ref})]),
        ("Outline and both holes agree.", [("submit_verdict", {
            "verdict": "MATCH", "model_id": ref, "confidence": 0.9,
            "explanation": "The outline and both holes line up."})]),
    ])
    _use(monkeypatch, provider)
    r = loop.run_test([photos / "plate_2holes_photo.jpg"], "reference", ref, provider="scripted")
    assert r["verdict"] == fusion.MATCH
    assert r["decided_by"] == "llm:scripted"
    assert r["explanation"] == "The outline and both holes line up."
    assert any(s["kind"] == "thought" and "both holes" in s["detail"] for s in r["log"])
    # the image from view_comparison was sent back to the model
    last = provider.received[-1]["messages"]
    assert any(m["role"] == "tool" and m["name"] == "view_comparison" and m["images"] for m in last)
    assert {t["name"] for t in provider.received[0]["tools"]} >= {"submit_verdict", "check_alternatives"}


def test_llm_that_never_decides_is_finished_by_rules(shared_library, photos, monkeypatch):
    _use(monkeypatch, ScriptedProvider([("Analyzing.", [("analyze_photos", {})])]))
    r = loop.run_test([photos / "sprocket_photo.jpg"], "library_search", provider="scripted")
    assert r["verdict"] == fusion.MATCH and r["decided_by"] == "rules"
    assert any(s["kind"] == "warning" for s in r["log"])


def test_llm_errors_fall_back_to_rules(shared_library, photos, monkeypatch):
    class Broken:
        name, model = "broken", "x"

        def chat(self, *a):
            raise ProviderError("503 overloaded")

    _use(monkeypatch, Broken())
    r = loop.run_test([photos / "sprocket_photo.jpg"], "library_search", provider="broken")
    assert r["verdict"] == fusion.MATCH and r["decided_by"] == "rules"
    assert any(s["kind"] == "error" and "503" in s["detail"] for s in r["log"])


def test_missing_api_key_falls_back_to_rules(shared_library, photos, monkeypatch):
    def unavailable(name=None):
        raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(loop, "get_provider", unavailable)
    r = loop.run_test([photos / "sprocket_photo.jpg"], "library_search", provider="anthropic")
    assert r["decided_by"] == "rules"
    assert any("ANTHROPIC_API_KEY" in s["detail"] for s in r["log"])


def test_bad_tool_calls_do_not_crash(shared_library, photos, monkeypatch):
    _use(monkeypatch, ScriptedProvider([
        ("", [("delete_everything", {})]),
        ("", [("compare_with_model", {"model_id": "does-not-exist"})]),
        ("", [("submit_verdict", {"verdict": "MAYBE", "confidence": 2, "explanation": "?"})]),
    ]))
    r = loop.run_test([photos / "sprocket_photo.jpg"], "library_search", provider="scripted")
    assert r["verdict"] in (fusion.MATCH, fusion.NOT_A_MATCH, fusion.UNCERTAIN)
    assert r["decided_by"] == "rules"


def test_unusable_photo_asks_for_a_better_one(shared_library, tmp_path):
    import numpy as np
    from PIL import Image

    blank = tmp_path / "blank.jpg"
    Image.fromarray(np.full((300, 400, 3), 128, np.uint8)).save(blank)
    r = loop.run_test([blank], "library_search")
    assert r["verdict"] == fusion.UNCERTAIN and r["needs_more_photos"]


def test_multiple_photos_are_combined(shared_library, photos):
    r = loop.run_test([photos / "plate_2holes_photo.jpg", photos / "plate_2holes_photo_2.jpg"],
                      "reference", shared_library["plate_2holes.step"])
    assert r["verdict"] == fusion.MATCH
    assert len(r["photos"]) == 2 and len(r["candidates"][0]["per_photo"]) == 2


def test_input_validation(shared_library, photos):
    with pytest.raises(ValueError):
        loop.run_test([], "library_search")
    with pytest.raises(ValueError):
        loop.run_test([photos / "sprocket_photo.jpg"], "reference", None)
    with pytest.raises(ValueError):
        loop.run_test([photos / "sprocket_photo.jpg"], "something")
