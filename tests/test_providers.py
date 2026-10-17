"""LLM providers: message conversion and response parsing (no network)."""
from __future__ import annotations

import pytest

from agent import llm

TOOLS = [{"name": "submit_verdict", "description": "finish", "parameters": {"type": "object", "properties": {}}}]
MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Is it a match?"}]},
    {"role": "assistant", "text": "Checking.", "tool_calls": [
        {"id": "t1", "name": "view_comparison", "arguments": {"model_id": "abc"}},
        {"id": "t2", "name": "list_library", "arguments": {}}]},
    {"role": "tool", "tool_call_id": "t1", "name": "view_comparison", "content": "{}", "images": ["QUJD"]},
    {"role": "tool", "tool_call_id": "t2", "name": "list_library", "content": "{\"models\": []}", "images": []},
]


@pytest.fixture
def capture(monkeypatch):
    sent = {}

    def fake_post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return sent.pop("reply")
    monkeypatch.setattr(llm, "_post", fake_post)
    return sent


def test_anthropic_format(monkeypatch, capture):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    capture["reply"] = {"content": [
        {"type": "text", "text": "Done."},
        {"type": "tool_use", "id": "u1", "name": "submit_verdict", "input": {"verdict": "MATCH"}}]}
    resp = llm.AnthropicProvider().chat("sys", MESSAGES, TOOLS)

    body = capture["body"]
    assert capture["headers"]["x-api-key"] == "sk-test"
    assert body["system"] == "sys" and body["tools"][0]["input_schema"]["type"] == "object"
    # both tool results go back in ONE user turn, with the image inside the first result
    results = body["messages"][2]
    assert results["role"] == "user" and [b["type"] for b in results["content"]] == ["tool_result", "tool_result"]
    assert results["content"][0]["content"][1]["type"] == "image"
    assert resp.text == "Done." and resp.tool_calls[0].arguments == {"verdict": "MATCH"}


def test_openai_format(monkeypatch, capture):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    capture["reply"] = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "function": {"name": "submit_verdict", "arguments": "{\"verdict\": \"NOT A MATCH\"}"}}]}}]}
    resp = llm.OpenAIProvider().chat("sys", MESSAGES, TOOLS)

    msgs = capture["body"]["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == "{\"model_id\": \"abc\"}"
    assert [m["role"] for m in msgs[3:]] == ["tool", "tool", "user"]   # images follow as a user turn
    assert msgs[5]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert resp.tool_calls[0].name == "submit_verdict"
    assert resp.tool_calls[0].arguments == {"verdict": "NOT A MATCH"}


def test_ollama_uses_openai_compatible_endpoint(monkeypatch, capture):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/")
    capture["reply"] = {"choices": [{"message": {"content": "hi", "tool_calls": None}}]}
    p = llm.get_provider("ollama")
    p.chat("sys", MESSAGES[:1], TOOLS)
    assert capture["url"] == "http://gpu-box:11434/v1/chat/completions"


def test_missing_keys_and_unknown_providers(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.get_provider("none") is None
    with pytest.raises(llm.ProviderUnavailable):
        llm.get_provider("anthropic")
    with pytest.raises(llm.ProviderUnavailable):
        llm.get_provider("openai")
    with pytest.raises(llm.ProviderUnavailable):
        llm.get_provider("gpt-17")
