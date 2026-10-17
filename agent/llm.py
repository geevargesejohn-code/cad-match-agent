"""Minimal tool-calling clients for Claude, OpenAI and Ollama (plain HTTPS, no SDKs).

All three share one neutral message format:
  {"role": "user", "content": [{"type": "text", "text": ...}, {"type": "image", "data": <base64 png>}]}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments": dict}]}
  {"role": "tool", "tool_call_id", "name", "content": str, "images": [<base64 png>]}
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import requests

from core.config import get_config


class ProviderUnavailable(Exception):
    """The provider is not configured (missing key, unknown name)."""


class ProviderError(Exception):
    """The provider was reachable but the call failed."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def png_b64(img: np.ndarray, max_side: int = 1100) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def _post(url: str, headers: dict, body: dict) -> dict:
    timeout = float(get_config()["agent"].get("request_timeout_s", 120))
    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as e:
        raise ProviderError(f"Could not reach {url}: {e}") from e
    if r.status_code >= 400:
        raise ProviderError(f"{url} returned {r.status_code}: {r.text[:400]}")
    return r.json()


class AnthropicProvider:
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self) -> None:
        self.key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not self.key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")
        self.model = os.environ.get("ANTHROPIC_MODEL") or get_config()["agent"]["anthropic_model"]

    def _messages(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": [self._part(p) for p in m["content"]]})
            elif m["role"] == "assistant":
                content = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                            for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": content})
            else:  # tool result; consecutive results share one user turn
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": [{"type": "text", "text": m["content"]}] +
                                    [self._part({"type": "image", "data": d}) for d in m.get("images", [])]}
                if out and out[-1]["role"] == "user" and out[-1]["content"] and \
                        out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    @staticmethod
    def _part(p: dict) -> dict:
        if p["type"] == "image":
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": p["data"]}}
        return {"type": "text", "text": p["text"]}

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model, "max_tokens": 1500, "system": system,
            "messages": self._messages(messages),
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools],
        }
        data = _post(self.URL, {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                "content-type": "application/json"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {})
                 for b in data.get("content", []) if b.get("type") == "tool_use"]
        return LLMResponse(text, calls)


class OpenAIProvider:
    """OpenAI Chat Completions; also used for Ollama's OpenAI-compatible endpoint."""
    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 name: str = "openai") -> None:
        self.name = name
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.key:
            raise ProviderUnavailable("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("OPENAI_MODEL") or get_config()["agent"]["openai_model"]

    def _messages(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        pending_images: list[tuple[str, str]] = []
        for m in messages:
            if m["role"] != "tool" and pending_images:
                out.append(self._image_turn(pending_images))
                pending_images = []
            if m["role"] == "user":
                out.append({"role": "user", "content": [self._part(p) for p in m["content"]]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                         for c in m["tool_calls"]]
                out.append(msg)
            else:
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
                pending_images += [(m["name"], d) for d in m.get("images", [])]
        if pending_images:
            out.append(self._image_turn(pending_images))
        return out

    def _image_turn(self, images: list[tuple[str, str]]) -> dict:
        # Tool messages cannot carry images in this API, so they follow as a user turn.
        parts: list[dict] = [{"type": "text", "text": "Image(s) returned by " + ", ".join(n for n, _ in images)}]
        parts += [self._part({"type": "image", "data": d}) for _, d in images]
        return {"role": "user", "content": parts}

    @staticmethod
    def _part(p: dict) -> dict:
        if p["type"] == "image":
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{p['data']}"}}
        return {"type": "text", "text": p["text"]}

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "tools": [{"type": "function", "function": t} for t in tools],
            "tool_choice": "auto",
        }
        data = _post(f"{self.base_url}/chat/completions",
                     {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}, body)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected response: {str(data)[:300]}") from e
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else dict(args)
            except json.JSONDecodeError:
                parsed = {}
            calls.append(ToolCall(c.get("id") or f"call_{i}", fn.get("name", ""), parsed))
        return LLMResponse(msg.get("content") or "", calls)


def ollama_provider() -> OpenAIProvider:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL") or get_config()["agent"]["ollama_model"]
    return OpenAIProvider(base_url=f"{base}/v1", api_key="ollama", model=model, name="ollama")


PROVIDERS = ("none", "anthropic", "openai", "ollama")


def default_provider_name() -> str:
    return (os.environ.get("CAD_MATCH_LLM_PROVIDER") or get_config()["agent"].get("provider") or "none").lower()


def get_provider(name: str | None = None):
    """Return a provider instance, or None for rule-based mode."""
    name = (name or default_provider_name()).lower()
    if name in ("", "none", "rules"):
        return None
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "ollama":
        return ollama_provider()
    raise ProviderUnavailable(f"Unknown provider '{name}'. Use one of: {', '.join(PROVIDERS)}")
