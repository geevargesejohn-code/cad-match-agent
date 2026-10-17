"""Tool registry: every action the agent can take, with a JSON schema for LLM tool calling."""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from agent.session import AgentSession


@dataclass
class ToolResult:
    summary: str                         # one human-readable line for the step log
    data: dict[str, Any]                 # structured result (sent to the LLM as JSON)
    images: list[np.ndarray] = field(default_factory=list)
    terminal: bool = False               # True ends the agent loop

    def as_text(self) -> str:
        return json.dumps(self.data, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., ToolResult]
    title: str                           # how the step appears in the UI log


TOOLS: dict[str, Tool] = {}


def tool(name: str, title: str, description: str, parameters: dict[str, Any] | None = None):
    params = parameters or {"type": "object", "properties": {}, "required": []}

    def wrap(fn):
        TOOLS[name] = Tool(name, description, params, fn, title)
        return fn
    return wrap


def schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in TOOLS.values() if names is None or t.name in names
    ]


def execute(session: AgentSession, name: str, args: dict[str, Any] | None = None) -> ToolResult:
    """Run a tool, record it in the session log, and never raise (errors become results)."""
    args = args or {}
    t = TOOLS.get(name)
    if t is None:
        res = ToolResult(f"Unknown tool '{name}'", {"error": f"unknown tool {name}"})
        session.add_log("error", f"Unknown tool {name}", res.summary)
        return res
    t0 = time.time()
    try:
        res = t.fn(session, **args)
    except TypeError as e:
        res = ToolResult(f"Bad arguments for {name}: {e}", {"error": f"bad arguments: {e}"})
    except Exception as e:  # keep the agent alive; the LLM or rules can react to the error
        res = ToolResult(f"{name} failed: {e}", {"error": str(e), "trace": traceback.format_exc(limit=3)})
    pretty_args = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in args.items() if k != "explanation")
    session.add_log(
        "tool", f"{t.title}" + (f" ({pretty_args})" if pretty_args else ""),
        res.summary, {"tool": name, "args": args, "elapsed_s": round(time.time() - t0, 2)},
    )
    return res


def load_all() -> None:
    """Import tool modules so they register themselves."""
    from tools import cad_tools, match_tools, photo_tools  # noqa: F401
