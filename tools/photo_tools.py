"""Agent tool: look at the uploaded photos."""
from __future__ import annotations

from agent.session import AgentSession
from matching.features import compute_photo_features
from matching.photo import analyze_photo
from tools.registry import ToolResult, tool


@tool(
    "analyze_photos", "Segment photos",
    "Remove the background from every uploaded photo, isolate the object, measure its "
    "silhouette features and report photo quality problems. Call this first.",
)
def analyze_photos(session: AgentSession) -> ToolResult:
    if not session.analyses:
        for path in session.photo_paths:
            a = analyze_photo(path)
            session.analyses.append(a)
            if a.usable:
                session.photo_features.append(compute_photo_features(a))
    reports = []
    for a in session.analyses:
        r = a.summary()
        if a.usable:
            pf = next(p for p in session.photo_features if p.analysis is a)
            r["silhouette"] = {"aspect": round(float(pf.shape[0]), 2), "solidity": round(float(pf.shape[1]), 2),
                               "visible_holes": int(pf.shape[2])}
        reports.append(r)
    usable = len(session.usable_photos)
    warn = sum(len(a.warnings) for a in session.analyses)
    summary = f"{usable} of {len(session.analyses)} photo(s) usable" + (f", {warn} warning(s)" if warn else "")
    return ToolResult(summary, {"usable_photos": usable, "photos": reports})
