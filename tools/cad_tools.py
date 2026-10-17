"""CAD-side helpers that are not agent tools themselves (the agent reaches CAD data
through match_tools: list_library, refine_views, view_comparison)."""
from __future__ import annotations

from pathlib import Path

from cad_processing.loader import supported_formats
from storage import library


def add_cad_files(paths: list[str | Path], names: list[str] | None = None) -> list[dict]:
    """Add several files to the library (used by the UI and API)."""
    results = []
    for i, p in enumerate(paths):
        p = Path(p)
        r = library.add_file(p, filename=(names[i] if names else p.name))
        results.append({
            "file": r.filename, "ok": r.ok, "duplicate": r.duplicate, "error": r.error,
            "id": r.item["id"] if r.item else None,
            "status": r.item["status"] if r.item else None,
        })
    return results


def accepted_extensions() -> list[str]:
    return sorted(supported_formats())
