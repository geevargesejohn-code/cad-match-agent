"""Tiny .env loader (no extra dependency). Existing environment variables win."""
from __future__ import annotations

import os
from pathlib import Path

from core.config import ROOT


def load_dotenv(path: Path | None = None) -> None:
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
