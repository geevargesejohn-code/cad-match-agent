"""SQLite storage for the CAD library and the test history. No server needed."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from core.config import path

SCHEMA = """
CREATE TABLE IF NOT EXISTS library (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    filename        TEXT NOT NULL,
    format          TEXT NOT NULL,
    sha256          TEXT NOT NULL UNIQUE,
    uploaded_at     TEXT NOT NULL,
    n_vertices      INTEGER,
    n_faces         INTEGER,
    watertight      INTEGER,
    proportions     TEXT,           -- JSON [1.0, b, c]
    original_extents TEXT,          -- JSON [x, y, z] in file units
    status          TEXT NOT NULL,  -- 'loaded' | 'ready' | 'error'
    error           TEXT
);

CREATE TABLE IF NOT EXISTS tests (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    mode            TEXT NOT NULL,  -- 'reference' | 'library_search'
    reference_id    TEXT,
    verdict         TEXT,           -- 'MATCH' | 'NOT A MATCH' | 'UNCERTAIN'
    confidence      REAL,
    best_match_id   TEXT,
    explanation     TEXT,
    photo_paths     TEXT,           -- JSON list
    log             TEXT,           -- JSON list of agent steps
    result          TEXT            -- JSON: the full result shown in the UI/API
);
"""

_JSON_FIELDS = {"proportions", "original_extents", "photo_paths", "log", "result"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    db_file = path("db_file")
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for key in _JSON_FIELDS & d.keys():
        if d[key] is not None:
            d[key] = json.loads(d[key])
    if "watertight" in d and d["watertight"] is not None:
        d["watertight"] = bool(d["watertight"])
    return d


def _encode(values: dict[str, Any]) -> dict[str, Any]:
    return {k: json.dumps(v) if k in _JSON_FIELDS and v is not None else v for k, v in values.items()}


# ---------- library ----------

def insert_item(item: dict[str, Any]) -> None:
    item = _encode(item)
    cols = ", ".join(item)
    marks = ", ".join("?" for _ in item)
    with connect() as c:
        c.execute(f"INSERT INTO library ({cols}) VALUES ({marks})", list(item.values()))


def update_item(item_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields = _encode(fields)
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE library SET {sets} WHERE id = ?", [*fields.values(), item_id])


def get_item(item_id: str) -> dict[str, Any] | None:
    with connect() as c:
        return _row_to_dict(c.execute("SELECT * FROM library WHERE id = ?", (item_id,)).fetchone())


def find_by_hash(sha256: str) -> dict[str, Any] | None:
    with connect() as c:
        return _row_to_dict(c.execute("SELECT * FROM library WHERE sha256 = ?", (sha256,)).fetchone())


def list_items() -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM library ORDER BY uploaded_at DESC, name").fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_item(item_id: str) -> bool:
    with connect() as c:
        return c.execute("DELETE FROM library WHERE id = ?", (item_id,)).rowcount > 0


# ---------- test history ----------

def insert_test(record: dict[str, Any]) -> None:
    record = _encode(record)
    cols = ", ".join(record)
    marks = ", ".join("?" for _ in record)
    with connect() as c:
        c.execute(f"INSERT INTO tests ({cols}) VALUES ({marks})", list(record.values()))


def get_test(test_id: str) -> dict[str, Any] | None:
    with connect() as c:
        return _row_to_dict(c.execute("SELECT * FROM tests WHERE id = ?", (test_id,)).fetchone())


def list_tests(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM tests ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_dict(r) for r in rows]
