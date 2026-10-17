"""Shared test setup.

Most tests get their own empty data folder. Tests that use the `shared_library`
fixture share one library with the 8 sample designs (indexed once per session,
because rendering every model for every test would be slow).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import reset_config_cache  # noqa: E402
from tests import shapes  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


def _use_data_dir(monkeypatch, d: Path) -> None:
    monkeypatch.setenv("CAD_MATCH_DATA_DIR", str(d))
    reset_config_cache()


@pytest.fixture(scope="session")
def _shared_library_dir(tmp_path_factory):
    """Build the shared sample library once."""
    d = tmp_path_factory.mktemp("shared_data")
    old = os.environ.get("CAD_MATCH_DATA_DIR")
    os.environ["CAD_MATCH_DATA_DIR"] = str(d)
    reset_config_cache()
    try:
        from matching import indexer
        from scripts.make_samples import DESIGNS, ensure_samples
        from storage import library

        info = ensure_samples(SAMPLES)
        indexer.install()
        ids = {}
        for name in DESIGNS:
            r = library.add_file(info["cad_dir"] / name)
            assert r.ok and r.item["status"] == "ready", r
            ids[name] = r.item["id"]
    finally:
        if old is None:
            os.environ.pop("CAD_MATCH_DATA_DIR", None)
        else:
            os.environ["CAD_MATCH_DATA_DIR"] = old
        reset_config_cache()
    return d, ids


@pytest.fixture(autouse=True)
def isolated_data_dir(request, tmp_path, monkeypatch):
    if "shared_library" in request.fixturenames:
        yield None  # shared_library sets the data dir
    else:
        _use_data_dir(monkeypatch, tmp_path / "data")
        yield tmp_path / "data"
    reset_config_cache()


@pytest.fixture
def shared_library(_shared_library_dir, monkeypatch):
    """{design file name: library id} for the 8 sample designs."""
    from matching import indexer

    d, ids = _shared_library_dir
    _use_data_dir(monkeypatch, d)
    indexer.install()
    return ids


@pytest.fixture
def photos():
    return SAMPLES / "photos"


@pytest.fixture
def cad_dir(tmp_path):
    d = tmp_path / "cad"
    d.mkdir()
    return d


@pytest.fixture
def bracket_step(cad_dir):
    return shapes.bracket_step(cad_dir / "bracket.step")
