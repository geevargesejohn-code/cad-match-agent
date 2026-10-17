"""REST API and UI construction."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLES


@pytest.fixture
def client():
    from app.api import create_api
    return TestClient(create_api())


def test_library_crud(client):
    with open(SAMPLES / "cad" / "hex_nut.ply", "rb") as a, open(SAMPLES / "cad" / "plate_2holes.step", "rb") as b:
        r = client.post("/api/library", files=[("files", ("hex_nut.ply", a)), ("files", ("plate.step", b)),
                                               ("files", ("notes.txt", b"hello"))])
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["ok"] for x in results] == [True, True, False]
    nut_id = results[0]["id"]

    assert len(client.get("/api/library").json()) == 2
    assert client.get(f"/api/library/{nut_id}/thumbnail").headers["content-type"] == "image/png"
    assert client.get(f"/api/library/{nut_id}/file").status_code == 200
    assert client.patch(f"/api/library/{nut_id}", json={"name": "Nut"}).json()["name"] == "Nut"
    assert client.patch(f"/api/library/{nut_id}", json={"name": " "}).status_code == 400
    assert client.delete(f"/api/library/{nut_id}").status_code == 200
    assert client.get(f"/api/library/{nut_id}").status_code == 404


def test_run_test_and_history(shared_library, client):
    with open(SAMPLES / "photos" / "hex_nut_photo.jpg", "rb") as f:
        r = client.post("/api/test", data={"mode": "reference", "reference_id": shared_library["hex_nut.ply"]},
                        files=[("photos", ("nut.jpg", f, "image/jpeg"))])
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["verdict"] == "MATCH"
    tid = result["test_id"]
    assert any(h["id"] == tid for h in client.get("/api/history").json())
    assert client.get(f"/api/history/{tid}").json()["verdict"] == "MATCH"
    assert client.get(f"/api/history/{tid}/comparison.png").status_code == 200


def test_run_test_validation(client):
    with open(SAMPLES / "photos" / "hex_nut_photo.jpg", "rb") as f:
        r = client.post("/api/test", data={"mode": "library_search"}, files=[("photos", ("n.jpg", f))])
    assert r.status_code == 400 and "empty" in r.json()["detail"]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["embedding_backend"] in ("hog", "dinov2")


def test_ui_builds():
    from app.ui import build_ui
    demo = build_ui()
    assert demo is not None
