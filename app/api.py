"""REST API (FastAPI). Interactive docs at /docs when the server is running."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.llm import default_provider_name
from agent.loop import run_test
from matching import indexer
from matching.embeddings import get_embedder
from storage import cache, db, library
from tools import registry
from tools.cad_tools import accepted_extensions, add_cad_files

router = APIRouter(prefix="/api")


class RenameBody(BaseModel):
    name: str


def _save_uploads(files: list[UploadFile], tmp: Path) -> tuple[list[Path], list[str]]:
    paths, names = [], []
    for i, f in enumerate(files):
        name = Path(f.filename or f"upload_{i}").name
        dest = tmp / f"{i:03d}_{name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
        names.append(name)
    return paths, names


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "library_size": len(library.list_items()),
            "embedding_backend": get_embedder().name, "llm_provider": default_provider_name()}


# ---------------- CAD library ----------------

@router.get("/library")
def list_library() -> list[dict]:
    return library.list_items()


@router.post("/library")
def upload_cad(files: list[UploadFile] = File(..., description="One or more CAD files")) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        paths, names = _save_uploads(files, Path(tmp))
        results = add_cad_files(paths, names)
    return {"results": results, "accepted_formats": accepted_extensions()}


@router.get("/library/{item_id}")
def get_item(item_id: str) -> dict:
    item = library.get_item(item_id)
    if not item:
        raise HTTPException(404, "No such library item")
    return item


@router.patch("/library/{item_id}")
def rename_item(item_id: str, body: RenameBody) -> dict:
    try:
        return library.rename(item_id, body.name)
    except KeyError:
        raise HTTPException(404, "No such library item")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/library/{item_id}")
def delete_item(item_id: str) -> dict:
    if not library.delete(item_id):
        raise HTTPException(404, "No such library item")
    return {"deleted": item_id}


@router.get("/library/{item_id}/thumbnail")
def thumbnail(item_id: str):
    p = cache.thumbnail_path(item_id)
    if not p.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(p, media_type="image/png")


@router.get("/library/{item_id}/file")
def original_file(item_id: str):
    p = library.original_file(item_id)
    if not p or not p.exists():
        raise HTTPException(404, "No such library item")
    return FileResponse(p, filename=p.name)


# ---------------- tests ----------------

@router.post("/test")
def create_test(
    photos: list[UploadFile] = File(..., description="One or more photos of the object"),
    mode: Literal["reference", "library_search"] = Form("library_search"),
    reference_id: str | None = Form(None),
    provider: str | None = Form(None, description="none | anthropic | openai | ollama"),
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        paths, names = _save_uploads(photos, Path(tmp))
        try:
            return run_test(paths, mode, reference_id or None, provider or None, photo_names=names)
        except ValueError as e:
            raise HTTPException(400, str(e))


@router.get("/history")
def history(limit: int = 50) -> list[dict]:
    return [
        {k: t[k] for k in ("id", "created_at", "mode", "reference_id", "verdict", "confidence",
                           "best_match_id", "explanation")}
        for t in db.list_tests(limit)
    ]


@router.get("/history/{test_id}")
def history_item(test_id: str) -> dict:
    t = db.get_test(test_id)
    if not t:
        raise HTTPException(404, "No such test")
    return t["result"]


@router.get("/history/{test_id}/comparison.png")
def comparison_png(test_id: str):
    t = db.get_test(test_id)
    img = t and t["result"].get("comparison_image")
    if not img or not Path(img).exists():
        raise HTTPException(404, "No comparison image")
    return FileResponse(img, media_type="image/png")


def create_api() -> FastAPI:
    indexer.install()
    registry.load_all()
    app = FastAPI(
        title="CAD Match Agent API",
        description="Check whether a real object in a photo matches a CAD design, regardless of size.",
        version="1.0.0",
    )
    app.include_router(router)
    return app
