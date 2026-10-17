"""Start the CAD Match Agent: web UI at /ui, REST API at /api, API docs at /docs.

    python -m app.main                  # http://localhost:7860
    python -m app.main --port 8080 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import logging
import os

from core.env import load_dotenv

load_dotenv()
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from app.api import create_api  # noqa: E402
from app.ui import CSS, THEME, build_ui  # noqa: E402
from core.config import path  # noqa: E402


def create_app():
    app = create_api()

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui")

    return gr.mount_gradio_app(app, build_ui(), path="/ui", theme=THEME, css=CSS,
                               allowed_paths=[str(path("data_dir"))])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CAD Match Agent server")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    shown = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"\n  CAD Match Agent  ->  http://{shown}:{args.port}/ui   (API docs: /docs)\n")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
