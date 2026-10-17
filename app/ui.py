"""Gradio web UI: CAD Library | Upload Object Photo | Run Test | Results / History."""
from __future__ import annotations

import html
import queue
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from agent.llm import PROVIDERS, default_provider_name
from agent.loop import run_test
from core.config import ROOT
from matching import indexer
from matching.embeddings import get_embedder
from matching.photo import analyze_photo, _rembg_available
from storage import cache, db, library
from tools import registry
from tools.cad_tools import accepted_extensions, add_cad_files

SAMPLES = ROOT / "data" / "samples"
MODE_REFERENCE = "Reference test: compare with one CAD file"
MODE_SEARCH = "Library search: find the matching CAD file"
VERDICT_STYLE = {
    "MATCH": ("#15803d", "#dcfce7", "✓ MATCH"),
    "NOT A MATCH": ("#b91c1c", "#fee2e2", "✗ NOT A MATCH"),
    "UNCERTAIN": ("#b45309", "#fef3c7", "? UNCERTAIN: another photo needed"),
}

CSS = """
.verdict-card {border-radius: 14px; padding: 18px 22px; border: 2px solid; margin-bottom: 6px;}
.verdict-title {font-size: 1.9rem; font-weight: 800; letter-spacing: .02em;}
.verdict-meta {opacity: .85; margin-top: 4px;}
.verdict-expl {margin-top: 10px; font-size: 1.02rem; line-height: 1.45;}
.step {padding: 6px 0; border-bottom: 1px solid rgba(128,128,128,.18);}
.step-kind {display:inline-block; min-width: 74px; font-size: .75rem; font-weight: 700; text-transform: uppercase;
            opacity: .7;}
"""


# ---------------- helpers ----------------

def _choices() -> list[tuple[str, str]]:
    return [(f"{i['name']}  ({i['format']}, {i['id']})", i["id"]) for i in library.list_items()]


def _library_table() -> pd.DataFrame:
    rows = [{
        "Name": i["name"], "Format": i["format"].upper(), "Faces": i["n_faces"],
        "Proportions": ", ".join(f"{x:.2f}" for x in (i["proportions"] or [])),
        "Status": i["status"], "Uploaded": i["uploaded_at"].replace("T", " ")[:16], "ID": i["id"],
    } for i in library.list_items()]
    return pd.DataFrame(rows, columns=["Name", "Format", "Faces", "Proportions", "Status", "Uploaded", "ID"])


def _gallery() -> list[tuple[str, str]]:
    out = []
    for i in library.list_items():
        p = cache.thumbnail_path(i["id"])
        if p.exists():
            out.append((str(p), f"{i['name']} ({i['format']})"))
    return out


def _library_outputs(status: str = ""):
    ch = _choices()
    return (status, _gallery(), _library_table(),
            gr.update(choices=ch, value=None), gr.update(choices=ch, value=ch[0][1] if ch else None))


def verdict_html(r: dict) -> str:
    color, bg, title = VERDICT_STYLE.get(r["verdict"], ("#444", "#eee", r["verdict"]))
    meta = [f"Confidence {100 * r['confidence']:.0f}%"]
    if r.get("reference"):
        meta.append(f"Reference: <b>{html.escape(r['reference']['name'])}</b>")
    if r.get("best_match"):
        label = "Model" if r["mode"] == "reference" else "Best match"
        meta.append(f"{label}: <b>{html.escape(r['best_match']['name'])}</b> (score {r['best_match']['score']:.2f})")
    who = r.get("decided_by", "")
    meta.append("Decided by: " + ("LLM judge (" + who.split(":", 1)[1] + ")" if who.startswith("llm:") else "rule-based agent"))
    if r.get("numeric_verdict") and r["numeric_verdict"] != r["verdict"]:
        meta.append(f"Numeric rules said: {r['numeric_verdict']}")
    return (f'<div class="verdict-card" style="border-color:{color};background:{bg};color:#111">'
            f'<div class="verdict-title" style="color:{color}">{title}</div>'
            f'<div class="verdict-meta">{" &nbsp;·&nbsp; ".join(meta)}</div>'
            f'<div class="verdict-expl">{html.escape(r["explanation"])}</div></div>')


def log_html(steps: list[dict]) -> str:
    if not steps:
        return "<i>The agent's steps will appear here.</i>"
    rows = []
    for s in steps:
        detail = html.escape(str(s.get("detail") or ""))
        rows.append(f'<div class="step"><span class="step-kind">{s["kind"]}</span> '
                    f'<b>{html.escape(s["title"])}</b> <span style="opacity:.6">({s["t"]}s)</span>'
                    f'<div style="margin-left:78px">{detail}</div></div>')
    return "".join(rows)


def _candidates_table(r: dict) -> pd.DataFrame:
    rows = []
    for c in r.get("candidates", []):
        sig = c["per_photo"][0]["signals"] if c["per_photo"] else {}
        rows.append({"Model": c["model_name"], "Score": c["score"], "Own verdict": c["numeric_verdict"],
                     "Silhouette": sig.get("iou"), "Edges": sig.get("edges"),
                     "Inner features": sig.get("inner_edges"), "Appearance": sig.get("embedding"),
                     "Proportions": sig.get("proportions"), "Views": c["views_compared"]})
    return pd.DataFrame(rows)


def _load_image(path: str | None):
    if not path or not Path(path).exists():
        return None
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# ---------------- actions ----------------

def add_files(files):
    if not files:
        return _library_outputs("Choose one or more CAD files first.")
    paths = [Path(f) for f in files]
    results = add_cad_files(paths, [p.name for p in paths])
    lines = []
    for r in results:
        if not r["ok"]:
            lines.append(f"❌ **{r['file']}**: {r['error']}")
        elif r["duplicate"]:
            lines.append(f"↺ **{r['file']}**: already in the library")
        else:
            lines.append(f"✅ **{r['file']}** added ({r['status']})")
    return _library_outputs("\n\n".join(lines))


def add_samples():
    from scripts.make_samples import DESIGNS, ensure_samples
    info = ensure_samples()
    paths = [info["cad_dir"] / n for n in DESIGNS]
    add_cad_files(paths, [p.name for p in paths])
    return _library_outputs(f"Added the {len(paths)} sample CAD files.")


def rename_item(item_id, new_name):
    if not item_id:
        return _library_outputs("Select a model first.")
    try:
        item = library.rename(item_id, new_name or "")
        return _library_outputs(f"Renamed to **{item['name']}**.")
    except (KeyError, ValueError) as e:
        return _library_outputs(f"❌ {e}")


def delete_item(item_id):
    if not item_id:
        return _library_outputs("Select a model first.")
    item = library.get_item(item_id)
    library.delete(item_id)
    return _library_outputs(f"Deleted **{item['name'] if item else item_id}**.")


def check_photos(files):
    if not files:
        return [], "Upload one or more photos first."
    previews, notes = [], []
    for f in files:
        a = analyze_photo(f)
        vis = a.rgb.copy()
        vis[~a.mask] = (vis[~a.mask] * 0.3).astype(np.uint8)
        contours, _ = cv2.findContours(a.mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 70, 70), 2)
        name = Path(f).name
        previews.append((vis, name))
        status = "✅ usable" if a.usable else "❌ not usable"
        warn = ("; ".join(a.warnings)) if a.warnings else "no problems found"
        notes.append(f"**{name}**: {status}, object covers {100 * a.area_fraction:.0f}% of the frame "
                     f"({a.backend} segmentation): {warn}")
    return previews, "\n\n".join(notes)


def sample_photos(names):
    if not names:
        return None
    return [str(SAMPLES / "photos" / n) for n in names]


def run(files, mode_label, reference_id, provider):
    empty = (gr.update(), None, "", pd.DataFrame(), None)
    if not files:
        yield ("<b>Upload at least one photo in the 'Upload Object Photo' tab.</b>",) + empty[1:]
        return
    mode = "reference" if mode_label == MODE_REFERENCE else "library_search"

    steps: queue.Queue = queue.Queue()
    box: dict = {}

    def worker():
        try:
            box["result"] = run_test(list(files), mode, reference_id if mode == "reference" else None,
                                     provider, photo_names=[Path(f).name for f in files], on_step=steps.put)
        except Exception as e:  # shown in the UI instead of crashing it
            box["error"] = str(e)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    seen: list[dict] = []
    while th.is_alive() or not steps.empty():
        try:
            seen.append(steps.get(timeout=0.3))
            yield ("<i>Agent is working…</i>", None, log_html(seen), pd.DataFrame(), None)
        except queue.Empty:
            continue
    th.join()
    if "error" in box:
        yield (f'<div class="verdict-card" style="border-color:#b91c1c">❌ {html.escape(box["error"])}</div>',
               None, log_html(seen), pd.DataFrame(), None)
        return
    r = box["result"]
    yield verdict_html(r), _load_image(r["comparison_image"]), log_html(r["log"]), _candidates_table(r), r


def history_table():
    rows = []
    names = {i["id"]: i["name"] for i in library.list_items()}
    for t in db.list_tests(200):
        rows.append({"When": t["created_at"].replace("T", " ")[:19],
                     "Mode": "Reference" if t["mode"] == "reference" else "Library search",
                     "Verdict": t["verdict"], "Confidence": round(t["confidence"] or 0, 2),
                     "Model": names.get(t["best_match_id"] or t["reference_id"], "(deleted)"),
                     "Decided by": (t["result"] or {}).get("decided_by", ""), "Test ID": t["id"]})
    df = pd.DataFrame(rows, columns=["When", "Mode", "Verdict", "Confidence", "Model", "Decided by", "Test ID"])
    ids = [(f"{r['When']}  {r['Verdict']}  ({r['Test ID']})", r["Test ID"]) for r in rows]
    return df, gr.update(choices=ids, value=ids[0][1] if ids else None)


def show_test(test_id):
    t = db.get_test(test_id) if test_id else None
    if not t:
        return "", None, ""
    r = t["result"]
    return verdict_html(r), _load_image(r.get("comparison_image")), log_html(r["log"])


def status_line() -> str:
    seg = "rembg (U²-Net) + hole recovery" if _rembg_available() else "classical (border color + GrabCut)"
    return (f"Appearance model: **{get_embedder().name}** · Segmentation: **{seg}** · "
            f"Default LLM judge: **{default_provider_name()}** · Library: **{len(library.list_items())} models**")


# ---------------- layout ----------------

def build_ui() -> gr.Blocks:
    indexer.install()
    registry.load_all()
    sample_photo_names = sorted(p.name for p in (SAMPLES / "photos").glob("*.jpg")) if (SAMPLES / "photos").exists() else []

    with gr.Blocks(title="CAD Match Agent") as demo:
        gr.Markdown("# CAD Match Agent\nDoes the real object match the CAD design? Size doesn't matter; "
                    "shape, proportions and features do.")
        status = gr.Markdown(status_line())

        with gr.Tab("CAD Library"):
            with gr.Row():
                with gr.Column(scale=2):
                    cad_files = gr.File(label="Upload CAD files (" + ", ".join(accepted_extensions()) + ")",
                                        file_count="multiple", file_types=accepted_extensions())
                    with gr.Row():
                        add_btn = gr.Button("Add to library", variant="primary")
                        sample_btn = gr.Button("Add sample CAD files")
                    lib_status = gr.Markdown()
                with gr.Column(scale=3):
                    gallery = gr.Gallery(label="Library", columns=4, height=360, object_fit="contain",
                                         value=_gallery())
            lib_table = gr.Dataframe(value=_library_table(), label="All models", interactive=False)
            with gr.Row():
                pick = gr.Dropdown(label="Select a model", choices=_choices(), scale=3)
                new_name = gr.Textbox(label="New name", scale=2)
                rename_btn = gr.Button("Rename", scale=1)
                delete_btn = gr.Button("Delete", variant="stop", scale=1)

        with gr.Tab("Upload Object Photo"):
            gr.Markdown("Upload one or more photos of the real object. **Tips:** plain background that contrasts "
                        "with the part, object fills about a third of the frame, no other objects in view. "
                        "Extra photos from different angles make the result more reliable.")
            with gr.Row():
                with gr.Column():
                    photos = gr.File(label="Object photo(s)", file_count="multiple", file_types=["image"])
                    if sample_photo_names:
                        sample_pick = gr.Dropdown(sample_photo_names, multiselect=True,
                                                  label="…or try bundled sample photos")
                    check_btn = gr.Button("Check photos (preview background removal)")
                    photo_notes = gr.Markdown()
                with gr.Column():
                    photo_preview = gr.Gallery(label="What the agent sees (red outline = detected object)",
                                               columns=2, height=420, object_fit="contain")

        with gr.Tab("Run Test"):
            with gr.Row():
                mode = gr.Radio([MODE_REFERENCE, MODE_SEARCH], value=MODE_REFERENCE, label="Test type")
                provider = gr.Dropdown(list(PROVIDERS), value=default_provider_name(),
                                       label="LLM judge (none = rule-based agent)")
            ref = gr.Dropdown(label="Reference CAD model", choices=_choices(),
                              value=(_choices()[0][1] if _choices() else None))
            run_btn = gr.Button("Run test", variant="primary", size="lg")
            verdict = gr.HTML()
            with gr.Row():
                comparison = gr.Image(label="Photo vs CAD (best pose)", type="numpy", show_label=False)
            with gr.Accordion("Agent steps", open=True):
                log_box = gr.HTML(log_html([]))
            candidates = gr.Dataframe(label="Candidates (signals at best pose, 0-1)", interactive=False)
            with gr.Accordion("Raw result (JSON)", open=False):
                raw = gr.JSON()

        with gr.Tab("Results / History"):
            refresh_btn = gr.Button("Refresh")
            hist = gr.Dataframe(label="Past tests", interactive=False)
            hist_pick = gr.Dropdown(label="Open a test")
            h_verdict = gr.HTML()
            h_image = gr.Image(label="Comparison", type="numpy", show_label=False)
            with gr.Accordion("Agent steps", open=False):
                h_log = gr.HTML()

        # wiring
        lib_outputs = [lib_status, gallery, lib_table, pick, ref]
        add_btn.click(add_files, cad_files, lib_outputs).then(status_line, None, status)
        sample_btn.click(add_samples, None, lib_outputs).then(status_line, None, status)
        rename_btn.click(rename_item, [pick, new_name], lib_outputs)
        delete_btn.click(delete_item, pick, lib_outputs).then(status_line, None, status)
        if sample_photo_names:
            sample_pick.change(sample_photos, sample_pick, photos)
        check_btn.click(check_photos, photos, [photo_preview, photo_notes])
        mode.change(lambda m: gr.update(visible=(m == MODE_REFERENCE)), mode, ref)
        run_btn.click(run, [photos, mode, ref, provider], [verdict, comparison, log_box, candidates, raw])
        refresh_btn.click(history_table, None, [hist, hist_pick])
        hist_pick.change(show_test, hist_pick, [h_verdict, h_image, h_log])
        demo.load(history_table, None, [hist, hist_pick])
        # fresh library state for every visitor (it may have changed through the API)
        demo.load(lambda: _library_outputs(""), None, lib_outputs).then(status_line, None, status)
    return demo


THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
