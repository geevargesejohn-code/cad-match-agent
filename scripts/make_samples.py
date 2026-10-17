"""Generate the bundled demo data: 8 CAD parts in every supported format, plus
synthetic "photos" of them (rendered with a perspective camera, metallic color,
textured background, noise, blur and JPEG compression).

    python -m scripts.make_samples            # writes into data/samples/
    python -m scripts.make_samples --force    # regenerate everything

The parts are original, simple mechanical designs, so there are no licensing issues.
"""
from __future__ import annotations

import argparse
import csv
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from cad_processing import renderer
from cad_processing.normalize import normalize_mesh
from core.config import ROOT

SAMPLES_DIR = ROOT / "data" / "samples"


# ------------------------------------------------------------------ CAD designs

def _occ_part(build, out: Path) -> Path:
    """Run a gmsh/OpenCascade build function and write STEP, or STL/OBJ/PLY via a mesh."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add(out.stem)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        if out.suffix.lower() in (".step", ".stp"):
            gmsh.write(str(out))
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 40)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
            gmsh.model.mesh.generate(2)
            with tempfile.TemporaryDirectory() as tmp:
                stl = Path(tmp) / "part.stl"
                gmsh.write(str(stl))
                trimesh.load(stl, force="mesh").export(out)
    finally:
        gmsh.finalize()
    return out


def plate_holes(n_holes: int, s: float = 1.0):
    def build(occ):
        plate = occ.addBox(0, 0, 0, 60 * s, 24 * s, 5 * s)
        xs = np.linspace(12, 48, n_holes) * s
        holes = [(3, occ.addCylinder(x, 12 * s, -1, 0, 0, 5 * s + 2, 3.2 * s)) for x in xs]
        occ.cut([(3, plate)], holes)
    return build


def flange(s: float = 1.0):
    def build(occ):
        disk = occ.addCylinder(0, 0, 0, 0, 0, 6 * s, 25 * s)
        cuts = [(3, occ.addCylinder(0, 0, -1, 0, 0, 6 * s + 2, 9 * s))]
        for k in range(4):
            a = k * math.pi / 2 + math.pi / 4
            cuts.append((3, occ.addCylinder(17 * s * math.cos(a), 17 * s * math.sin(a), -1, 0, 0, 6 * s + 2, 2.6 * s)))
        occ.cut([(3, disk)], cuts)
    return build


def l_bracket(s: float = 1.0):
    def build(occ):
        base = occ.addBox(0, 0, 0, 50 * s, 30 * s, 5 * s)
        wall = occ.addBox(0, 0, 0, 5 * s, 30 * s, 35 * s)
        fused, _ = occ.fuse([(3, base)], [(3, wall)])
        hole = occ.addCylinder(32 * s, 15 * s, -1, 0, 0, 7 * s, 4 * s)
        occ.cut(fused, [(3, hole)])
    return build


def t_bracket(s: float = 1.0):
    def build(occ):
        top = occ.addBox(0, 0, 0, 60 * s, 14 * s, 6 * s)
        stem = occ.addBox(23 * s, 14 * s, 0, 14 * s, 36 * s, 6 * s)
        occ.fuse([(3, top)], [(3, stem)])
    return build


def hex_nut(s: float = 1.0):
    def build(occ):
        pts = [occ.addPoint(14 * s * math.cos(k * math.pi / 3), 14 * s * math.sin(k * math.pi / 3), 0) for k in range(6)]
        lines = [occ.addLine(pts[k], pts[(k + 1) % 6]) for k in range(6)]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        body = occ.extrude([(2, face)], 0, 0, 11 * s)
        vol = [e for e in body if e[0] == 3]
        occ.cut(vol, [(3, occ.addCylinder(0, 0, -1, 0, 0, 11 * s + 2, 7 * s))])
    return build


def spacer_tube(s: float = 1.0):
    def build(occ):
        outer = occ.addCylinder(0, 0, 0, 0, 0, 45 * s, 10 * s)
        occ.cut([(3, outer)], [(3, occ.addCylinder(0, 0, -1, 0, 0, 45 * s + 2, 6.5 * s))])
    return build


def sprocket(s: float = 1.0):
    def build(occ):
        disk = occ.addCylinder(0, 0, 0, 0, 0, 5 * s, 20 * s)
        teeth = []
        for k in range(10):
            box = occ.addBox(-3 * s, 18 * s, 0, 6 * s, 7 * s, 5 * s)
            occ.rotate([(3, box)], 0, 0, 0, 0, 0, 1, k * 2 * math.pi / 10)
            teeth.append((3, box))
        fused, _ = occ.fuse([(3, disk)], teeth)
        occ.cut(fused, [(3, occ.addCylinder(0, 0, -1, 0, 0, 7 * s, 6 * s))])
    return build


DESIGNS = {
    # file name               builder
    "plate_2holes.step": plate_holes(2),
    "plate_3holes.step": plate_holes(3),
    "flange_50mm.step": flange(1.0),
    "l_bracket.stl": l_bracket(),
    "t_bracket.obj": t_bracket(),
    "hex_nut.ply": hex_nut(),
    "spacer_tube.stl": spacer_tube(),
    "sprocket.stp": sprocket(),
}

# A 10x larger flange: demonstrates that size never matters.
SCALED = {"flange_500mm.step": flange(10.0)}


def build_cad(out_dir: Path, force: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, builder in {**DESIGNS, **SCALED}.items():
        p = out_dir / name
        if force or not p.exists():
            _occ_part(builder, p)
        paths[name] = p
    return paths


# ------------------------------------------------------------------ synthetic photos

def _background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = rng.choice([[200, 196, 188], [150, 115, 80], [60, 95, 70], [225, 228, 232], [70, 72, 80]])
    base = np.asarray(base, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1 + 0.18 * ((xx / w - 0.5) * rng.uniform(-1, 1) + (yy / h - 0.5) * rng.uniform(-1, 1))
    img = base[None, None] * grad[..., None]
    grain = cv2.GaussianBlur(rng.normal(0, 9, (h, w)).astype(np.float32), (0, 0), 2.5)
    img += grain[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def synth_photo(mesh: trimesh.Trimesh, seed: int, size=(640, 480)) -> np.ndarray:
    """A realistic-ish photo of a part lying on a table, from a random viewpoint."""
    rng = np.random.default_rng(seed)
    w, h = size
    elev = math.radians(rng.uniform(35, 75))
    azim = rng.uniform(0, 2 * math.pi)
    direction = [math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)]
    color = tuple(int(c) for c in rng.choice([[175, 178, 185], [150, 160, 185], [190, 160, 95], [120, 122, 128]]))
    # Like a real user following the photo tips: background contrasts with the part.
    for _ in range(20):
        bg = _background(rng, w * 2, h * 2)
        if abs(float(np.mean(bg)) - float(np.mean(color))) > 55:
            break
    img, _ = renderer.render(
        normalize_mesh(mesh), direction, size=(w, h), roll_deg=rng.uniform(0, 360),
        fov_deg=rng.uniform(28, 42), color=color, background=bg, fill=rng.uniform(0.55, 0.8),
    )
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.0))
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


PHOTOS = [
    # (photo file,                   source CAD,          seed)
    ("plate_2holes_photo.jpg",       "plate_2holes.step", 11),
    ("plate_2holes_photo_2.jpg",     "plate_2holes.step", 12),
    ("plate_3holes_photo.jpg",       "plate_3holes.step", 21),
    ("flange_500mm_photo.jpg",       "flange_500mm.step", 31),   # the 10x larger object
    ("flange_photo_2.jpg",           "flange_50mm.step", 32),
    ("l_bracket_photo.jpg",          "l_bracket.stl", 41),
    ("t_bracket_photo.jpg",          "t_bracket.obj", 51),
    ("hex_nut_photo.jpg",            "hex_nut.ply", 61),
    ("spacer_tube_photo.jpg",        "spacer_tube.stl", 71),
    ("sprocket_photo.jpg",           "sprocket.stp", 81),
]

# Which library design each photo truly shows (the scaled flange is the same design as flange_50mm).
TRUE_DESIGN = {"flange_500mm.step": "flange_50mm.step"}


def build_photos(cad: dict[str, Path], out_dir: Path, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, trimesh.Trimesh] = {}
    rows = []
    for photo, src, seed in PHOTOS:
        p = out_dir / photo
        if force or not p.exists():
            if src not in meshes:
                from cad_processing.loader import load_cad
                meshes[src] = load_cad(cad[src])
            img = synth_photo(meshes[src], seed)
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({"photo": photo, "design": TRUE_DESIGN.get(src, src)})
    labels = out_dir.parent / "labels.csv"
    with open(labels, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo", "design"])
        writer.writeheader()
        writer.writerows(rows)
    return labels


def ensure_samples(root: Path = SAMPLES_DIR, force: bool = False) -> dict:
    cad = build_cad(root / "cad", force)
    labels = build_photos(cad, root / "photos", force)
    return {"cad_dir": root / "cad", "photos_dir": root / "photos", "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    parser.add_argument("--out", type=Path, default=SAMPLES_DIR)
    args = parser.parse_args()
    info = ensure_samples(args.out, args.force)
    print(f"CAD files: {info['cad_dir']}\nPhotos:    {info['photos_dir']}\nLabels:    {info['labels']}")


if __name__ == "__main__":
    main()
