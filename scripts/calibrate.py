"""Measure how well the matcher separates true matches from wrong designs, and
suggest thresholds (and optionally weights) for config.yaml.

    python -m scripts.calibrate                       # bundled samples
    python -m scripts.calibrate --extra 3             # + 3 extra random synthetic photos per design
    python -m scripts.calibrate --cad my_cad/ --photos my_photos/ --labels my_labels.csv
    python -m scripts.calibrate --optimize            # also search for better weights

labels.csv has two columns:  photo,design   (design = CAD file name in --cad)
It runs in a temporary data folder, so your real library is never touched.
"""
from __future__ import annotations

import argparse
import csv
import zlib
import os
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cad", type=Path)
    ap.add_argument("--photos", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--extra", type=int, default=0, help="extra synthetic photos per sample design")
    ap.add_argument("--optimize", action="store_true", help="search weights that maximize separation")
    ap.add_argument("--save-rows", type=Path, help="save raw scores to JSON for fast re-analysis")
    ap.add_argument("--rows", type=Path, help="re-analyze saved raw scores with the current config")
    args = ap.parse_args()

    if args.rows:
        import json
        from matching import fusion
        rows = [tuple(r) for r in json.load(open(args.rows))]
        rows = [(a, b, c, raw, rescore(raw)) for a, b, c, raw, _ in rows]
        report(rows, fusion)
        if args.optimize:
            optimize(rows, fusion)
        return

    tmp = tempfile.mkdtemp(prefix="cadmatch_calib_")
    os.environ["CAD_MATCH_DATA_DIR"] = tmp

    from core.config import reset_config_cache
    reset_config_cache()

    import cv2
    from cad_processing.loader import load_cad
    from matching import fusion, indexer
    from matching.features import compute_photo_features
    from matching.photo import analyze_photo
    from matching.scorer import score_photo
    from scripts import make_samples
    from storage import library

    if args.cad is None:
        info = make_samples.ensure_samples()
        cad_dir, photos_dir, labels = info["cad_dir"], info["photos_dir"], info["labels"]
        designs = list(make_samples.DESIGNS)
    else:
        cad_dir, photos_dir, labels = args.cad, args.photos, args.labels
        designs = sorted({r["design"] for r in csv.DictReader(open(labels))})
        designs += [p.name for p in cad_dir.iterdir() if p.name not in designs and p.suffix.lower() in
                    {".stl", ".obj", ".ply", ".step", ".stp"}]

    indexer.install()
    ids = {}
    print(f"Indexing {len(designs)} CAD models ...")
    for name in designs:
        r = library.add_file(cad_dir / name)
        if not r.ok:
            raise SystemExit(f"Could not add {name}: {r.error}")
        ids[name] = r.item["id"]
    views = {name: indexer.load_features(i) for name, i in ids.items()}

    pairs = [(photos_dir / r["photo"], r["design"]) for r in csv.DictReader(open(labels))]
    if args.extra and args.cad is None:
        extra_dir = Path(tmp) / "extra"
        extra_dir.mkdir()
        for name in designs:
            mesh = load_cad(cad_dir / name)
            for k in range(args.extra):
                p = extra_dir / f"{Path(name).stem}_x{k}.jpg"
                img = make_samples.synth_photo(mesh, seed=1000 + 37 * k + zlib.crc32(name.encode()) % 997)
                cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                pairs.append((p, name))

    print(f"Scoring {len(pairs)} photos against {len(designs)} models ...\n")
    rows = []  # (photo, design, candidate, raw dict, total)
    for photo_path, true_design in pairs:
        pf = compute_photo_features(analyze_photo(photo_path))
        for cand, v in views.items():
            s = score_photo(pf, v)
            rows.append((photo_path.name, true_design, cand, s.raw, s.score))

    if args.save_rows:
        import json
        json.dump(rows, open(args.save_rows, "w"))
    report(rows, fusion)
    if args.optimize:
        optimize(rows, fusion)
    print(f"\n(temporary data in {tmp})")


def rescore(raw: dict) -> float:
    from matching import fusion
    from matching.embeddings import get_embedder
    norm = {k: (fusion.normalize(k, v, get_embedder().name) if k == "embedding" else fusion.normalize(k, v))
            for k, v in raw.items()}
    return float(fusion.combine(norm))


def report(rows, fusion) -> None:
    hi, lo = fusion.thresholds()
    by_photo: dict[str, list] = {}
    for r in rows:
        by_photo.setdefault(r[0], []).append(r)

    print(f"{'PHOTO':<30}{'TRUE SCORE':>11}  {'BEST WRONG DESIGN':<22}{'SCORE':>6}  {'MARGIN':>7}  RANK")
    true_scores, wrong_scores, top1 = [], [], 0
    for photo, rs in by_photo.items():
        true = next(r for r in rs if r[2] == r[1])
        wrong = [r for r in rs if r[2] != r[1]]
        best_wrong = max(wrong, key=lambda r: r[4])
        rank = 1 + sum(r[4] > true[4] for r in wrong)
        top1 += rank == 1
        true_scores.append(true[4])
        wrong_scores += [r[4] for r in wrong]
        print(f"{photo:<30}{true[4]:>11.3f}  {best_wrong[2]:<22}{best_wrong[4]:>6.3f}  {true[4]-best_wrong[4]:>+7.3f}  {rank}")

    print("\nRaw signals at each pair's best pose (true pairs vs wrong designs):")
    for sig in rows[0][3]:
        tv = np.array([r[3][sig] for r in rows if r[1] == r[2]])
        wv = np.array([r[3][sig] for r in rows if r[1] != r[2]])
        print(f"  {sig:<12} true: p10 {np.percentile(tv,10):.3f} median {np.median(tv):.3f} | "
              f"wrong: median {np.median(wv):.3f} p90 {np.percentile(wv,90):.3f}")

    t, w = np.array(true_scores), np.array(wrong_scores)
    print(f"\nLibrary search top-1 accuracy: {top1}/{len(by_photo)}")
    print(f"True-match scores : min {t.min():.3f}  mean {t.mean():.3f}")
    print(f"Wrong-design scores: max {w.max():.3f}  95th pct {np.percentile(w, 95):.3f}  mean {w.mean():.3f}")

    tp = (t >= hi).mean()
    tn = (w <= lo).mean()
    unc = ((t < hi) & (t > lo)).sum() + ((w < hi) & (w > lo)).sum()
    fp = (w >= hi).sum()
    print(f"\nWith current thresholds (match {hi:.2f} / no-match {lo:.2f}):")
    print(f"  true matches called MATCH       : {100*tp:.0f}%")
    print(f"  wrong designs called NOT A MATCH: {100*tn:.0f}%")
    print(f"  false MATCHes                   : {fp}")
    print(f"  uncertain (agent investigates)  : {unc}")

    if t.min() > w.max():
        mid = (t.min() + w.max()) / 2
        gap = t.min() - w.max()
        print(f"\nScores separate cleanly. Suggested: match_threshold {mid + gap/4:.2f}, "
              f"no_match_threshold {mid - gap/4:.2f}")
    else:
        cand = np.linspace(0.3, 0.95, 131)
        acc = [((t >= c).mean() + (w < c).mean()) / 2 for c in cand]
        c = cand[int(np.argmax(acc))]
        print(f"\nScores overlap. Best single cut {c:.2f} (balanced accuracy {max(acc):.1%}). "
              f"Suggested: match_threshold {c + 0.04:.2f}, no_match_threshold {c - 0.04:.2f}")


def _normalized_matrix(rows, fusion):
    from matching.embeddings import get_embedder
    names = list(rows[0][3].keys())
    be = get_embedder().name
    cols = []
    for n in names:
        raw = np.array([r[3][n] for r in rows])
        cols.append(fusion.normalize(n, raw, be) if n == "embedding" else fusion.normalize(n, raw))
    return names, np.stack(cols, axis=1)


def optimize(rows, fusion, samples: int = 6000) -> None:
    """Random search over weights. Objective: verification accuracy (true matches above
    the cut, wrong designs below it), then search top-1 accuracy, then worst margin."""
    names, norm = _normalized_matrix(rows, fusion)
    is_true = np.array([r[1] == r[2] for r in rows])
    photos = np.array([r[0] for r in rows])
    groups = [((photos == p) & is_true, (photos == p) & ~is_true) for p in np.unique(photos)]
    cuts = np.linspace(0.3, 0.9, 61)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(samples):
        w = rng.dirichlet(np.ones(len(names)))
        if "hu" in names:
            w[names.index("hu")] *= rng.choice([0, 1])
        w = w / w.sum()
        s = norm @ w
        t, f = s[is_true], s[~is_true]
        bal = max(((t >= c).mean() + (f < c).mean()) / 2 for c in cuts)
        margins = [s[a].max() - s[b].max() for a, b in groups]
        top1 = sum(m > 0 for m in margins)
        key = (round(bal, 3), top1, min(margins))
        if best is None or key > best[0]:
            best = (key, w)
    (bal, top1, margin), w = best
    print("\nBest weights found: " + ", ".join(f"{n} {x:.2f}" for n, x in zip(names, w)))
    print(f"  balanced accuracy {bal:.1%}, search top-1 {top1}/{len(groups)}, worst margin {margin:+.3f}")
    if len(groups) < 20:
        print(f"  Note: only {len(groups)} photos. Weights fitted to so few photos overfit; prefer the "
              "default weights unless you have 20+ labeled photos covering several designs.")


if __name__ == "__main__":
    main()
