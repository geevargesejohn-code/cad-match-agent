"""Command-line access to the CAD library (handy before the UI exists).

    python -m scripts.library_cli add part1.stl part2.step folder_of_cad_files/
    python -m scripts.library_cli list
    python -m scripts.library_cli rename <id> "New name"
    python -m scripts.library_cli delete <id>
    python -m scripts.library_cli reindex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cad_processing.loader import supported_formats
from matching import indexer
from storage import library


def _expand(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    exts = supported_formats()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
        else:
            files.append(p)
    return files


def cmd_add(args) -> int:
    files = _expand(args.paths)
    if not files:
        print("No CAD files found.")
        return 1
    failures = 0
    for r in library.add_many(files):
        if not r.ok:
            failures += 1
            print(f"  FAILED     {r.filename}: {r.error}")
        elif r.duplicate:
            print(f"  DUPLICATE  {r.filename} (already in library as {r.item['id']})")
        else:
            print(f"  ADDED      {r.filename} -> {r.item['id']}  proportions={r.item['proportions']}")
    return 1 if failures else 0


def cmd_list(_args) -> int:
    items = library.list_items()
    if not items:
        print("Library is empty.")
        return 0
    print(f"{'ID':<14}{'NAME':<28}{'FMT':<6}{'FACES':>9}  {'PROPORTIONS':<24}STATUS")
    for it in items:
        props = ", ".join(f"{x:.2f}" for x in it["proportions"] or [])
        print(f"{it['id']:<14}{it['name'][:27]:<28}{it['format']:<6}{it['n_faces']:>9}  {props:<24}{it['status']}")
    return 0


def cmd_reindex(_args) -> int:
    n = indexer.reindex_all()
    print(f"Re-indexed {n} model(s).")
    return 0


def cmd_rename(args) -> int:
    item = library.rename(args.id, args.name)
    print(f"Renamed {item['id']} -> {item['name']}")
    return 0


def cmd_delete(args) -> int:
    print("Deleted." if library.delete(args.id) else "No such item.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the CAD library")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="add CAD files or folders")
    p_add.add_argument("paths", nargs="+")
    p_add.set_defaults(func=cmd_add)

    sub.add_parser("list", help="list library items").set_defaults(func=cmd_list)

    p_ren = sub.add_parser("rename", help="rename an item")
    p_ren.add_argument("id")
    p_ren.add_argument("name")
    p_ren.set_defaults(func=cmd_rename)

    sub.add_parser("reindex", help="re-render all models (after changing view count or embedder)"
                   ).set_defaults(func=cmd_reindex)

    p_del = sub.add_parser("delete", help="delete an item")
    p_del.add_argument("id")
    p_del.set_defaults(func=cmd_delete)

    args = parser.parse_args(argv)
    indexer.install()  # new files get rendered and indexed right away
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
