"""`pyscf-cli examples` — copy bundled sample molecules to the current directory.

Removes the classic first hurdle of "I don't have an input file": students
type `pyscf-cli examples h2o` and immediately have something to calculate.
"""

from __future__ import annotations

import os
from importlib import resources

from .core import InputError


def register(subparsers):
    parser = subparsers.add_parser(
        "examples",
        help="list or copy bundled sample XYZ files",
        description=(
            "Without arguments, list the bundled sample molecules. "
            "With names (or 'all'), copy them into the current directory."
        ),
    )
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help="sample names to copy (e.g. h2o), or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files")
    parser.set_defaults(func=run)
    return parser


def _bundled():
    """Return [(name, filename, description, traversable), ...]."""
    data_dir = resources.files("pyscf_cli").joinpath("data")
    items = []
    for entry in sorted(data_dir.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".xyz"):
            continue
        lines = entry.read_text(encoding="utf-8").splitlines()
        description = lines[1].strip() if len(lines) > 1 else ""
        items.append((entry.name[:-4], entry.name, description, entry))
    return items


def run(args):
    bundled = _bundled()

    if not args.names:
        print("Bundled sample molecules:\n")
        width = max(len(name) for name, *_ in bundled)
        for name, _filename, description, _entry in bundled:
            print(f"  {name:<{width}s}  {description}")
        print(
            "\nCopy one into the current directory with:\n"
            "  pyscf-cli examples h2o\n"
            "then run, e.g.:\n"
            "  pyscf-cli energy h2o.xyz --basis 6-31g"
        )
        return 0

    by_name = {name: (filename, entry) for name, filename, _d, entry in bundled}
    if args.names == ["all"]:
        wanted = list(by_name)
    else:
        wanted = args.names
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise InputError(
                f"Unknown sample name(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(by_name)} (or 'all')"
            )

    for name in wanted:
        filename, entry = by_name[name]
        if os.path.exists(filename) and not args.force:
            print(f"skipped {filename} (already exists; use --force to overwrite)")
            continue
        with open(filename, "w", encoding="utf-8") as f:
            f.write(entry.read_text(encoding="utf-8"))
        print(f"wrote {filename}")

    return 0
