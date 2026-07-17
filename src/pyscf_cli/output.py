"""Report formatting shared by all pyscf-cli subcommands.

Every subcommand builds a :class:`Report`: a human-readable text block
(printed to stdout and optionally saved as .txt) plus a machine-readable
dict (optionally emitted as JSON via the common ``--json`` flag).

Energies are ALWAYS shown in both Hartree and eV — keeping students aware
of unit conversions is part of the point.
"""

from __future__ import annotations

import io
import json
import sys

from .core import HARTREE_TO_EV

WIDTH = 46
_LABEL = 14  # label column width, matches the legacy scripts' look


class Report:
    """Collects a formatted text report and a parallel data dict."""

    def __init__(self, title):
        self._buf = io.StringIO()
        self.data = {"title": title}
        self.rule("=")
        self.line(f" {title}")
        self.rule("=")

    # -- text building ---------------------------------------------------

    def line(self, text=""):
        print(text, file=self._buf)

    def rule(self, char="-"):
        self.line(char * WIDTH)

    def kv(self, label, value, key=None):
        """A 'Label : value' line; optionally mirrored into the JSON data."""
        self.line(f"{label:<{_LABEL}s}: {value}")
        if key:
            self.data[key] = value

    def energy(self, label, e_hartree, key=None):
        """An energy line in both Hartree and eV."""
        self.line(
            f"{label:<{_LABEL}s}: {e_hartree: .10f} Ha  "
            f"({e_hartree * HARTREE_TO_EV: .6f} eV)"
        )
        if key:
            self.data[key] = {
                "hartree": float(e_hartree),
                "eV": float(e_hartree * HARTREE_TO_EV),
            }

    def add(self, key, value):
        """JSON-only data (tables, arrays) that has its own text formatting."""
        self.data[key] = value

    # -- emission ---------------------------------------------------------

    def text(self):
        return self._buf.getvalue()

    def emit(self, txt_path=None, json_target=None):
        """Print to stdout; optionally save .txt and/or JSON.

        ``json_target``: None (off), '-' (stdout), or a file path — wired
        directly to the common ``--json`` flag.
        """
        sys.stdout.write(self.text())
        if txt_path:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self.text())
        if json_target == "-":
            json.dump(self.data, sys.stdout, indent=2, default=_jsonable)
            sys.stdout.write("\n")
        elif json_target:
            with open(json_target, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=_jsonable)
                f.write("\n")


def use_headless_matplotlib():
    """Configure matplotlib for file output (no display, writable config dir)."""
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib


def _jsonable(obj):
    """Fallback serializer for numpy scalars/arrays."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")
