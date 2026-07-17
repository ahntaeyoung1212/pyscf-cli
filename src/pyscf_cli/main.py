"""pyscf-cli entry point: subcommand dispatch.

Each subcommand lives in its own module exposing ``register(subparsers)``,
which adds its parser and sets ``func`` to its run function.  The run
function returns an exit code (or None for success).  User-input problems
are raised as :class:`pyscf_cli.core.InputError` and printed here without
a traceback.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core import EXIT_INPUT_ERROR, InputError

EPILOG = """\
examples:
  pyscf-cli info basis                      list common basis sets
  pyscf-cli energy h2o.xyz --basis 6-31g**  single-point RHF energy

pyscf-cli is an independent educational project and is NOT an official
tool of the PySCF developers. Please cite PySCF in academic work.
"""


def _subcommand_modules():
    """Import and return the subcommand modules, in help-display order.

    Imports happen here (not at module top level) so that ``--help`` stays
    fast and a broken optional dependency in one subcommand does not take
    down the whole CLI.
    """
    from . import (
        convert,
        dos,
        energy,
        examples,
        info,
        orbitals,
        relax,
        thermo,
        vib,
        vibmovie,
    )

    return [energy, relax, vib, thermo, dos, orbitals, vibmovie, convert,
            examples, info]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pyscf-cli",
        description=(
            "Run quantum chemistry calculations (PySCF) from a single XYZ file. "
            "Designed for teaching: sensible defaults, readable output, "
            "helpful error messages."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"pyscf-cli {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for module in _subcommand_modules():
        module.register(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.func(args) or 0
    except InputError as exc:
        print(f"pyscf-cli: error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except KeyboardInterrupt:
        print("\npyscf-cli: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
