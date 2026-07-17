"""`pyscf-cli info` — list available basis sets, XC functionals, versions."""

from __future__ import annotations

from . import __version__
from .core import BASIS_SETS, THEORY_CHOICES, XC_FUNCTIONALS


def register(subparsers):
    parser = subparsers.add_parser(
        "info",
        help="list common basis sets, XC functionals, and version info",
        description=(
            "Show the curated lists used by pyscf-cli. Other PySCF-supported "
            "basis sets and functionals are accepted too; these are the "
            "recommended starting points for coursework."
        ),
    )
    parser.add_argument(
        "topic",
        nargs="?",
        choices=["basis", "xc", "theory", "version"],
        default=None,
        help="what to list (default: everything)",
    )
    parser.set_defaults(func=run)
    return parser


def _print_table(title, table):
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(name) for name in table)
    for name, note in table.items():
        print(f"  {name:<{width}s}  {note}")


def run(args):
    topic = args.topic

    if topic in (None, "version"):
        import pyscf

        print(f"pyscf-cli {__version__} (PySCF {pyscf.__version__})")

    if topic in (None, "basis"):
        _print_table("Common basis sets (--basis)", BASIS_SETS)

    if topic in (None, "xc"):
        _print_table("Common XC functionals (--xc, with --theory dft)", XC_FUNCTIONALS)

    if topic in (None, "theory"):
        print("\nLevels of theory (--theory)")
        print("---------------------------")
        notes = {
            "scf": "Hartree-Fock (RHF/UHF/ROHF via --method)",
            "dft": "Kohn-Sham DFT (choose functional with --xc)",
            "mp2": "2nd-order Moller-Plesset perturbation theory",
            "ccsd": "coupled cluster singles and doubles",
            "ccsd_t": "CCSD with perturbative triples — 'gold standard', expensive",
        }
        for name in THEORY_CHOICES:
            print(f"  {name:<7s}  {notes[name]}")

    return 0
