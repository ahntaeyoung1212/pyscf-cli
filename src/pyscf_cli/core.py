"""Shared infrastructure for all pyscf-cli subcommands.

This module centralizes everything the original ``calc_pyscf*.py`` scripts
duplicated: XYZ parsing, molecule construction, SCF/DFT/post-HF drivers,
common command-line arguments, and physical constants.

Design rules
------------
* Error messages are written for students: say what went wrong AND how to
  fix it.  Raise :class:`InputError` for anything caused by user input so
  ``main.py`` can print it cleanly (no PySCF tracebacks) and exit with
  :data:`EXIT_INPUT_ERROR`.
* Every subcommand gets its options from :func:`add_common_arguments` /
  :func:`finalize_common_args` so flags behave identically everywhere.
"""

from __future__ import annotations

import difflib
import os
import sys

from pyscf import cc, dft, gto, mp, scf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: CODATA 2018 value, used consistently by every subcommand.
HARTREE_TO_EV = 27.211386245988
CM1_TO_EV = 1.239841984e-4
HARTREE_TO_CM1 = 219474.6313705
BOHR_TO_ANG = 0.529177210903

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_NOT_CONVERGED = 3

# ---------------------------------------------------------------------------
# Curated basis sets / XC functionals (with one-line notes for `info`)
# ---------------------------------------------------------------------------

BASIS_SETS = {
    "sto-3g": "Minimal basis. Fastest; qualitative results only. Good first try.",
    "3-21g": "Small split-valence basis. Quick but crude.",
    "6-31g": "Split-valence double-zeta. A standard teaching basis.",
    "6-31g*": "6-31G + d polarization on non-H atoms.",
    "6-31g**": "6-31G + polarization on all atoms. Good default for organics.",
    "6-31+g": "6-31G + diffuse functions (anions, weak interactions).",
    "6-31+g*": "Diffuse + polarization on non-H atoms.",
    "6-31+g**": "Diffuse + polarization on all atoms.",
    "6-311g": "Triple-zeta valence.",
    "6-311g*": "Triple-zeta + polarization on non-H atoms.",
    "6-311g**": "Triple-zeta + polarization on all atoms.",
    "6-311+g": "Triple-zeta + diffuse.",
    "6-311+g*": "Triple-zeta + diffuse + polarization (non-H).",
    "6-311+g**": "Triple-zeta + diffuse + polarization (all atoms).",
    "cc-pvdz": "Correlation-consistent double-zeta. Standard with MP2/CCSD.",
    "cc-pvtz": "Correlation-consistent triple-zeta. Accurate, slower.",
    "cc-pvqz": "Correlation-consistent quadruple-zeta. Expensive.",
    "aug-cc-pvdz": "cc-pVDZ + diffuse functions.",
    "aug-cc-pvtz": "cc-pVTZ + diffuse functions.",
    "aug-cc-pvqz": "cc-pVQZ + diffuse functions. Very expensive.",
    "def2-svp": "Karlsruhe double-zeta + polarization. Good all-rounder.",
    "def2-tzvp": "Karlsruhe triple-zeta + polarization.",
    "def2-tzvpp": "Karlsruhe triple-zeta, doubly polarized.",
    "def2-qzvp": "Karlsruhe quadruple-zeta. Near basis-set limit; expensive.",
}
COMMON_BASIS_CHOICES = list(BASIS_SETS)

XC_FUNCTIONALS = {
    "lda,vwn": "Local density approximation. Historic baseline.",
    "pbe": "GGA. Robust general-purpose functional.",
    "pbe0": "Hybrid GGA (25% exact exchange).",
    "b3lyp": "The classic hybrid. Standard teaching choice.",
    "blyp": "GGA (Becke 88 exchange + LYP correlation).",
    "bp86": "GGA (Becke 88 exchange + Perdew 86 correlation).",
    "m06": "Minnesota hybrid meta-GGA.",
    "m06-2x": "Hybrid meta-GGA. Good main-group thermochemistry.",
    "wb97x": "Range-separated hybrid.",
    "wb97x-d": "Range-separated hybrid with dispersion correction.",
}
COMMON_XC_CHOICES = list(XC_FUNCTIONALS)

THEORY_CHOICES = ["scf", "dft", "mp2", "ccsd", "ccsd_t"]
METHOD_CHOICES = ["auto", "rhf", "uhf", "rohf"]


class InputError(Exception):
    """A problem caused by user input. Printed without a traceback."""


# ---------------------------------------------------------------------------
# XYZ file handling
# ---------------------------------------------------------------------------

def read_xyz(xyz_file):
    """Read a standard XYZ file into ``[(element, x, y, z), ...]``.

    Raises :class:`InputError` with a student-friendly message on any
    formatting problem instead of an anonymous ValueError.
    """
    if not os.path.isfile(xyz_file):
        hint = ""
        try:
            nearby = sorted(
                f for f in os.listdir(os.path.dirname(xyz_file) or ".")
                if f.lower().endswith(".xyz")
            )
        except OSError:
            nearby = []
        if nearby:
            hint = "\nXYZ files found here: " + ", ".join(nearby[:8])
        raise InputError(f"XYZ file not found: '{xyz_file}'{hint}")

    with open(xyz_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise InputError(f"'{xyz_file}' is empty.")

    try:
        natoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise InputError(
            f"Line 1 of '{xyz_file}' must be the number of atoms, "
            f"but found: {lines[0].rstrip()!r}\n"
            "An XYZ file looks like:\n"
            "  3\n  water molecule (comment line)\n"
            "  O 0.000 0.000 0.117\n  H 0.000 0.757 -0.469\n  H 0.000 -0.757 -0.469"
        )

    if natoms <= 0:
        raise InputError(
            f"'{xyz_file}' declares {natoms} atoms on line 1; "
            "an XYZ file must contain at least one atom."
        )

    atom_lines = lines[2:2 + natoms]
    if len(atom_lines) < natoms:
        raise InputError(
            f"'{xyz_file}' declares {natoms} atoms on line 1 but only "
            f"{len(atom_lines)} atom lines follow the comment line."
        )

    atoms = []
    for i, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise InputError(
                f"Line {i} of '{xyz_file}' must be 'element x y z', "
                f"but found: {line.rstrip()!r}"
            )
        elem = fields[0]
        try:
            x, y, z = (float(v) for v in fields[1:4])
        except ValueError:
            raise InputError(
                f"Line {i} of '{xyz_file}': coordinates must be numbers, "
                f"but found: {line.rstrip()!r}"
            )
        atoms.append((elem, x, y, z))

    return atoms


def write_xyz(filename, atoms, comment=""):
    """Write ``[(element, x, y, z), ...]`` as a standard XYZ file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment.replace("\n", " ") + "\n")
        for e, x, y, z in atoms:
            f.write(f"{e:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def atoms_to_pyscf_string(atoms):
    return "; ".join(f"{e} {x} {y} {z}" for e, x, y, z in atoms)


# ---------------------------------------------------------------------------
# Input validation with "did you mean" suggestions
# ---------------------------------------------------------------------------

def _suggest(name, known):
    matches = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return matches[0] if matches else None


def normalize_basis(basis):
    """Validate a basis-set name.

    Known names pass through.  A likely typo raises with a suggestion.
    Anything else is forwarded to PySCF (it knows far more basis sets
    than our curated list) after a note on stderr.
    """
    basis = basis.lower()
    if basis in BASIS_SETS:
        return basis
    suggestion = _suggest(basis, COMMON_BASIS_CHOICES)
    if suggestion:
        raise InputError(
            f"Unknown basis set '{basis}'. Did you mean '{suggestion}'?\n"
            "List the common choices with: pyscf-cli info basis"
        )
    print(
        f"note: basis '{basis}' is not in the common list; "
        "passing it to PySCF as-is (see: pyscf-cli info basis)",
        file=sys.stderr,
    )
    return basis


def normalize_xc(xc):
    """Validate an XC functional name (same policy as :func:`normalize_basis`)."""
    xc = xc.lower()
    if xc in XC_FUNCTIONALS:
        return xc
    suggestion = _suggest(xc, COMMON_XC_CHOICES)
    if suggestion:
        raise InputError(
            f"Unknown XC functional '{xc}'. Did you mean '{suggestion}'?\n"
            "List the common choices with: pyscf-cli info xc"
        )
    print(
        f"note: XC functional '{xc}' is not in the common list; "
        "passing it to PySCF as-is (see: pyscf-cli info xc)",
        file=sys.stderr,
    )
    return xc


def resolve_method(method, spin):
    """Resolve ``auto`` to RHF (closed-shell) or UHF (open-shell)."""
    if method == "auto":
        return "rhf" if spin == 0 else "uhf"
    return method


# ---------------------------------------------------------------------------
# Molecule and mean-field construction
# ---------------------------------------------------------------------------

def build_mol(atoms, basis, charge=0, spin=0, unit="Angstrom"):
    """Build a PySCF Mole, translating common failures into advice."""
    try:
        return gto.M(
            atom=atoms_to_pyscf_string(atoms),
            basis=basis,
            charge=charge,
            spin=spin,
            unit=unit,
            verbose=0,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "basis" in msg.lower():
            raise InputError(
                f"PySCF could not find basis '{basis}' for this molecule.\n"
                f"PySCF said: {msg.strip()}\n"
                "List common choices with: pyscf-cli info basis"
            )
        if "spin" in msg.lower() or "electron" in msg.lower():
            raise InputError(
                f"Impossible charge/spin combination (charge={charge}, spin 2S={spin}).\n"
                "Remember: --spin is 2S, the number of UNPAIRED electrons "
                "(0 = singlet, 1 = doublet, 2 = triplet).\n"
                "Check that (total electrons) and (2S) are both even or both odd.\n"
                f"PySCF said: {msg}"
            )
        raise
    except KeyError as exc:
        raise InputError(
            f"PySCF does not know basis {exc} for one of the elements in this molecule.\n"
            "Try another basis set (see: pyscf-cli info basis)."
        )


def build_mf(mol, method):
    """Return an SCF object for rhf / uhf / rohf."""
    method = method.lower()
    if method == "rhf":
        return scf.RHF(mol)
    if method == "uhf":
        return scf.UHF(mol)
    if method == "rohf":
        return scf.ROHF(mol)
    raise InputError(f"Unknown method '{method}': choose rhf, uhf, or rohf")


def build_ks(mol, spin, xc, method="auto"):
    """Return a Kohn-Sham object honoring the requested reference type.

    rhf -> RKS (ROKS for open shells), rohf -> ROKS, uhf -> UKS,
    auto -> RKS closed-shell / UKS open-shell.
    """
    method = (method or "auto").lower()
    if method == "uhf":
        mf = dft.UKS(mol)
    elif method == "rohf":
        mf = dft.ROKS(mol)
    elif method == "rhf":
        mf = dft.RKS(mol)  # PySCF promotes this to ROKS for open shells
    else:
        mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = xc
    return mf


def build_reference(mol, theory, method, xc):
    """Return an unconverged mean-field object and its display label.

    The HF label comes from the actual object class: PySCF silently
    promotes RHF to ROHF for open shells, and the label should say so
    (the legacy scripts printed "RHF" in that case).
    """
    if theory == "dft":
        mf = build_ks(mol, mol.spin, xc, method)
        return mf, f"DFT ({xc}, {type(mf).__name__})"
    mf = build_mf(mol, method)
    return mf, type(mf).__name__


def require_hessian_capable(method, spin):
    """Fail early for references whose analytic Hessian PySCF lacks."""
    if method == "rohf" or (method == "rhf" and spin != 0):
        raise InputError(
            "Analytic Hessians for restricted open-shell references (ROHF/ROKS) "
            "are not implemented in PySCF.\n"
            "Use --method uhf for open-shell vibrational or thermochemistry "
            "calculations."
        )


def scf_exit_code(*mfs):
    """EXIT_OK if every SCF converged, else EXIT_NOT_CONVERGED."""
    if all(getattr(mf, "converged", True) for mf in mfs):
        return EXIT_OK
    return EXIT_NOT_CONVERGED


def run_scf(mf, warn=True):
    """Run the SCF, warning on stderr if it did not converge."""
    mf.verbose = 0
    mf.kernel()
    if warn and not mf.converged:
        print(
            "WARNING: SCF did not converge; results come from the last cycle.\n"
            "Hints: check --spin and --charge, try a smaller basis first, or\n"
            "start from a more reasonable geometry.",
            file=sys.stderr,
        )
    return mf


def run_theory(mol, theory, method, xc):
    """Converge the requested level of theory.

    Returns ``(mf, e_tot, info)`` where ``mf`` is the (converged) SCF/DFT
    reference and ``info`` carries the display label plus any correlation
    energy components:

    * scf/dft: ``{"label"}``
    * mp2:     ``{"label", "e_ref", "e_corr"}``
    * ccsd:    ``{"label", "e_ref", "e_corr_ccsd"}``
    * ccsd_t:  ``{"label", "e_ref", "e_corr_ccsd", "e_corr_t"}``
    """
    mf, label = build_reference(mol, theory, method, xc)
    run_scf(mf)

    if theory in ("scf", "dft"):
        return mf, mf.e_tot, {"label": label}

    if theory == "mp2":
        post = mp.MP2(mf)
        post.verbose = 0
        post.kernel()
        return mf, post.e_tot, {
            "label": f"{label}-MP2",
            "e_ref": mf.e_tot,
            "e_corr": post.e_corr,
        }

    if theory == "ccsd":
        post = cc.CCSD(mf)
        post.verbose = 0
        post.kernel()
        return mf, mf.e_tot + post.e_corr, {
            "label": f"{label}-CCSD",
            "e_ref": mf.e_tot,
            "e_corr_ccsd": post.e_corr,
        }

    if theory == "ccsd_t":
        post = cc.CCSD(mf)
        post.verbose = 0
        post.kernel()
        et = post.ccsd_t()
        return mf, mf.e_tot + post.e_corr + et, {
            "label": f"{label}-CCSD(T)",
            "e_ref": mf.e_tot,
            "e_corr_ccsd": post.e_corr,
            "e_corr_t": et,
        }

    raise InputError(f"Unknown theory '{theory}': choose from {', '.join(THEORY_CHOICES)}")


# ---------------------------------------------------------------------------
# Cost estimation (rough, for a pre-run warning)
# ---------------------------------------------------------------------------

_EXPENSIVE_THEORY = {"mp2": 1, "ccsd": 2, "ccsd_t": 3}
_BIG_BASIS_MARKERS = ("cc-pvtz", "cc-pvqz", "aug-", "def2-tzv", "def2-qzv", "6-311")


def warn_if_expensive(natoms, theory, basis):
    """Print a heads-up when a calculation may take very long in class."""
    heavy_theory = _EXPENSIVE_THEORY.get(theory, 0)
    big_basis = any(m in basis for m in _BIG_BASIS_MARKERS)
    score = heavy_theory + (1 if big_basis else 0) + (1 if natoms > 10 else 0)
    if score >= 3:
        print(
            f"note: {theory.upper()}/{basis} on {natoms} atoms may take a long time "
            "(CCSD(T) scales as N^7). Consider a smaller basis or fewer atoms for exercises.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Common command-line arguments
# ---------------------------------------------------------------------------

def add_common_arguments(
    parser,
    *,
    default_basis="sto-3g",
    theories=tuple(THEORY_CHOICES),
    include_method=True,
    include_dry_run=False,
):
    """Attach the option set shared by every calculation subcommand."""
    parser.add_argument(
        "xyz_positional",
        nargs="?",
        default=None,
        metavar="XYZ",
        help="input XYZ file",
    )
    parser.add_argument(
        "--xyz",
        dest="xyz_option",
        default=None,
        metavar="FILE",
        help="input XYZ file (alternative to the positional argument)",
    )
    parser.add_argument(
        "--basis",
        default=default_basis,
        type=str.lower,
        metavar="BASIS",
        help=f"basis set (default: {default_basis}; list: pyscf-cli info basis)",
    )
    if theories:
        parser.add_argument(
            "--theory",
            default="scf",
            choices=list(theories),
            help="electronic-structure level (default: scf)",
        )
        parser.add_argument(
            "--xc",
            default="b3lyp",
            type=str.lower,
            metavar="XC",
            help="XC functional for --theory dft (default: b3lyp; list: pyscf-cli info xc)",
        )
    if include_method:
        parser.add_argument(
            "--method",
            default="auto",
            choices=METHOD_CHOICES,
            help="SCF reference (default: auto = RHF if spin 0, else UHF)",
        )
    parser.add_argument("--spin", type=int, default=0,
                        help="spin 2S = number of unpaired electrons (default: 0)")
    parser.add_argument("--charge", type=int, default=0,
                        help="total charge (default: 0)")
    parser.add_argument("--unit", default="Angstrom", choices=["Angstrom", "Bohr"],
                        help="unit of the XYZ coordinates (default: Angstrom)")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="also write results as JSON to FILE ('-' or no value = stdout)",
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="print an equivalent PySCF Python script and exit "
                 "(see what runs under the hood, then graduate to PySCF itself)",
        )
    return parser


def finalize_common_args(args):
    """Validate/normalize the common arguments in place.

    Resolves the XYZ path (positional vs --xyz), normalizes basis/xc,
    resolves method 'auto', and reads the molecule into ``args.atoms``.
    """
    xyz = args.xyz_option or args.xyz_positional
    if xyz is None:
        raise InputError(
            "No XYZ file given.\n"
            "Usage: pyscf-cli <command> your_molecule.xyz [options]\n"
            "Get sample files with: pyscf-cli examples"
        )
    if args.xyz_option and args.xyz_positional and args.xyz_option != args.xyz_positional:
        raise InputError(
            f"Two different XYZ files given: '{args.xyz_positional}' and "
            f"'{args.xyz_option}'. Please pass only one."
        )
    args.xyz = xyz
    args.atoms = read_xyz(xyz)

    json_target = getattr(args, "json", None)
    if json_target and json_target != "-" and json_target.lower().endswith(".xyz"):
        raise InputError(
            f"--json captured '{json_target}', which looks like an input XYZ file.\n"
            "Place --json AFTER the XYZ file, or give it an explicit value:\n"
            "  pyscf-cli energy molecule.xyz --json result.json\n"
            "  pyscf-cli energy molecule.xyz --json -        (JSON to stdout)"
        )

    args.basis = normalize_basis(args.basis)
    if hasattr(args, "xc"):
        args.xc = normalize_xc(args.xc)
    if hasattr(args, "method"):
        args.method = resolve_method(args.method, args.spin)
    if hasattr(args, "theory"):
        warn_if_expensive(len(args.atoms), args.theory, args.basis)
    return args


def output_stem(args, suffix):
    """Consistent output naming: <input stem>_<suffix>."""
    root, _ = os.path.splitext(os.path.basename(args.xyz))
    return f"{root}_{suffix}"
