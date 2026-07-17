"""`pyscf-cli convert` — convert structure files (SDF) to XYZ.

Port of the course's sdf2xyz.py: enables the PubChem workflow
(search molecule -> download 3D SDF -> convert -> calculate) without
leaving the CLI.  Keeps the course naming convention SDF_name.sdf ->
XYZ_name.xyz when no output name is given.
"""

from __future__ import annotations

import os

from . import core
from .core import InputError


def register(subparsers):
    parser = subparsers.add_parser(
        "convert",
        help="convert an SDF structure file (e.g. from PubChem) to XYZ",
        description=(
            "Convert a V2000 SDF file to XYZ. Typical workflow: search the "
            "molecule on PubChem, download the 3D Conformer as SDF, convert "
            "it here, then relax and calculate. SDF_name.sdf becomes "
            "XYZ_name.xyz unless -o is given."
        ),
    )
    parser.add_argument("sdf", metavar="SDF", help="input SDF file")
    parser.add_argument("-o", "--output", default=None, metavar="FILE",
                        help="output XYZ file (default: derived from the input name)")
    parser.set_defaults(func=run)
    return parser


def sdf_to_atoms(sdf_file):
    """Parse the atom block of a V2000 SDF/MOL file."""
    if not os.path.isfile(sdf_file):
        raise InputError(f"SDF file not found: '{sdf_file}'")

    with open(sdf_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise InputError(
            f"'{sdf_file}' is too short to be an SDF file "
            "(needs 3 header lines, a counts line, and atoms)."
        )

    counts_line = lines[3].split()
    try:
        natoms = int(counts_line[0])
    except (IndexError, ValueError):
        raise InputError(
            f"Could not read the atom count from line 4 of '{sdf_file}'.\n"
            "Is this a V2000 SDF/MOL file? (PubChem's '3D Conformer' "
            "SDF download has the right format.)"
        )
    if natoms <= 0:
        raise InputError(f"'{sdf_file}' declares {natoms} atoms.")

    atom_lines = lines[4:4 + natoms]
    if len(atom_lines) < natoms:
        raise InputError(
            f"'{sdf_file}' declares {natoms} atoms but the atom block is shorter."
        )

    atoms = []
    for i, line in enumerate(atom_lines, start=5):
        fields = line.split()
        if len(fields) < 4:
            raise InputError(f"Malformed atom line {i} in '{sdf_file}': {line.rstrip()!r}")
        try:
            x, y, z = float(fields[0]), float(fields[1]), float(fields[2])
        except ValueError:
            raise InputError(f"Malformed coordinates on line {i} of '{sdf_file}'.")
        atoms.append((fields[3], x, y, z))
    return atoms


def default_xyz_name(sdf_file):
    root, _ = os.path.splitext(sdf_file)
    directory = os.path.dirname(root)
    base = os.path.basename(root)
    if base.startswith("SDF_"):
        base = "XYZ_" + base[4:]
    return os.path.join(directory, base + ".xyz")


def run(args):
    atoms = sdf_to_atoms(args.sdf)
    xyz_file = args.output or default_xyz_name(args.sdf)
    core.write_xyz(
        xyz_file, atoms,
        comment=f"Converted from {os.path.basename(args.sdf)} (Angstrom)",
    )
    print(f"Converted: {args.sdf} -> {xyz_file}  ({len(atoms)} atoms)")
    print(f"Next: pyscf-cli relax {xyz_file} --basis 6-31g")
    return 0
