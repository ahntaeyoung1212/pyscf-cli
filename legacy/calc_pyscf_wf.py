#!/usr/bin/env python3

__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Tokyo Univ of Sci, Mochizuki group"
__version__ = "0.2"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__date__ = "May 5th, 2026"

import argparse
import os

import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen

HARTREE_TO_EV = 27.211386

COMMON_BASIS_CHOICES = [
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "6-31+g",
    "6-31+g*",
    "6-31+g**",
    "6-311g",
    "6-311g*",
    "6-311g**",
    "6-311+g",
    "6-311+g*",
    "6-311+g**",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "aug-cc-pvqz",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
]


def read_xyz(xyz_file):
    with open(xyz_file, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atom_lines = lines[2 : 2 + natoms]

    atoms = []
    for line in atom_lines:
        elem, x, y, z = line.split()
        atoms.append((elem, float(x), float(y), float(z)))

    return atoms


def atoms_to_pyscf_string(atoms):
    return "; ".join([f"{e} {x} {y} {z}" for e, x, y, z in atoms])


def build_mf(mol, method):
    method = method.lower()
    if method == "rhf":
        return scf.RHF(mol)
    if method == "uhf":
        return scf.UHF(mol)
    if method == "rohf":
        return scf.ROHF(mol)
    raise ValueError("Unknown method: choose rhf, uhf, or rohf")


def run_scf(atom_str, args, method):
    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
    )

    mf = build_mf(mol, method)
    mf.verbose = 0
    mf.kernel()

    if not mf.converged:
        print("WARNING: SCF did not converge. Cube files will use the last SCF cycle.")

    return mf


def split_mo_channels(mf):
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = getattr(mf, "mo_occ", None)

    if isinstance(mo_coeff, (tuple, list)):
        labels = ("alpha", "beta")
        return [
            {
                "label": labels[i],
                "energy": np.asarray(mo_energy[i]),
                "coeff": np.asarray(mo_coeff[i]),
                "occ": None if mo_occ is None else np.asarray(mo_occ[i]),
            }
            for i in range(len(mo_coeff))
        ]

    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)

    if mo_energy.ndim == 2 and mo_coeff.ndim == 3 and mo_energy.shape[0] == 2:
        labels = ("alpha", "beta")
        return [
            {
                "label": labels[i],
                "energy": mo_energy[i],
                "coeff": mo_coeff[i],
                "occ": None if mo_occ is None else np.asarray(mo_occ[i]),
            }
            for i in range(2)
        ]

    return [
        {
            "label": "restricted",
            "energy": mo_energy,
            "coeff": mo_coeff,
            "occ": None if mo_occ is None else np.asarray(mo_occ),
        }
    ]


def sanitize_label(label):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label)


def default_output_dir(xyz_file):
    root, _ = os.path.splitext(os.path.basename(xyz_file))
    return f"WF_{root}"


def available_spins(channels):
    return [channel["label"] for channel in channels]


def select_channels(channels, spin_choice):
    if spin_choice == "auto":
        return channels

    if spin_choice == "all":
        return channels

    selected = [channel for channel in channels if channel["label"] == spin_choice]
    if not selected:
        raise ValueError(
            f"Spin channel '{spin_choice}' is not available. Choose from: {', '.join(available_spins(channels))}"
        )
    return selected


def resolve_mo_targets(channel, mo_list, homo, lumo):
    nmo = channel["coeff"].shape[1]
    selected = {}

    if mo_list:
        for mo in mo_list:
            if mo < 1 or mo > nmo:
                raise ValueError(f"MO index {mo} is out of range for {channel['label']} channel (1..{nmo})")
            selected.setdefault(mo, set()).add("MO")

    occ = channel["occ"]
    if occ is not None and (homo or lumo):
        occupied = np.where(occ > 1.0e-8)[0]
        virtual = np.where(occ <= 1.0e-8)[0]

        if homo:
            if occupied.size == 0:
                raise ValueError(f"No occupied orbitals found in {channel['label']} channel")
            selected.setdefault(int(occupied[-1]) + 1, set()).add("HOMO")

        if lumo:
            if virtual.size == 0:
                raise ValueError(f"No virtual orbitals found in {channel['label']} channel")
            selected.setdefault(int(virtual[0]) + 1, set()).add("LUMO")

    if not selected:
        raise ValueError("Specify at least one target orbital with --mo and/or --homo/--lumo")

    return [(mo, selected[mo]) for mo in sorted(selected)]


def cube_kwargs(args):
    kwargs = {"margin": args.margin}
    if args.resolution is not None:
        kwargs["resolution"] = args.resolution
    else:
        kwargs["nx"] = args.nx
        kwargs["ny"] = args.ny
        kwargs["nz"] = args.nz
    return kwargs


def generate_orbital_cubes(mf, args, method):
    channels = split_mo_channels(mf)
    selected_channels = select_channels(channels, args.spin_channel)
    os.makedirs(args.output_dir, exist_ok=True)
    common_kwargs = cube_kwargs(args)

    generated = []

    for channel in selected_channels:
        mo_targets = resolve_mo_targets(channel, args.mo, args.homo, args.lumo)

        for mo, tags in mo_targets:
            coeff = channel["coeff"][:, mo - 1]
            energy_ha = channel["energy"][mo - 1]
            energy_ev = energy_ha * HARTREE_TO_EV
            occ = None if channel["occ"] is None else channel["occ"][mo - 1]

            spin_suffix = "" if channel["label"] == "restricted" else f"_{sanitize_label(channel['label'])}"
            file_variants = []
            if "HOMO" in tags:
                file_variants.append(f"{args.prefix}MO_{mo:03d}_HOMO{spin_suffix}.cube")
            if "LUMO" in tags:
                file_variants.append(f"{args.prefix}MO_{mo:03d}_LUMO{spin_suffix}.cube")
            if "MO" in tags or not file_variants:
                file_variants.append(f"{args.prefix}MO_{mo:03d}{spin_suffix}.cube")

            for filename in file_variants:
                outfile = os.path.join(args.output_dir, filename)
                cubegen.orbital(mf.mol, outfile, coeff, **common_kwargs)
                generated.append(
                    {
                        "spin": channel["label"],
                        "mo": mo,
                        "energy_ha": energy_ha,
                        "energy_ev": energy_ev,
                        "occ": occ,
                        "file": outfile,
                    }
                )

    return generated, channels


def print_summary(mf, method, args, generated, channels):
    print("==========================================")
    print(" PySCF Wavefunction Cube Generator")
    print("==========================================")
    print(f"XYZ file      : {args.xyz}")
    print(f"Method        : {method.upper()}")
    print(f"Basis         : {args.basis}")
    print(f"Charge        : {args.charge}")
    print(f"Spin (2S)     : {args.spin}")
    print(f"Spin channel  : {args.spin_channel}")
    if args.resolution is not None:
        print(f"Grid mode     : resolution = {args.resolution:.4f} Bohr")
    else:
        print(f"Grid mode     : nx, ny, nz = {args.nx}, {args.ny}, {args.nz}")
    print(f"Margin        : {args.margin:.3f} Bohr")
    print(f"Output dir    : {args.output_dir}")
    print("------------------------------------------")

    available = ", ".join(available_spins(channels))
    print(f"Available spin: {available}")
    print("\nGenerated cube files:")
    print("spin         MO    energy(Ha)    energy(eV)    occ    file")
    print("--------------------------------------------------------------------------")

    for row in generated:
        occ_text = "NA" if row["occ"] is None else f"{row['occ']:.3g}"
        print(
            f"{row['spin']:<10s} {row['mo']:3d} "
            f"{row['energy_ha']:12.6f} {row['energy_ev']:12.4f} "
            f"{occ_text:>6s}  {row['file']}"
        )

    print("==========================================")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VESTA-readable Gaussian cube files for PySCF molecular orbitals"
    )

    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument(
        "--basis",
        default="sto-3g",
        type=str.lower,
        choices=COMMON_BASIS_CHOICES,
        help="Basis set",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "rhf", "uhf", "rohf"],
        help="SCF method",
    )
    parser.add_argument("--spin", type=int, default=0, help="Spin = 2S")
    parser.add_argument("--charge", type=int, default=0, help="Total charge")
    parser.add_argument(
        "--unit",
        default="Angstrom",
        choices=["Angstrom", "Bohr"],
        help="XYZ coordinate unit",
    )
    parser.add_argument(
        "--mo",
        type=int,
        nargs="+",
        default=None,
        help="1-based MO indices to export, e.g. --mo 33 34",
    )
    parser.add_argument("--homo", action="store_true", help="Also export HOMO")
    parser.add_argument("--lumo", action="store_true", help="Also export LUMO")
    parser.add_argument(
        "--spin-channel",
        default="auto",
        choices=["auto", "restricted", "alpha", "beta", "all"],
        help="MO spin channel to export",
    )
    parser.add_argument("--nx", type=int, default=80, help="Number of grid points along x")
    parser.add_argument("--ny", type=int, default=80, help="Number of grid points along y")
    parser.add_argument("--nz", type=int, default=80, help="Number of grid points along z")
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Cube-grid resolution in Bohr; overrides nx, ny, nz",
    )
    parser.add_argument("--margin", type=float, default=3.0, help="Cube box margin in Bohr")
    parser.add_argument("--output-dir", default=None, help="Directory for cube files")
    parser.add_argument("--prefix", default="", help="Filename prefix inside output directory")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.xyz)

    if args.resolution is not None and args.resolution <= 0.0:
        parser.error("--resolution must be positive")
    if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
        parser.error("--nx, --ny, and --nz must be positive integers")
    if args.margin < 0.0:
        parser.error("--margin must be non-negative")

    atoms = read_xyz(args.xyz)

    if args.method == "auto":
        method = "rhf" if args.spin == 0 else "uhf"
    else:
        method = args.method

    atom_str = atoms_to_pyscf_string(atoms)
    mf = run_scf(atom_str, args, method)
    generated, channels = generate_orbital_cubes(mf, args, method)
    print_summary(mf, method, args, generated, channels)


if __name__ == "__main__":
    main()
