"""`pyscf-cli orbitals` — export MOs as Gaussian cube files (VESTA-ready).

Port of calc_pyscf_wf.py.
"""

from __future__ import annotations

import os

import numpy as np

from . import core
from .core import HARTREE_TO_EV, InputError
from .output import Report


def register(subparsers):
    parser = subparsers.add_parser(
        "orbitals",
        help="export molecular orbitals as cube files for VESTA etc.",
        description=(
            "Run an SCF calculation and write selected molecular orbitals "
            "as Gaussian cube files, ready for visualization in VESTA, "
            "Avogadro, or similar viewers."
        ),
    )
    core.add_common_arguments(parser, theories=())
    parser.add_argument("--mo", type=int, nargs="+", default=None,
                        metavar="N",
                        help="1-based MO indices to export, e.g. --mo 3 4 5")
    parser.add_argument("--homo", action="store_true", help="export the HOMO")
    parser.add_argument("--lumo", action="store_true", help="export the LUMO")
    parser.add_argument("--spin-channel", default="auto",
                        choices=["auto", "restricted", "alpha", "beta", "all"],
                        help="MO spin channel to export (default: auto)")
    parser.add_argument("--nx", type=int, default=80,
                        help="grid points along x (default: 80)")
    parser.add_argument("--ny", type=int, default=80,
                        help="grid points along y (default: 80)")
    parser.add_argument("--nz", type=int, default=80,
                        help="grid points along z (default: 80)")
    parser.add_argument("--resolution", type=float, default=None,
                        help="cube-grid resolution in Bohr; overrides --nx/--ny/--nz")
    parser.add_argument("--margin", type=float, default=3.0,
                        help="cube box margin in Bohr (default: 3.0)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="directory for cube files (default: <input>_orbitals)")
    parser.add_argument("--prefix", default="",
                        help="filename prefix inside the output directory")
    parser.set_defaults(func=run)
    return parser


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
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)


def select_channels(channels, spin_choice):
    if spin_choice in ("auto", "all"):
        return channels
    selected = [c for c in channels if c["label"] == spin_choice]
    if not selected:
        available = ", ".join(c["label"] for c in channels)
        raise InputError(
            f"Spin channel '{spin_choice}' is not available. "
            f"This calculation has: {available}"
        )
    return selected


def resolve_mo_targets(channel, mo_list, homo, lumo):
    nmo = channel["coeff"].shape[1]
    selected = {}

    if mo_list:
        for mo in mo_list:
            if mo < 1 or mo > nmo:
                raise InputError(
                    f"MO index {mo} is out of range for the {channel['label']} "
                    f"channel (valid: 1..{nmo})"
                )
            selected.setdefault(mo, set()).add("MO")

    occ = channel["occ"]
    if occ is not None and (homo or lumo):
        occupied = np.where(occ > 1.0e-8)[0]
        virtual = np.where(occ <= 1.0e-8)[0]

        if homo:
            if occupied.size == 0:
                raise InputError(
                    f"No occupied orbitals found in the {channel['label']} channel"
                )
            selected.setdefault(int(occupied[-1]) + 1, set()).add("HOMO")

        if lumo:
            if virtual.size == 0:
                raise InputError(
                    f"No virtual orbitals found in the {channel['label']} channel"
                )
            selected.setdefault(int(virtual[0]) + 1, set()).add("LUMO")

    if not selected:
        raise InputError(
            "Specify at least one target orbital with --mo and/or --homo/--lumo.\n"
            "Example: pyscf-cli orbitals h2o.xyz --homo --lumo"
        )

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


def generate_orbital_cubes(mf, args):
    from pyscf.tools import cubegen

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
            occ = None if channel["occ"] is None else channel["occ"][mo - 1]

            spin_suffix = (
                "" if channel["label"] == "restricted"
                else f"_{sanitize_label(channel['label'])}"
            )
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
                        "energy_ha": float(energy_ha),
                        "energy_ev": float(energy_ha * HARTREE_TO_EV),
                        "occ": None if occ is None else float(occ),
                        "file": outfile,
                    }
                )

    return generated, channels


def run(args):
    core.finalize_common_args(args)

    if args.output_dir is None:
        args.output_dir = core.output_stem(args, "orbitals")
    if args.resolution is not None and args.resolution <= 0.0:
        raise InputError("--resolution must be positive")
    if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
        raise InputError("--nx, --ny, and --nz must be positive integers")
    if args.margin < 0.0:
        raise InputError("--margin must be non-negative")

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf = core.build_mf(mol, args.method)
    core.run_scf(mf)

    generated, channels = generate_orbital_cubes(mf, args)

    r = Report("PySCF Orbital Cube Generator (pyscf-cli orbitals)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Method", args.method.upper(), key="method")
    r.kv("Basis", args.basis, key="basis")
    r.kv("Charge", args.charge, key="charge")
    r.kv("Spin (2S)", args.spin, key="spin")
    r.kv("Spin channel", args.spin_channel, key="spin_channel")
    if args.resolution is not None:
        r.kv("Grid mode", f"resolution = {args.resolution:.4f} Bohr")
    else:
        r.kv("Grid mode", f"nx, ny, nz = {args.nx}, {args.ny}, {args.nz}")
    r.kv("Margin", f"{args.margin:.3f} Bohr")
    r.kv("Output dir", args.output_dir, key="output_dir")
    r.rule()
    r.kv("Available spin", ", ".join(c["label"] for c in channels))
    r.line()
    r.line("Generated cube files:")
    r.line("spin         MO    energy(Ha)    energy(eV)    occ    file")
    r.line("-" * 74)
    for row in generated:
        occ_text = "NA" if row["occ"] is None else f"{row['occ']:.3g}"
        r.line(
            f"{row['spin']:<10s} {row['mo']:3d} "
            f"{row['energy_ha']:12.6f} {row['energy_ev']:12.4f} "
            f"{occ_text:>6s}  {row['file']}"
        )
    r.add("cubes", generated)
    r.rule()
    r.line("VESTA tip: open each .cube directly (File > Open); the cube already")
    r.line("contains the molecular geometry. If an extra x,y,z compass appears,")
    r.line("a second Molecule-type phase is loaded in the same window - remove it")
    r.line("via Edit > Edit Data > Phase... > select it > Delete.")
    r.rule("=")
    r.emit(json_target=args.json)
    return 0
