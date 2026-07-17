"""`pyscf-cli relax` — geometry optimization (port of calc_pyscf_relax.py).

The optimization runs on the SCF or DFT surface (geomeTRIC).  For post-HF
theories (mp2/ccsd/ccsd_t) the geometry is optimized at the reference level
and the correlated energy is a single point at the optimized structure —
exactly like the legacy script, but now stated explicitly in the output.
"""

from __future__ import annotations

import os

from pyscf.geomopt.geometric_solver import optimize

from . import core, dryrun
from .output import Report


def register(subparsers):
    parser = subparsers.add_parser(
        "relax",
        help="geometry optimization (writes an optimized XYZ file)",
        description=(
            "Optimize the molecular geometry with geomeTRIC on the SCF/DFT "
            "surface, then report the final energy and write the optimized "
            "structure as <input>_opt.xyz."
        ),
    )
    core.add_common_arguments(parser, include_dry_run=True)
    parser.add_argument("--txt", default=None, metavar="FILE",
                        help="text summary file (default: <input>_relax.txt)")
    parser.add_argument("--out-xyz", default=None, metavar="FILE",
                        help="optimized geometry file (default: <input>-finish.xyz)")
    parser.set_defaults(func=run)
    return parser


def run(args):
    core.finalize_common_args(args)

    if args.dry_run:
        print(dryrun.relax_script(args))
        return 0

    txt_file = args.txt or f"{core.output_stem(args, 'relax')}.txt"
    input_root = os.path.splitext(os.path.basename(args.xyz))[0]
    out_xyz = args.out_xyz or f"{input_root}-finish.xyz"

    ref_theory = "dft" if args.theory == "dft" else "scf"

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf, method_label = core.build_reference(mol, ref_theory, args.method, args.xc)
    mf.verbose = 0

    mol_opt = optimize(mf)

    # final single point at the requested level of theory
    mf_final, e_tot, info = core.run_theory(mol_opt, args.theory, args.method, args.xc)

    coords = mol_opt.atom_coords(unit="Angstrom")
    symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
    atoms_opt = [(symbols[i], *coords[i]) for i in range(len(symbols))]
    core.write_xyz(out_xyz, atoms_opt, comment=f"Optimized geometry, E = {e_tot:.10f} Ha")

    r = Report("PySCF Geometry Optimization (pyscf-cli relax)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Basis", args.basis, key="basis")
    r.kv("Method", method_label, key="method_label")
    r.kv("Theory", args.theory, key="theory")
    if args.theory == "dft":
        r.kv("XC functional", args.xc, key="xc")
    r.kv("Charge", args.charge, key="charge")
    r.kv("Spin (2S)", args.spin, key="spin")
    r.rule()
    r.line("Optimization finished.")
    if args.theory not in ("scf", "dft"):
        r.line(f"Note: geometry optimized on the {method_label} surface; "
               f"{info['label']} energy is a single point there.")
    r.energy("Final energy", e_tot, key="e_tot")
    r.kv("Output XYZ", out_xyz, key="out_xyz")
    r.kv("Text output", txt_file, key="txt_file")
    r.add("optimized_atoms", [
        {"element": e, "x": float(x), "y": float(y), "z": float(z)}
        for e, x, y, z in atoms_opt
    ])
    r.rule("=")
    r.emit(txt_path=txt_file, json_target=args.json)
    return 0
