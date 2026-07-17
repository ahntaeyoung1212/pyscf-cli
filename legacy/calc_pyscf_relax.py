#!/usr/bin/env python3

__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Tokyo Univ of Sci, Mochizuki group"
__version__ = "1.0"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__date__ = "May 14th, 2026"

import argparse
import io
import os
from pyscf import gto, scf, dft, mp, cc
from pyscf.geomopt.geometric_solver import optimize

HARTREE_TO_EV = 27.211386
THEORY_ALIASES = {"scf", "dft", "mp2", "ccsd", "ccsd_t"}


def read_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atom_lines = lines[2:2 + natoms]

    atoms = []
    for line in atom_lines:
        elem, x, y, z = line.split()
        atoms.append((elem, float(x), float(y), float(z)))

    return atoms


def write_xyz(filename, atoms, comment=""):
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for e, x, y, z in atoms:
            f.write(f"{e:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def atoms_to_pyscf(atoms):
    return "; ".join([f"{e} {x} {y} {z}" for e, x, y, z in atoms])


def default_txt_name(xyz_file):
    root, _ = os.path.splitext(os.path.basename(xyz_file))
    return f"relax_{root}.txt"


def build_output_text(args, method, out_xyz, energy_ha, energy_ev):
    out = io.StringIO()
    print("==========================================", file=out)
    print(" PySCF Geometry Optimization", file=out)
    print("==========================================", file=out)
    print(f"XYZ file      : {args.xyz}", file=out)
    print(f"Basis         : {args.basis}", file=out)
    print(f"Method        : {method}", file=out)
    print(f"Theory        : {args.theory}", file=out)
    if args.theory == "dft":
        print(f"XC functional : {args.xc}", file=out)
    print(f"Charge        : {args.charge}", file=out)
    print(f"Spin (2S)     : {args.spin}", file=out)
    print(f"Text output   : {args.txt}", file=out)
    print("------------------------------------------", file=out)
    print("Optimization finished.", file=out)
    print(f"Final energy  : {energy_ev:.6f} eV", file=out)
    #print(f"              : {energy_ha:.10f} Hartree", file=out)
    print(f"Output XYZ    : {out_xyz}", file=out)
    print("==========================================", file=out)
    return out.getvalue()


def build_reference(mol, args):
    if args.theory == "dft":
        if args.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = args.xc
        method = "DFT"
        return mf, method

    if args.spin == 0:
        return scf.RHF(mol), "RHF"
    return scf.UHF(mol), "UHF"


def correlated_energy(mf_ref, theory):
    if theory in ("scf", "dft"):
        return mf_ref.e_tot
    if theory == "mp2":
        post = mp.MP2(mf_ref)
        post.verbose = 0
        post.kernel()
        return post.e_tot
    if theory == "ccsd":
        post = cc.CCSD(mf_ref)
        post.verbose = 0
        post.kernel()
        return mf_ref.e_tot + post.e_corr
    if theory == "ccsd_t":
        post = cc.CCSD(mf_ref)
        post.verbose = 0
        post.kernel()
        return mf_ref.e_tot + post.e_corr + post.ccsd_t()
    raise RuntimeError(f"Unknown theory: {theory}")


def main():
    parser = argparse.ArgumentParser(
        description="PySCF geometry optimization from XYZ file"
    )
    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument("--basis", default="sto-3g", help="Basis set")
    parser.add_argument("--theory", default="scf",
                        choices=["scf", "dft", "mp2", "ccsd", "ccsd_t"],
                        help="Electronic-structure level")
    parser.add_argument("--xc", default="b3lyp", type=str.lower,
                        help="XC functional for DFT")
    parser.add_argument("--spin", type=int, default=0, help="Spin = 2S")
    parser.add_argument("--charge", type=int, default=0, help="Total charge")
    parser.add_argument("--unit", default="Angstrom",
                        choices=["Angstrom", "Bohr"],
                        help="XYZ coordinate unit")
    parser.add_argument("--txt", default=None, help="Output text summary file")

    args = parser.parse_args()
    if args.txt is None:
        args.txt = default_txt_name(args.xyz)
    if args.basis in THEORY_ALIASES:
        if args.theory == "scf":
            args.theory = args.basis
        args.basis = "cc-pvdz"
        print(f"Note          : interpreted '--basis {args.theory}' as theory; basis set set to cc-pvdz")

    atoms = read_xyz(args.xyz)

    mol = gto.M(
        atom=atoms_to_pyscf(atoms),
        basis=args.basis,
        spin=args.spin,
        charge=args.charge,
        unit=args.unit
    )

    mf, method = build_reference(mol, args)

    mf.verbose = 0

    # ---- geometry optimization ----
    mol_opt = optimize(mf)

    # ---- final single-point calculation ----
    mf_final, _ = build_reference(mol_opt, args)

    mf_final.verbose = 0
    mf_final.kernel()

    E_Ha = correlated_energy(mf_final, args.theory)
    E_eV = E_Ha * HARTREE_TO_EV

    coords = mol_opt.atom_coords(unit="Angstrom")
    symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
    atoms_xyz = [(symbols[i], *coords[i]) for i in range(len(symbols))]

    out_xyz = os.path.splitext(args.xyz)[0] + "-finish.xyz"
    write_xyz(
        out_xyz,
        atoms_xyz,
        comment=f"Optimized geometry, E = {E_Ha:.10f} Ha"
    )
    output_text = build_output_text(args, method, out_xyz, E_Ha, E_eV)
    with open(args.txt, "w") as f:
        f.write(output_text)
    print(output_text, end="")


if __name__ == "__main__":
    main()
