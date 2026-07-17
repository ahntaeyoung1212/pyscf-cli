#!/usr/bin/env python3

__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Tokyo Univ of Sci, Mochizuki group"
__version__ = "1.0"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__date__ = "May 15th, 2026"

import argparse
import numpy as np
from pyscf import gto, scf, dft
from pyscf.hessian import thermo

HARTREE_TO_EV = 27.211386245988

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

COMMON_XC_CHOICES = [
    "lda,vwn",
    "pbe",
    "pbe0",
    "b3lyp",
    "blyp",
    "bp86",
    "m06",
    "m06-2x",
    "wb97x",
    "wb97x-d",
]


def read_xyz(xyz_file):
    with open(xyz_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atom_lines = lines[2:2 + natoms]

    atoms = []
    for line in atom_lines:
        elem, x, y, z = line.split()
        atoms.append((elem, float(x), float(y), float(z)))
    return atoms


def atoms_to_pyscf_string(atoms):
    return "; ".join([f"{e} {x} {y} {z}" for e, x, y, z in atoms])


def build_ks(mol, spin, xc):
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc = xc
    return mf


def build_mf(mol, method):
    method = method.lower()
    if method == "rhf":
        return scf.RHF(mol)
    if method == "uhf":
        return scf.UHF(mol)
    if method == "rohf":
        return scf.ROHF(mol)
    raise ValueError("Unknown method: choose rhf, uhf, or rohf")


def eh_tuple_to_values(result, key):
    value, unit = result[key]
    if unit != "Eh":
        return value, unit, None
    return value, unit, value * HARTREE_TO_EV


def ehk_tuple_to_values(result, key):
    value, unit = result[key]
    if unit != "Eh/K":
        return value, unit, None
    return value, unit, value * HARTREE_TO_EV


def main():
    parser = argparse.ArgumentParser(
        description="PySCF thermochemistry calculator (E/H/G from harmonic analysis)"
    )
    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument(
        "--basis",
        default="6-31g",
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
    parser.add_argument(
        "--theory",
        default="scf",
        choices=["scf", "dft"],
        help="Electronic-structure level for Hessian",
    )
    parser.add_argument(
        "--xc",
        default="b3lyp",
        type=str.lower,
        choices=COMMON_XC_CHOICES,
        help="XC functional for DFT",
    )
    parser.add_argument("--spin", type=int, default=0, help="Spin = 2S")
    parser.add_argument("--charge", type=int, default=0, help="Total charge")
    parser.add_argument(
        "--unit",
        default="Angstrom",
        choices=["Angstrom", "Bohr"],
        help="XYZ coordinate unit",
    )
    parser.add_argument("--temp", type=float, default=298.15, help="Temperature in K")
    parser.add_argument("--pressure", type=float, default=101325.0, help="Pressure in Pa")

    args = parser.parse_args()

    atoms = read_xyz(args.xyz)
    atom_str = atoms_to_pyscf_string(atoms)

    if args.method == "auto":
        method = "rhf" if args.spin == 0 else "uhf"
    else:
        method = args.method

    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
    )

    if args.theory == "dft":
        mf = build_ks(mol, args.spin, args.xc)
        method_label = f"DFT ({args.xc})"
    else:
        mf = build_mf(mol, method)
        method_label = method.upper()

    mf.verbose = 0
    mf.kernel()

    hess = mf.Hessian().kernel()
    vib = thermo.harmonic_analysis(mol, hess)
    freqs_cm1 = np.array(vib["freq_wavenumber"], dtype=complex)
    positive_real = np.sum((np.abs(freqs_cm1.imag) < 1e-8) & (freqs_cm1.real > 1e-6))

    thermo_result = thermo.thermo(
        mf, vib["freq_au"], temperature=args.temp, pressure=args.pressure
    )

    print("==========================================")
    print(" PySCF Thermochemistry")
    print("==========================================")
    print(f"XYZ file      : {args.xyz}")
    print(f"Method        : {method_label}")
    print(f"Basis         : {args.basis}")
    print(f"Charge        : {args.charge}")
    print(f"Spin (2S)     : {args.spin}")
    print(f"Temperature   : {args.temp:.2f} K")
    print(f"Pressure      : {args.pressure:.1f} Pa")
    print(f"Real vib modes: {positive_real}")
    print("------------------------------------------")

    for key in ["E0", "ZPE", "E_0K", "E_tot", "H_tot", "G_tot"]:
        val, unit, val_ev = eh_tuple_to_values(thermo_result, key)
        if unit == "Eh":
            print(f"{key:12s}: {val: .10f} Eh  ({val_ev: .6f} eV)")
        else:
            print(f"{key:12s}: {val} {unit}")

    s_tot, s_unit, s_tot_evk = ehk_tuple_to_values(thermo_result, "S_tot")
    cp_tot, cp_unit, cp_tot_evk = ehk_tuple_to_values(thermo_result, "Cp_tot")
    if s_unit == "Eh/K":
        print(f"{'S_tot':12s}: {s_tot: .10e} Eh/K ({s_tot_evk: .10e} eV/K)")
    else:
        print(f"{'S_tot':12s}: {s_tot: .10e} {s_unit}")
    if cp_unit == "Eh/K":
        print(f"{'Cp_tot':12s}: {cp_tot: .10e} Eh/K ({cp_tot_evk: .10e} eV/K)")
    else:
        print(f"{'Cp_tot':12s}: {cp_tot: .10e} {cp_unit}")
    print("------------------------------------------")
    print("Component Gibbs energies (Eh):")
    for key in ["G_elec", "G_trans", "G_rot", "G_vib"]:
        val, unit, val_ev = eh_tuple_to_values(thermo_result, key)
        if unit == "Eh":
            print(f"{key:12s}: {val: .10f} Eh  ({val_ev: .6f} eV)")
        else:
            print(f"{key:12s}: {val: .10f} {unit}")
    print("==========================================")


if __name__ == "__main__":
    main()
