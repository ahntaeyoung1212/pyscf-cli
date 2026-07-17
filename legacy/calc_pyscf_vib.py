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
CM1_TO_EV = 1.239841984e-4

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
    with open(xyz_file, "r") as f:
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


def build_mf(mol, theory, method, xc):
    method = method.lower()
    theory = theory.lower()

    if theory == "dft":
        if method in ("rhf", "rohf"):
            mf = dft.RKS(mol)
        elif method == "uhf":
            mf = dft.UKS(mol)
        else:
            raise ValueError("Unknown method for DFT: choose rhf, uhf, or rohf")
        mf.xc = xc
        return mf

    if method == "rhf":
        return scf.RHF(mol)
    if method == "uhf":
        return scf.UHF(mol)
    if method == "rohf":
        return scf.ROHF(mol)

    raise ValueError("Unknown method: choose rhf, uhf, or rohf")


def energy_levels_from_freq(freq_cm1, nmax):
    quanta = freq_cm1 * CM1_TO_EV
    levels = []
    for n in range(nmax + 1):
        levels.append((n, (n + 0.5) * quanta))
    return levels


def main():
    parser = argparse.ArgumentParser(
        description="PySCF vibrational frequencies and harmonic vibrational levels"
    )
    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument("--basis", default="6-31g", type=str.lower,
                        choices=COMMON_BASIS_CHOICES,
                        help="Basis set")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "rhf", "uhf", "rohf"],
                        help="SCF method")
    parser.add_argument("--theory", default="scf", choices=["scf", "dft"],
                        help="Electronic-structure level for Hessian")
    parser.add_argument("--xc", default="b3lyp", type=str.lower,
                        choices=COMMON_XC_CHOICES,
                        help="XC functional for DFT")
    parser.add_argument("--spin", type=int, default=0, help="Spin = 2S")
    parser.add_argument("--charge", type=int, default=0, help="Total charge")
    parser.add_argument("--unit", default="Angstrom", choices=["Angstrom", "Bohr"],
                        help="XYZ coordinate unit")
    parser.add_argument("--nmax", type=int, default=3,
                        help="Maximum vibrational quantum number n to print")

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

    mf = build_mf(mol, args.theory, method, args.xc)
    mf.verbose = 0
    mf.kernel()

    hess = mf.Hessian().kernel()
    vib = thermo.harmonic_analysis(mf.mol, hess)

    freqs = np.array(vib["freq_wavenumber"], dtype=complex)

    print("==========================================")
    print(" PySCF Vibrational Level Analysis")
    print("==========================================")
    print(f"XYZ file      : {args.xyz}")
    print(f"Theory        : {args.theory}")
    print(f"Method        : {method}")
    if args.theory == "dft":
        print(f"XC functional : {args.xc}")
    print(f"Basis         : {args.basis}")
    print(f"Charge        : {args.charge}")
    print(f"Spin (2S)     : {args.spin}")
    print(f"SCF E_tot     : {mf.e_tot:.10f} Hartree ({mf.e_tot * HARTREE_TO_EV:.6f} eV)")
    print("------------------------------------------")
    print("Mode  Frequency(cm^-1)  Type")

    real_positive_modes = []
    for i, f in enumerate(freqs, start=1):
        if abs(f.imag) > 1e-8:
            print(f"{i:4d}  {f.real:14.4f}i  imaginary")
            continue

        f_real = float(f.real)
        if f_real > 1e-6:
            print(f"{i:4d}  {f_real:16.4f}  vibrational")
            real_positive_modes.append((i, f_real))
        else:
            print(f"{i:4d}  {f_real:16.4f}  transl/rot")

    print("------------------------------------------")
    if len(real_positive_modes) == 0:
        print("No positive real vibrational modes were found.")
        print("==========================================")
        return

    print(f"Harmonic vibrational levels up to n={args.nmax}")
    print("(E_n = (n + 1/2) h nu, for each normal mode)")

    for mode_idx, freq_cm1 in real_positive_modes:
        print(f"\nMode {mode_idx}  nu = {freq_cm1:.4f} cm^-1")
        print("  n   E_n (eV)")
        levels = energy_levels_from_freq(freq_cm1, args.nmax)
        for n, en in levels:
            print(f" {n:2d}   {en:10.6f}")

    print("==========================================")


if __name__ == "__main__":
    main()
