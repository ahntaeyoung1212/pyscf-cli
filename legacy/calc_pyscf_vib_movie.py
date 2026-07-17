#!/usr/bin/env python3

import argparse
import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from pyscf import gto, scf, dft
from pyscf.hessian import thermo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COMMON_BASIS_CHOICES = [
    "sto-3g", "3-21g", "6-31g", "6-31g*", "6-31g**",
    "6-31+g", "6-31+g*", "6-31+g**", "6-311g", "6-311g*", "6-311g**",
    "6-311+g", "6-311+g*", "6-311+g**", "cc-pvdz", "cc-pvtz", "cc-pvqz",
    "aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "def2-svp", "def2-tzvp",
    "def2-tzvpp", "def2-qzvp",
]

COMMON_XC_CHOICES = [
    "lda,vwn", "pbe", "pbe0", "b3lyp", "blyp", "bp86", "m06", "m06-2x", "wb97x", "wb97x-d",
]

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}

COLORS = {
    "H": "#dddddd", "C": "#4d4d4d", "N": "#2a5bd7", "O": "#d7352a",
    "F": "#33aa55", "P": "#dd9922", "S": "#d8c018", "Cl": "#33aa55",
    "Br": "#8b4513", "I": "#7b3f98",
}


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


def build_mf(mol, theory, method, xc):
    theory = theory.lower()
    method = method.lower()

    if theory == "dft":
        if method in ("rhf", "rohf"):
            mf = dft.RKS(mol)
        elif method == "uhf":
            mf = dft.UKS(mol)
        else:
            raise ValueError("Unknown method for DFT: rhf, uhf, rohf")
        mf.xc = xc
        return mf

    if method == "rhf":
        return scf.RHF(mol)
    if method == "uhf":
        return scf.UHF(mol)
    if method == "rohf":
        return scf.ROHF(mol)

    raise ValueError("Unknown method: rhf, uhf, rohf")


def infer_bonds(symbols, coords):
    bonds = []
    nat = len(symbols)
    for i in range(nat):
        ri = COVALENT_RADII.get(symbols[i], 0.8)
        for j in range(i + 1, nat):
            rj = COVALENT_RADII.get(symbols[j], 0.8)
            cutoff = 1.25 * (ri + rj)
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= cutoff:
                bonds.append((i, j))
    return bonds


def write_multixyz(path, symbols, frames):
    with open(path, "w", encoding="utf-8") as f:
        nat = len(symbols)
        for iframe, xyz in enumerate(frames):
            f.write(f"{nat}\n")
            f.write(f"frame {iframe}\n")
            for s, (x, y, z) in zip(symbols, xyz):
                f.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def render_gif(path, symbols, frames, bonds, fps):
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    all_xyz = np.array(frames)
    xyz_min = all_xyz.min(axis=(0, 1))
    xyz_max = all_xyz.max(axis=(0, 1))
    center = 0.5 * (xyz_min + xyz_max)
    span = float(np.max(xyz_max - xyz_min))
    span = max(span, 2.0)

    scat = ax.scatter([], [], [], s=140)
    lines = [ax.plot([], [], [], lw=2.0, color="#808080")[0] for _ in bonds]

    colors = [COLORS.get(s, "#999999") for s in symbols]

    def set_axes():
        half = 0.6 * span
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X (Ang)")
        ax.set_ylabel("Y (Ang)")
        ax.set_zlabel("Z (Ang)")
        ax.view_init(elev=20, azim=35)

    def init():
        set_axes()
        return [scat] + lines

    def update(k):
        xyz = frames[k]
        scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        scat.set_color(colors)
        for ln, (i, j) in zip(lines, bonds):
            ln.set_data([xyz[i, 0], xyz[j, 0]], [xyz[i, 1], xyz[j, 1]])
            ln.set_3d_properties([xyz[i, 2], xyz[j, 2]])
        ax.set_title(f"Frame {k + 1}/{len(frames)}")
        return [scat] + lines

    ani = FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate vibrational-mode animations from XYZ using PySCF")
    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument("--basis", default="6-31g", type=str.lower, choices=COMMON_BASIS_CHOICES)
    parser.add_argument("--method", default="auto", choices=["auto", "rhf", "uhf", "rohf"])
    parser.add_argument("--theory", default="scf", choices=["scf", "dft"])
    parser.add_argument("--xc", default="b3lyp", type=str.lower, choices=COMMON_XC_CHOICES)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--unit", default="Angstrom", choices=["Angstrom", "Bohr"])

    parser.add_argument("--mode", type=int, default=None, help="1-based mode index to animate")
    parser.add_argument("--amplitude", type=float, default=0.25, help="Displacement amplitude in Angstrom")
    parser.add_argument("--nframes", type=int, default=32, help="Number of frames per cycle")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    parser.add_argument("--outdir", default=None, help="Output directory")

    args = parser.parse_args()

    atoms = read_xyz(args.xyz)
    symbols = [a[0] for a in atoms]
    coords0 = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)

    method = "rhf" if args.method == "auto" and args.spin == 0 else ("uhf" if args.method == "auto" else args.method)

    mol = gto.M(
        atom=atoms_to_pyscf_string(atoms),
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
    modes = np.array(vib["norm_mode"], dtype=float)

    valid = []
    for i, f in enumerate(freqs, start=1):
        if abs(f.imag) < 1e-8 and f.real > 1e-6:
            valid.append(i)

    if args.mode is not None:
        if args.mode < 1 or args.mode > len(freqs):
            raise ValueError(f"--mode must be 1..{len(freqs)}")
        selected = [args.mode]
    else:
        selected = valid

    if len(selected) == 0:
        raise RuntimeError("No positive real vibrational mode selected.")

    xyz_basename = os.path.basename(args.xyz)
    xyz_stem = os.path.splitext(xyz_basename)[0]
    if args.outdir is None:
        if xyz_stem.startswith("XYZ_") and len(xyz_stem) > 4:
            tag = xyz_stem[4:]
        else:
            tag = xyz_stem
        outdir = f"vib_movies_{tag}"
    else:
        outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)
    bonds = infer_bonds(symbols, coords0)

    print("==========================================")
    print("PySCF Vibrational Movie Generator")
    print("==========================================")
    print(f"XYZ file  : {args.xyz}")
    print(f"E_tot     : {mf.e_tot:.10f} Hartree")
    print(f"outdir    : {outdir}")
    print(f"selected  : {selected}")

    phase = np.linspace(0.0, 2.0 * np.pi, args.nframes, endpoint=False)

    for midx in selected:
        disp = modes[midx - 1]
        disp_norm = np.max(np.linalg.norm(disp, axis=1))
        if disp_norm < 1e-12:
            print(f"Mode {midx:3d}: skipped (zero displacement)")
            continue
        disp = disp / disp_norm

        frames = []
        for ph in phase:
            xyz = coords0 + args.amplitude * np.sin(ph) * disp
            frames.append(xyz)

        mode_freq = freqs[midx - 1]
        tag = f"mode{midx:03d}"
        xyz_path = os.path.join(outdir, f"{tag}.xyz")
        gif_path = os.path.join(outdir, f"{tag}.gif")

        write_multixyz(xyz_path, symbols, frames)
        render_gif(gif_path, symbols, frames, bonds, fps=args.fps)

        print(f"Mode {midx:3d}: {mode_freq.real:10.4f} cm^-1 -> {xyz_path}, {gif_path}")

    print("==========================================")


if __name__ == "__main__":
    main()
