"""`pyscf-cli vibmovie` — animate vibrational normal modes as GIFs.

Port of calc_pyscf_vib_movie.py.  For each selected real mode, writes a
multi-frame XYZ trajectory and an animated GIF.
"""

from __future__ import annotations

import os

import numpy as np
from pyscf.hessian import thermo as pyscf_thermo

from . import core
from .core import InputError
from .output import Report, use_headless_matplotlib

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}

COLORS = {
    "H": "#dddddd", "C": "#4d4d4d", "N": "#2a5bd7", "O": "#d7352a",
    "F": "#33aa55", "P": "#dd9922", "S": "#d8c018", "Cl": "#33aa55",
    "Br": "#8b4513", "I": "#7b3f98",
}


def register(subparsers):
    parser = subparsers.add_parser(
        "vibmovie",
        help="animate vibrational normal modes as GIF movies",
        description=(
            "Compute the normal modes and write, for each selected mode, "
            "a multi-frame XYZ trajectory and an animated GIF."
        ),
    )
    core.add_common_arguments(
        parser, default_basis="6-31g", theories=("scf", "dft")
    )
    parser.add_argument("--mode", type=int, default=None,
                        help="1-based mode index to animate (default: all real modes)")
    parser.add_argument("--amplitude", type=float, default=0.25,
                        help="displacement amplitude in Angstrom (default: 0.25)")
    parser.add_argument("--nframes", type=int, default=32,
                        help="frames per vibration cycle (default: 32)")
    parser.add_argument("--fps", type=int, default=12,
                        help="GIF frame rate (default: 12)")
    parser.add_argument("--outdir", default=None, metavar="DIR",
                        help="output directory (default: <input>_vibmovie)")
    parser.set_defaults(func=run)
    return parser


def infer_bonds(symbols, coords):
    bonds = []
    nat = len(symbols)
    for i in range(nat):
        ri = COVALENT_RADII.get(symbols[i], 0.8)
        for j in range(i + 1, nat):
            rj = COVALENT_RADII.get(symbols[j], 0.8)
            cutoff = 1.25 * (ri + rj)
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
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
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

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

    ani = FuncAnimation(fig, update, init_func=init, frames=len(frames),
                        interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def run(args):
    core.finalize_common_args(args)
    core.require_hessian_capable(args.method, args.spin)
    use_headless_matplotlib()

    symbols = [a[0] for a in args.atoms]
    coords0 = np.array([[a[1], a[2], a[3]] for a in args.atoms], dtype=float)
    if args.unit.lower().startswith("b"):
        # Animation, bond detection, amplitude, and the multi-frame XYZ all
        # work in Angstrom; convert Bohr input coordinates once here.
        coords0 *= core.BOHR_TO_ANG

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf, method_label = core.build_reference(mol, args.theory, args.method, args.xc)
    core.run_scf(mf)

    hess = mf.Hessian().kernel()
    vib_data = pyscf_thermo.harmonic_analysis(mf.mol, hess)

    freqs = np.atleast_1d(np.asarray(vib_data["freq_wavenumber"], dtype=complex))
    modes = np.asarray(vib_data["norm_mode"], dtype=float)

    valid = [
        i for i, f in enumerate(freqs, start=1)
        if abs(f.imag) < 1e-8 and f.real > 1e-6
    ]

    if args.mode is not None:
        if args.mode < 1 or args.mode > len(freqs):
            raise InputError(f"--mode must be between 1 and {len(freqs)}")
        selected = [args.mode]
    else:
        selected = valid

    if not selected:
        raise InputError(
            "No positive real vibrational mode found to animate.\n"
            "If frequencies are imaginary, relax the geometry first: "
            "pyscf-cli relax <file.xyz>"
        )

    outdir = args.outdir or core.output_stem(args, "vibmovie")
    os.makedirs(outdir, exist_ok=True)
    bonds = infer_bonds(symbols, coords0)

    r = Report("PySCF Vibrational Movie Generator (pyscf-cli vibmovie)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Method", method_label, key="method_label")
    r.kv("Basis", args.basis, key="basis")
    r.energy("E_tot", mf.e_tot, key="e_tot")
    r.kv("Output dir", outdir, key="outdir")
    r.kv("Modes", ", ".join(str(m) for m in selected), key="selected_modes")
    r.rule()

    phase = np.linspace(0.0, 2.0 * np.pi, args.nframes, endpoint=False)
    movies = []

    for midx in selected:
        disp = modes[midx - 1]
        disp_norm = np.max(np.linalg.norm(disp, axis=1))
        if disp_norm < 1e-12:
            r.line(f"Mode {midx:3d}: skipped (zero displacement)")
            continue
        disp = disp / disp_norm

        frames = [
            coords0 + args.amplitude * np.sin(ph) * disp for ph in phase
        ]

        mode_freq = freqs[midx - 1]
        tag = f"mode{midx:03d}"
        xyz_path = os.path.join(outdir, f"{tag}.xyz")
        gif_path = os.path.join(outdir, f"{tag}.gif")

        write_multixyz(xyz_path, symbols, frames)
        render_gif(gif_path, symbols, frames, bonds, fps=args.fps)

        imaginary = abs(mode_freq.imag) > 1e-8
        if imaginary:
            freq_text = f"{abs(mode_freq.imag):10.4f}i cm^-1 (imaginary mode)"
            freq_value = -abs(float(mode_freq.imag))
        else:
            freq_text = f"{mode_freq.real:10.4f} cm^-1"
            freq_value = float(mode_freq.real)
        r.line(f"Mode {midx:3d}: {freq_text} -> {xyz_path}, {gif_path}")
        movies.append({
            "mode": midx,
            "freq_cm1": freq_value,
            "imaginary": imaginary,
            "xyz": xyz_path,
            "gif": gif_path,
        })

    r.add("movies", movies)
    r.rule("=")
    r.emit(json_target=args.json)
    return core.scf_exit_code(mf)
