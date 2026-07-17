"""`pyscf-cli vib` — vibrational frequencies and harmonic levels.

Port of calc_pyscf_vib.py.
"""

from __future__ import annotations

import numpy as np
from pyscf.hessian import thermo as pyscf_thermo

from . import core, dryrun
from .core import CM1_TO_EV
from .output import Report


def register(subparsers):
    parser = subparsers.add_parser(
        "vib",
        help="vibrational frequencies and harmonic energy levels",
        description=(
            "Compute the Hessian, perform harmonic analysis, classify the "
            "normal modes, and print the quantized vibrational levels "
            "E_n = (n + 1/2) h nu for each real mode."
        ),
    )
    core.add_common_arguments(
        parser, default_basis="6-31g", theories=("scf", "dft"), include_dry_run=True
    )
    parser.add_argument("--nmax", type=int, default=3,
                        help="highest vibrational quantum number to print (default: 3)")
    parser.set_defaults(func=run)
    return parser


def energy_levels_from_freq(freq_cm1, nmax):
    quanta = freq_cm1 * CM1_TO_EV
    return [(n, (n + 0.5) * quanta) for n in range(nmax + 1)]


def run(args):
    core.finalize_common_args(args)

    if args.dry_run:
        print(dryrun.vib_script(args))
        return 0

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf, method_label = core.build_reference(mol, args.theory, args.method, args.xc)
    core.run_scf(mf)

    hess = mf.Hessian().kernel()
    vib_data = pyscf_thermo.harmonic_analysis(mf.mol, hess)
    freqs = np.atleast_1d(np.asarray(vib_data["freq_wavenumber"], dtype=complex))

    r = Report("PySCF Vibrational Analysis (pyscf-cli vib)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Theory", args.theory, key="theory")
    r.kv("Method", method_label, key="method_label")
    if args.theory == "dft":
        r.kv("XC functional", args.xc, key="xc")
    r.kv("Basis", args.basis, key="basis")
    r.kv("Charge", args.charge, key="charge")
    r.kv("Spin (2S)", args.spin, key="spin")
    r.energy("SCF E_tot", mf.e_tot, key="e_scf")
    r.rule()
    r.line("Mode  Frequency(cm^-1)  Type")

    real_positive_modes = []
    mode_table = []
    for i, f in enumerate(freqs, start=1):
        if abs(f.imag) > 1e-8:
            r.line(f"{i:4d}  {f.imag:14.4f}i  imaginary")
            mode_table.append({"mode": i, "freq_cm1": float(f.imag),
                               "type": "imaginary"})
            continue

        f_real = float(f.real)
        if f_real > 1e-6:
            r.line(f"{i:4d}  {f_real:16.4f}  vibrational")
            real_positive_modes.append((i, f_real))
            mode_table.append({"mode": i, "freq_cm1": f_real, "type": "vibrational"})
        else:
            r.line(f"{i:4d}  {f_real:16.4f}  transl/rot")
            mode_table.append({"mode": i, "freq_cm1": f_real, "type": "transl/rot"})
    r.add("modes", mode_table)

    r.rule()
    if not real_positive_modes:
        r.line("No positive real vibrational modes were found.")
        if any(abs(f.imag) > 1e-8 for f in freqs):
            r.line("Imaginary frequencies suggest this geometry is not a minimum:")
            r.line("try 'pyscf-cli relax' first, then rerun the analysis.")
        r.rule("=")
        r.emit(json_target=args.json)
        return 0

    r.line(f"Harmonic vibrational levels up to n={args.nmax}")
    r.line("(E_n = (n + 1/2) h nu, for each normal mode)")

    for mode_idx, freq_cm1 in real_positive_modes:
        r.line()
        r.line(f"Mode {mode_idx}  nu = {freq_cm1:.4f} cm^-1")
        r.line("  n   E_n (eV)")
        for n, en in energy_levels_from_freq(freq_cm1, args.nmax):
            r.line(f" {n:2d}   {en:10.6f}")

    r.rule("=")
    r.emit(json_target=args.json)
    return 0
