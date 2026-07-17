"""`pyscf-cli thermo` — thermochemistry from harmonic analysis.

Port of calc_pyscf_thermo.py: ZPE, thermal corrections, enthalpy, Gibbs
free energy, entropy, and heat capacity at a given T and p.
"""

from __future__ import annotations

import numpy as np
from pyscf.hessian import thermo as pyscf_thermo

from . import core, dryrun
from .core import HARTREE_TO_EV
from .output import Report


def register(subparsers):
    parser = subparsers.add_parser(
        "thermo",
        help="thermochemistry: ZPE, H, G, S, Cp at given T and p",
        description=(
            "Run harmonic analysis and ideal-gas statistical thermodynamics "
            "to obtain zero-point energy, thermal corrections, enthalpy, "
            "Gibbs free energy, entropy, and heat capacity."
        ),
    )
    core.add_common_arguments(
        parser, default_basis="6-31g", theories=("scf", "dft"), include_dry_run=True
    )
    parser.add_argument("--temp", type=float, default=298.15,
                        help="temperature in K (default: 298.15)")
    parser.add_argument("--pressure", type=float, default=101325.0,
                        help="pressure in Pa (default: 101325)")
    parser.set_defaults(func=run)
    return parser


def _energy_row(r, key, result):
    value, unit = result[key]
    if unit == "Eh":
        r.line(f"{key:12s}: {value * HARTREE_TO_EV: .6f} eV  ({value: .10f} Eh)")
        r.add(key, {"hartree": float(value), "eV": float(value * HARTREE_TO_EV)})
    else:
        r.line(f"{key:12s}: {value} {unit}")
        r.add(key, {"value": value, "unit": unit})


def _entropy_row(r, key, result):
    value, unit = result[key]
    if unit == "Eh/K":
        r.line(f"{key:12s}: {value * HARTREE_TO_EV: .10e} eV/K ({value: .10e} Eh/K)")
        r.add(key, {"hartree_per_K": float(value),
                    "eV_per_K": float(value * HARTREE_TO_EV)})
    else:
        r.line(f"{key:12s}: {value: .10e} {unit}")
        r.add(key, {"value": value, "unit": unit})


def run(args):
    core.finalize_common_args(args)

    if args.dry_run:
        print(dryrun.thermo_script(args))
        return 0

    core.require_hessian_capable(args.method, args.spin)

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf, method_label = core.build_reference(mol, args.theory, args.method, args.xc)
    core.run_scf(mf)

    hess = mf.Hessian().kernel()
    vib_data = pyscf_thermo.harmonic_analysis(mol, hess)
    freqs_cm1 = np.atleast_1d(np.asarray(vib_data["freq_wavenumber"], dtype=complex))
    positive_real = int(np.sum(
        (np.abs(freqs_cm1.imag) < 1e-8) & (freqs_cm1.real > 1e-6)
    ))
    n_imag = int(np.sum(np.abs(freqs_cm1.imag) > 1e-8))

    thermo_result = pyscf_thermo.thermo(
        mf, vib_data["freq_au"], temperature=args.temp, pressure=args.pressure
    )

    r = Report("PySCF Thermochemistry (pyscf-cli thermo)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Method", method_label, key="method_label")
    r.kv("Basis", args.basis, key="basis")
    r.kv("Charge", args.charge, key="charge")
    r.kv("Spin (2S)", args.spin, key="spin")
    r.kv("Temperature", f"{args.temp:.2f} K", key="temperature_K")
    r.kv("Pressure", f"{args.pressure:.1f} Pa", key="pressure_Pa")
    r.kv("Real vib modes", positive_real, key="n_real_modes")
    if n_imag:
        r.kv("Imag. modes", n_imag, key="n_imaginary_modes")
        r.line("WARNING: imaginary frequencies found - this geometry is not a")
        r.line("minimum, so the thermochemistry below is NOT valid (imaginary")
        r.line("modes are silently excluded). Optimize first: pyscf-cli relax")
    r.rule()

    for key in ("E0", "ZPE", "E_0K", "E_tot", "H_tot", "G_tot"):
        _energy_row(r, key, thermo_result)
    _entropy_row(r, "S_tot", thermo_result)
    _entropy_row(r, "Cp_tot", thermo_result)

    r.rule()
    r.line("Component Gibbs energies:")
    for key in ("G_elec", "G_trans", "G_rot", "G_vib"):
        _energy_row(r, key, thermo_result)

    r.rule("=")
    r.emit(json_target=args.json)
    return core.scf_exit_code(mf)
