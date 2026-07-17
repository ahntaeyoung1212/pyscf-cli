"""`pyscf-cli energy` — single-point energy, MOs, PES scan, and analysis.

Port of legacy calc_pyscf.py.  Numerical formulas are kept identical;
only input handling and output formatting go through core/output.
"""

from __future__ import annotations

import numpy as np
from pyscf import scf
from pyscf.hessian import thermo as pyscf_thermo

from . import core, dryrun
from .core import HARTREE_TO_EV, InputError
from .output import Report


def register(subparsers):
    parser = subparsers.add_parser(
        "energy",
        help="single-point energy with MO analysis (SCF/DFT/MP2/CCSD/CCSD(T))",
        description=(
            "Compute the total energy of a molecule and report MO energies and "
            "<S^2>. Extras: diatomic PES scan, MOM Delta-SCF excited states, "
            "harmonic ZPE, and total-energy decompositions."
        ),
    )
    core.add_common_arguments(parser, include_dry_run=True)

    pes = parser.add_argument_group("PES scan (diatomics, --theory scf)")
    pes.add_argument("--pes", action="store_true",
                     help="scan the bond distance instead of a single point")
    pes.add_argument("--rmin", type=float, default=0.5,
                     help="minimum bond distance, in --unit units (default: 0.5)")
    pes.add_argument("--rmax", type=float, default=6.0,
                     help="maximum bond distance, in --unit units (default: 6.0)")
    pes.add_argument("--npts", type=int, default=25,
                     help="number of scan points (default: 25)")

    mom = parser.add_argument_group(
        "MOM Delta-SCF excited state (--theory scf, --method uhf)"
    )
    mom.add_argument("--mom", action="store_true",
                     help="converge an excited state via the maximum overlap method")
    mom.add_argument("--promote-from", type=int, default=None,
                     help="0-based MO index to depopulate (default: HOMO)")
    mom.add_argument("--promote-to", type=int, default=None,
                     help="0-based MO index to populate (default: LUMO)")
    mom.add_argument("--promote-spin", default="alpha", choices=["alpha", "beta"],
                     help="spin channel for the promotion (default: alpha)")

    ana = parser.add_argument_group("analysis")
    ana.add_argument("--zpe", action="store_true",
                     help="harmonic zero-point energy from the SCF/DFT Hessian")
    ana.add_argument("--decompose-total-energy", action="store_true",
                     help="decompose the SCF total energy into T, V_ne, U, J, V_nn")
    ana.add_argument("--fixed-occ-decomp", action="store_true",
                     help="energy-term changes for a fixed-orbital occupation "
                          "promotion (RHF/UHF, --theory scf)")

    parser.set_defaults(func=run)
    return parser


# ---------------------------------------------------------------------------
# Energy decompositions (formulas identical to legacy calc_pyscf.py)
# ---------------------------------------------------------------------------

def rhf_energy_decomposition(mf, dm):
    hcore = mf.get_hcore()
    j, k = mf.get_jk(dm=dm)
    e_one = np.einsum("ij,ji->", hcore, dm).real
    e_u = 0.5 * np.einsum("ij,ji->", j, dm).real
    e_j = -0.25 * np.einsum("ij,ji->", k, dm).real
    e_nuc = mf.energy_nuc()
    return {
        "e_one": e_one,
        "e_u": e_u,
        "e_j": e_j,
        "e_nuc": e_nuc,
        "e_tot": e_one + e_u + e_j + e_nuc,
    }


def uhf_energy_decomposition(mf, dma, dmb):
    hcore = mf.get_hcore()
    dm_tot = dma + dmb
    j = mf.get_j(dm=dm_tot)
    ka, kb = mf.get_k(dm=(dma, dmb))

    e_one = (np.einsum("ij,ji->", hcore, dma).real +
             np.einsum("ij,ji->", hcore, dmb).real)
    e_u = 0.5 * np.einsum("ij,ji->", j, dm_tot).real
    e_j = -0.5 * (
        np.einsum("ij,ji->", ka, dma).real +
        np.einsum("ij,ji->", kb, dmb).real
    )
    e_nuc = mf.energy_nuc()
    return {
        "e_one": e_one,
        "e_u": e_u,
        "e_j": e_j,
        "e_nuc": e_nuc,
        "e_tot": e_one + e_u + e_j + e_nuc,
    }


def decompose_total_energy(mf):
    hcore = mf.get_hcore()
    t_ao = mf.mol.intor_symmetric("int1e_kin")
    vne_ao = hcore - t_ao
    enuc = mf.energy_nuc()

    if isinstance(mf, scf.uhf.UHF):
        dma, dmb = mf.make_rdm1()
        dm_tot = dma + dmb
        e_kin = np.einsum("ij,ji->", t_ao, dm_tot).real
        e_vne = np.einsum("ij,ji->", vne_ao, dm_tot).real
        j = mf.get_j(dm=dm_tot)
        ka, kb = mf.get_k(dm=(dma, dmb))
        e_u = 0.5 * np.einsum("ij,ji->", j, dm_tot).real
        e_j = -0.5 * (
            np.einsum("ij,ji->", ka, dma).real +
            np.einsum("ij,ji->", kb, dmb).real
        )
    else:
        dm = mf.make_rdm1()
        e_kin = np.einsum("ij,ji->", t_ao, dm).real
        e_vne = np.einsum("ij,ji->", vne_ao, dm).real
        j, k = mf.get_jk(dm=dm)
        e_u = 0.5 * np.einsum("ij,ji->", j, dm).real
        e_j = -0.25 * np.einsum("ij,ji->", k, dm).real

    return {
        "e_kin": e_kin,
        "e_vne": e_vne,
        "e_u": e_u,
        "e_j": e_j,
        "e_nuc": enuc,
        "e_tot": e_kin + e_vne + e_u + e_j + enuc,
    }


# ---------------------------------------------------------------------------
# MOM Delta-SCF (port of run_mom_excited_scf)
# ---------------------------------------------------------------------------

def run_mom_excited_scf(args):
    if args.method != "uhf":
        raise InputError(
            "MOM excited-state SCF requires --method uhf "
            "(the promotion needs independent alpha/beta orbitals)."
        )

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    ref_mf = core.build_mf(mol, args.method)
    core.run_scf(ref_mf)

    mo0 = ref_mf.mo_coeff
    occ0 = np.array(ref_mf.mo_occ, copy=True)
    target_occ = np.array(occ0, copy=True)

    spin_idx = 0 if args.promote_spin == "alpha" else 1
    spin_label = "Alpha" if spin_idx == 0 else "Beta"

    occ_spin = target_occ[spin_idx]
    occ_idx = np.where(occ_spin > 0.5)[0]
    vir_idx = np.where(occ_spin < 0.5)[0]
    if len(occ_idx) == 0 or len(vir_idx) == 0:
        raise InputError(
            f"Cannot determine occupied/virtual orbitals for {spin_label}."
        )

    promote_from = args.promote_from if args.promote_from is not None else int(occ_idx[-1])
    promote_to = args.promote_to if args.promote_to is not None else int(vir_idx[0])

    nmo = target_occ.shape[1]
    if not (0 <= promote_from < nmo and 0 <= promote_to < nmo):
        raise InputError(
            f"Promotion indices out of range: nmo={nmo}, "
            f"promote_from={promote_from}, promote_to={promote_to}"
        )
    if target_occ[spin_idx, promote_from] < 1.0:
        raise InputError(
            f"{spin_label} MO {promote_from + 1} is not occupied in the reference state."
        )
    if target_occ[spin_idx, promote_to] > 0.0:
        raise InputError(
            f"{spin_label} MO {promote_to + 1} is already occupied in the reference state."
        )

    target_occ[spin_idx, promote_from] -= 1.0
    target_occ[spin_idx, promote_to] += 1.0

    mf = core.build_mf(mol, args.method)
    mf.verbose = 0
    mf = scf.addons.mom_occ(mf, mo0, target_occ)
    mf.kernel(dm0=ref_mf.make_rdm1())

    return mf, promote_from, promote_to, spin_label


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _results_section(r, mf, e_tot, info):
    r.rule()
    r.kv("Method", info["label"], key="method_label")
    if "e_ref" in info:
        r.energy("Reference E", info["e_ref"], key="e_ref")
    if "e_corr" in info:
        r.energy("MP2 corr E", info["e_corr"], key="e_corr_mp2")
    if "e_corr_ccsd" in info:
        r.energy("CCSD corr E", info["e_corr_ccsd"], key="e_corr_ccsd")
    if "e_corr_t" in info:
        r.energy("(T) corr E", info["e_corr_t"], key="e_corr_t")
    r.energy("Total Energy", e_tot, key="e_tot")

    if hasattr(mf, "spin_square"):
        s2, mult = mf.spin_square()
        r.kv("<S^2>", f"{s2:.6f}", key="s_squared")
        r.kv("Multiplicity", f"{mult:.1f}", key="multiplicity")

    r.line()
    r.line("MO energies (Hartree / eV):")
    moe = np.asarray(mf.mo_energy)
    if moe.ndim == 1:
        for i, e in enumerate(moe):
            r.line(f"  MO {i + 1:2d}: {e:10.6f}  {e * HARTREE_TO_EV:10.4f}")
    elif moe.ndim == 2:
        for spin_name, energies in zip(("Alpha", "Beta"), moe):
            r.line(f"  {spin_name} MO energies:")
            for i, e in enumerate(energies):
                r.line(f"  MO {i + 1:2d}: {e:10.6f}  {e * HARTREE_TO_EV:10.4f}")
    else:
        r.line("  Unexpected mf.mo_energy dimensions.")
    r.add("mo_energies_hartree", moe)


def _total_decomposition_section(r, mf):
    d = decompose_total_energy(mf)
    r.line()
    r.line("Total-energy decomposition:")
    rows = [
        ("Kinetic (T)", "e_kin", "decomp_kinetic"),
        ("Nuc-el (V_ne)", "e_vne", "decomp_vne"),
        ("Hartree (U)", "e_u", "decomp_hartree"),
        ("Exchange (J)", "e_j", "decomp_exchange"),
        ("Nuc-nuc (V_nn)", "e_nuc", "decomp_vnn"),
        ("Sum", "e_tot", "decomp_sum"),
    ]
    for label, dkey, jkey in rows:
        ev = d[dkey] * HARTREE_TO_EV
        r.line(f"  {label:<14s}: {ev: .6f} eV")
        r.add(jkey, {"hartree": float(d[dkey]), "eV": float(ev)})


def _zpe_section(r, mf):
    r.line()
    r.line("ZPE analysis:")
    if mf.mol.natm < 2:
        r.line("  Skipped: ZPE requires at least 2 atoms.")
        return

    hess = mf.Hessian().kernel()
    vib = pyscf_thermo.harmonic_analysis(mf.mol, hess)

    freqs = np.atleast_1d(np.asarray(vib["freq_wavenumber"]))
    if np.iscomplexobj(freqs):
        freqs = freqs.real[np.abs(freqs.imag) < 1e-8]
    real_freqs = freqs[np.isfinite(freqs) & (freqs > 1e-6)]
    if real_freqs.size == 0:
        r.line("  No positive real vibrational frequencies found.")
        return

    # ZPE = (1/2) sum(h nu) over the real modes.  (The legacy script summed
    # PySCF's "freq_au" values as if they were Hartree, overestimating the
    # ZPE by a factor of ~42.7; fixed here by converting from wavenumbers.)
    zpe_h = 0.5 * np.sum(real_freqs) / core.HARTREE_TO_CM1
    r.kv("  Vib modes", int(real_freqs.size), key="zpe_n_modes")
    r.energy("  ZPE", zpe_h, key="zpe")


def _fixed_occ_section(r, args):
    if args.method == "rohf":
        raise InputError("--fixed-occ-decomp supports RHF/UHF only (not ROHF).")
    if args.method == "rhf" and args.spin != 0:
        raise InputError(
            "RHF fixed-occ decomposition requires --spin 0. "
            "Use --method uhf for open-shell molecules."
        )
    if args.method == "uhf" and args.spin == 0:
        raise InputError(
            "UHF fixed-occ decomposition is for open-shell cases. "
            "Use --method rhf (or omit --method) for closed-shell molecules."
        )

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf = core.build_mf(mol, args.method)
    core.run_scf(mf)

    if args.method == "rhf":
        occ_ref = np.array(mf.mo_occ, copy=True)
        occ_idx = np.where(occ_ref > 1.0)[0]
        vir_idx = np.where(occ_ref < 1.0)[0]
        if len(occ_idx) == 0 or len(vir_idx) == 0:
            raise InputError("Cannot determine occupied/virtual orbitals for RHF reference.")

        promote_from = args.promote_from if args.promote_from is not None else int(occ_idx[-1])
        promote_to = args.promote_to if args.promote_to is not None else int(vir_idx[0])

        nmo = occ_ref.shape[0]
        if not (0 <= promote_from < nmo and 0 <= promote_to < nmo):
            raise InputError(
                f"Promotion indices out of range: nmo={nmo}, "
                f"promote_from={promote_from}, promote_to={promote_to}"
            )
        if occ_ref[promote_from] < 1.0:
            raise InputError(f"MO {promote_from + 1} is not occupied in reference RHF.")
        if occ_ref[promote_to] > 1.0:
            raise InputError(f"MO {promote_to + 1} is already occupied in reference RHF.")

        occ_exc = np.array(occ_ref, copy=True)
        occ_exc[promote_from] -= 1.0
        occ_exc[promote_to] += 1.0

        dm_ref = mf.make_rdm1(mf.mo_coeff, occ_ref)
        dm_exc = mf.make_rdm1(mf.mo_coeff, occ_exc)

        d_ref = rhf_energy_decomposition(mf, dm_ref)
        d_exc = rhf_energy_decomposition(mf, dm_exc)
        promo_text = f"MO {promote_from + 1} -> MO {promote_to + 1} (fixed orbitals)"
        model_text = "RHF"
        reference_text = "closed-shell SCF occupancy"
    else:
        spin_idx = 0 if args.promote_spin == "alpha" else 1
        spin_label = "Alpha" if spin_idx == 0 else "Beta"
        occ_ref = np.array(mf.mo_occ, copy=True)
        occ_spin = occ_ref[spin_idx]
        occ_idx = np.where(occ_spin > 0.5)[0]
        vir_idx = np.where(occ_spin < 0.5)[0]
        if len(occ_idx) == 0 or len(vir_idx) == 0:
            raise InputError(
                f"Cannot determine occupied/virtual orbitals for {spin_label}."
            )

        promote_from = args.promote_from if args.promote_from is not None else int(occ_idx[-1])
        promote_to = args.promote_to if args.promote_to is not None else int(vir_idx[0])
        nmo = occ_spin.shape[0]
        if not (0 <= promote_from < nmo and 0 <= promote_to < nmo):
            raise InputError(
                f"Promotion indices out of range: nmo={nmo}, "
                f"promote_from={promote_from}, promote_to={promote_to}"
            )
        if occ_ref[spin_idx, promote_from] < 0.5:
            raise InputError(
                f"{spin_label} MO {promote_from + 1} is not occupied in reference UHF."
            )
        if occ_ref[spin_idx, promote_to] > 0.5:
            raise InputError(
                f"{spin_label} MO {promote_to + 1} is already occupied in reference UHF."
            )

        occ_exc = np.array(occ_ref, copy=True)
        occ_exc[spin_idx, promote_from] -= 1.0
        occ_exc[spin_idx, promote_to] += 1.0

        dma_ref, dmb_ref = mf.make_rdm1(mf.mo_coeff, occ_ref)
        dma_exc, dmb_exc = mf.make_rdm1(mf.mo_coeff, occ_exc)
        d_ref = uhf_energy_decomposition(mf, dma_ref, dmb_ref)
        d_exc = uhf_energy_decomposition(mf, dma_exc, dmb_exc)
        promo_text = (
            f"{spin_label} MO {promote_from + 1} -> MO {promote_to + 1} (fixed orbitals)"
        )
        model_text = "UHF"
        reference_text = "open-shell SCF occupancy"

    r.rule()
    r.line(f"Fixed-orbital Occupation Decomposition ({model_text})")
    r.kv("Reference", reference_text)
    r.kv("Promotion", promo_text, key="fixed_occ_promotion")
    for key, label in (
        ("e_one", "One-electron"),
        ("e_u", "Hartree (U)"),
        ("e_j", "Exchange (J)"),
        ("e_nuc", "Nuclear rep."),
        ("e_tot", "Total"),
    ):
        v0 = d_ref[key] * HARTREE_TO_EV
        v1 = d_exc[key] * HARTREE_TO_EV
        dv = (d_exc[key] - d_ref[key]) * HARTREE_TO_EV
        r.line(f"{label:13s}: {v0:12.6f} -> {v1:12.6f} eV   (Delta = {dv:10.6f} eV)")
    r.add("fixed_occ_reference_eV", {k: v * HARTREE_TO_EV for k, v in d_ref.items()})
    r.add("fixed_occ_excited_eV", {k: v * HARTREE_TO_EV for k, v in d_exc.items()})


def _pes_section(r, args):
    if len(args.atoms) != 2:
        raise InputError(
            f"--pes works for diatomic molecules only (got {len(args.atoms)} atoms)."
        )

    elem1 = args.atoms[0][0]
    elem2 = args.atoms[1][0]
    r_values = np.linspace(args.rmin, args.rmax, args.npts)

    r.line()
    r.line("PES scan:")
    r.line(f"R ({args.unit})    E_SCF (Hartree)    E_SCF (eV)")
    r.rule()

    table = []
    for R in r_values:
        atoms = [(elem1, 0.0, 0.0, 0.0), (elem2, 0.0, 0.0, float(R))]
        mol = core.build_mol(atoms, args.basis, args.charge, args.spin, args.unit)
        mf = core.build_mf(mol, args.method)
        core.run_scf(mf, warn=False)
        E = mf.e_tot
        r.line(f"{R:7.3f}     {E:14.8f}     {E * HARTREE_TO_EV:10.4f}")
        table.append({"R": float(R), "e_hartree": float(E)})
    r.add("pes_unit", args.unit)
    r.add("pes_scan", table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args):
    core.finalize_common_args(args)

    if args.dry_run:
        print(dryrun.energy_script(args))
        return 0

    if args.pes and args.theory != "scf":
        raise InputError("--pes currently supports --theory scf only.")
    if args.fixed_occ_decomp and args.theory != "scf":
        raise InputError("--fixed-occ-decomp supports --theory scf only.")
    if args.mom and args.theory != "scf":
        raise InputError("--mom currently supports --theory scf only.")

    r = Report("PySCF Calculation (pyscf-cli energy)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Basis", args.basis, key="basis")
    r.kv("Charge", args.charge, key="charge")
    r.kv("Spin (2S)", args.spin, key="spin")
    r.kv("Theory", args.theory, key="theory")
    if args.theory == "dft":
        r.kv("XC functional", args.xc, key="xc")
    if args.mom:
        r.kv("Excited-state", "MOM Delta-SCF enabled")
        if args.promote_from is not None and args.promote_to is not None:
            r.kv(
                "Promotion",
                f"{args.promote_spin} MO {args.promote_from + 1} "
                f"-> MO {args.promote_to + 1}",
            )
        else:
            r.kv("Promotion", f"{args.promote_spin} HOMO -> LUMO (auto)")

    if args.pes:
        _pes_section(r, args)
    elif args.fixed_occ_decomp:
        _fixed_occ_section(r, args)
    else:
        if args.mom:
            mf, p_from, p_to, p_spin = run_mom_excited_scf(args)
            r.kv("MOM used", f"{p_spin} MO {p_from + 1} -> MO {p_to + 1}",
                 key="mom_promotion")
            e_tot = mf.e_tot
            info = {"label": args.method.upper()}
        else:
            mol = core.build_mol(args.atoms, args.basis, args.charge,
                                 args.spin, args.unit)
            mf, e_tot, info = core.run_theory(mol, args.theory, args.method, args.xc)
        _results_section(r, mf, e_tot, info)

        if args.decompose_total_energy:
            if args.theory != "scf":
                r.line()
                r.line("Note: decomposition uses the SCF/DFT reference density.")
            _total_decomposition_section(r, mf)

        if args.zpe:
            if args.theory in ("mp2", "ccsd", "ccsd_t"):
                r.line()
                r.line("Note: ZPE is evaluated from the SCF reference Hessian.")
            _zpe_section(r, mf)

    r.rule("=")
    r.emit(json_target=args.json)
    return 0
