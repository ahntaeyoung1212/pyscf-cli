"""Regression tests against known energies and internal identities.

Reference values were computed with PySCF 2.12/2.13 on the bundled sample
geometries and cross-checked against the legacy calc_pyscf*.py scripts.
Tolerances are set so that BLAS/platform differences pass but a broken
formula fails loudly.
"""

import numpy as np
import pytest
from pyscf.hessian import thermo as pyscf_thermo

from pyscf_cli import core, energy

H2O_ATOMS = [
    ("O", 0.0, 0.0, 0.1173),
    ("H", 0.0, 0.7572, -0.4692),
    ("H", 0.0, -0.7572, -0.4692),
]
O2_ATOMS = [("O", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.208)]


# ---------------------------------------------------------------------------
# Total energies (Hartree)
# ---------------------------------------------------------------------------

def test_h2o_rhf_sto3g():
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    _, e_tot, info = core.run_theory(mol, "scf", "rhf", "b3lyp")
    assert info["label"] == "RHF"
    assert e_tot == pytest.approx(-74.9630231385, abs=2e-6)


def test_h2o_b3lyp_sto3g():
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    _, e_tot, _ = core.run_theory(mol, "dft", "rhf", "b3lyp")
    assert e_tot == pytest.approx(-75.3125218863, abs=1e-4)


def test_h2o_mp2_sto3g():
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    _, e_tot, info = core.run_theory(mol, "mp2", "rhf", "b3lyp")
    assert info["e_corr"] < 0.0
    assert e_tot == pytest.approx(-74.9985687717, abs=2e-6)


def test_h2o_ccsd_below_mp2():
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    _, e_mp2, _ = core.run_theory(mol, "mp2", "rhf", "b3lyp")
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    _, e_ccsd, _ = core.run_theory(mol, "ccsd", "rhf", "b3lyp")
    assert e_ccsd < e_mp2  # CCSD recovers more correlation here


def test_o2_triplet_uhf_sto3g():
    mol = core.build_mol(O2_ATOMS, "sto-3g", spin=2)
    mf, e_tot, _ = core.run_theory(mol, "scf", "uhf", "b3lyp")
    assert e_tot == pytest.approx(-147.6339696085, abs=2e-6)
    s2, mult = mf.spin_square()
    assert s2 == pytest.approx(2.0034, abs=5e-3)
    assert mult == pytest.approx(3.0, abs=5e-3)


# ---------------------------------------------------------------------------
# Energy-decomposition identities: T + V_ne + U + J + V_nn == E_SCF
# ---------------------------------------------------------------------------

def test_decomposition_identity_rhf():
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    mf, e_tot, _ = core.run_theory(mol, "scf", "rhf", "b3lyp")
    d = energy.decompose_total_energy(mf)
    assert d["e_tot"] == pytest.approx(e_tot, abs=1e-8)
    parts = d["e_kin"] + d["e_vne"] + d["e_u"] + d["e_j"] + d["e_nuc"]
    assert parts == pytest.approx(e_tot, abs=1e-8)


def test_decomposition_identity_uhf():
    mol = core.build_mol(O2_ATOMS, "sto-3g", spin=2)
    mf, e_tot, _ = core.run_theory(mol, "scf", "uhf", "b3lyp")
    d = energy.decompose_total_energy(mf)
    assert d["e_tot"] == pytest.approx(e_tot, abs=1e-8)


def test_fixed_occ_reference_matches_scf():
    """The fixed-occupation 'reference' decomposition must reproduce E_SCF."""
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    mf, e_tot, _ = core.run_theory(mol, "scf", "rhf", "b3lyp")
    dm = mf.make_rdm1()
    d = energy.rhf_energy_decomposition(mf, dm)
    assert d["e_tot"] == pytest.approx(e_tot, abs=1e-8)


def test_uhf_energy_decomposition_matches_scf():
    mol = core.build_mol(O2_ATOMS, "sto-3g", spin=2)
    mf, e_tot, _ = core.run_theory(mol, "scf", "uhf", "b3lyp")
    dma, dmb = mf.make_rdm1()
    d = energy.uhf_energy_decomposition(mf, dma, dmb)
    assert d["e_tot"] == pytest.approx(e_tot, abs=1e-8)


# ---------------------------------------------------------------------------
# Vibrations and ZPE
# ---------------------------------------------------------------------------

def test_h2o_frequencies_sto3g():
    """H2O/STO-3G RHF harmonic frequencies (regression, cm^-1)."""
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    mf, _, _ = core.run_theory(mol, "scf", "rhf", "b3lyp")
    hess = mf.Hessian().kernel()
    vib = pyscf_thermo.harmonic_analysis(mol, hess)
    freqs = np.sort(np.asarray(vib["freq_wavenumber"]).real)
    expected = [2043.11, 4488.05, 4790.30]
    assert freqs[-3:] == pytest.approx(expected, rel=5e-3)


def test_zpe_consistent_with_pyscf_thermo():
    """Our wavenumber-based ZPE must match pyscf's own thermo ZPE.

    Guards against the legacy bug where 'freq_au' was summed as if it
    were in Hartree (a factor-of-42.7 overestimate).
    """
    mol = core.build_mol(H2O_ATOMS, "sto-3g")
    mf, _, _ = core.run_theory(mol, "scf", "rhf", "b3lyp")
    hess = mf.Hessian().kernel()
    vib = pyscf_thermo.harmonic_analysis(mol, hess)

    freqs = np.asarray(vib["freq_wavenumber"]).real
    real_freqs = freqs[np.isfinite(freqs) & (freqs > 1e-6)]
    zpe_ours = 0.5 * np.sum(real_freqs) / core.HARTREE_TO_CM1

    thermo_result = pyscf_thermo.thermo(mf, vib["freq_au"],
                                        temperature=298.15, pressure=101325.0)
    zpe_pyscf, unit = thermo_result["ZPE"]
    assert unit == "Eh"
    assert zpe_ours == pytest.approx(zpe_pyscf, rel=1e-6)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_unit_constants_consistent():
    # 1 Hartree in cm^-1 and eV must be mutually consistent with CM1_TO_EV.
    assert core.HARTREE_TO_EV / core.HARTREE_TO_CM1 == pytest.approx(
        core.CM1_TO_EV, rel=1e-6
    )
