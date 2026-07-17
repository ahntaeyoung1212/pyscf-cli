"""Tests locking in the fixes from the pre-release adversarial review."""

import json
import subprocess
import sys

import numpy as np
import pytest
from pyscf import dft, scf

from conftest import run_cli
from pyscf_cli import core, energy


OH_XYZ = "2\nhydroxyl radical (doublet: --spin 1)\nO 0 0 0\nH 0 0 0.97\n"
LINEAR_H2O = "3\nlinear water (saddle point)\nO 0 0 0\nH 0 0 0.95\nH 0 0 -0.95\n"


# ---------------------------------------------------------------------------
# Input validation fixes
# ---------------------------------------------------------------------------

def test_zero_atom_xyz_exits_2(in_tmp):
    (in_tmp / "zero.xyz").write_text("0\nempty\n")
    assert run_cli(["energy", "zero.xyz"]) == 2


def test_totally_unknown_basis_exits_2(h2_xyz, capsys):
    assert run_cli(["energy", h2_xyz, "--basis", "qzq-99"]) == 2
    assert "qzq-99" in capsys.readouterr().err


def test_mom_pes_combination_rejected(h2_xyz):
    assert run_cli(["energy", h2_xyz, "--mom", "--pes"]) == 2


def test_json_swallowing_xyz_detected(h2o_xyz):
    assert run_cli(["energy", "--json", h2o_xyz]) == 2


def test_json_dash_is_pure_json(h2_xyz, capsys):
    assert run_cli(["energy", h2_xyz, "--json", "-"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)  # must parse as-is: no text report mixed in
    assert data["e_tot"]["hartree"] == pytest.approx(-1.11675931, abs=1e-5)


# ---------------------------------------------------------------------------
# ROHF handling
# ---------------------------------------------------------------------------

def test_rohf_hessian_guard(in_tmp):
    (in_tmp / "oh.xyz").write_text(OH_XYZ)
    assert run_cli(["vib", "oh.xyz", "--basis", "sto-3g",
                    "--spin", "1", "--method", "rohf"]) == 2
    assert run_cli(["thermo", "oh.xyz", "--basis", "sto-3g",
                    "--spin", "1", "--method", "rohf"]) == 2
    assert run_cli(["energy", "oh.xyz", "--basis", "sto-3g",
                    "--spin", "1", "--method", "rohf", "--zpe"]) == 2
    # restricted open-shell via --method rhf on an open shell: same limitation
    assert run_cli(["vib", "oh.xyz", "--basis", "sto-3g",
                    "--spin", "1", "--method", "rhf"]) == 2


def test_rohf_decompose_total_energy_works(in_tmp, capsys):
    (in_tmp / "oh.xyz").write_text(OH_XYZ)
    assert run_cli(["energy", "oh.xyz", "--basis", "sto-3g", "--spin", "1",
                    "--method", "rohf", "--decompose-total-energy"]) == 0
    out = capsys.readouterr().out
    assert "Kinetic (T)" in out


def test_rohf_decomposition_identity():
    atoms = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.97)]
    mol = core.build_mol(atoms, "sto-3g", spin=1)
    mf = core.build_mf(mol, "rohf")
    core.run_scf(mf)
    d = energy.decompose_total_energy(mf)
    assert d["e_tot"] == pytest.approx(mf.e_tot, abs=1e-8)


def test_build_ks_honors_method():
    atoms = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.97)]
    mol = core.build_mol(atoms, "sto-3g", spin=1)
    assert isinstance(core.build_ks(mol, 1, "b3lyp", "uhf"), dft.uks.UKS)
    assert isinstance(core.build_ks(mol, 1, "b3lyp", "rohf"), dft.roks.ROKS)
    mol0 = core.build_mol([("He", 0.0, 0.0, 0.0)], "sto-3g")
    assert isinstance(core.build_ks(mol0, 0, "b3lyp", "rhf"), dft.rks.RKS)


# ---------------------------------------------------------------------------
# Fixed-occupation decomposition: promoted state is a true determinant
# ---------------------------------------------------------------------------

def test_rhf_fixed_occ_promoted_state_is_determinant_energy():
    atoms = [("O", 0.0, 0.0, 0.1173),
             ("H", 0.0, 0.7572, -0.4692),
             ("H", 0.0, -0.7572, -0.4692)]
    mol = core.build_mol(atoms, "sto-3g")
    mf, e_tot, _ = core.run_theory(mol, "scf", "rhf", "b3lyp")

    d_ref, d_exc = energy.rhf_fixed_occ_decompositions(mf, 4, 5)  # HOMO->LUMO
    assert d_ref["e_tot"] == pytest.approx(e_tot, abs=1e-8)

    # Cross-check the promoted energy against PySCF's own UHF energy
    # functional evaluated on the same frozen-orbital densities.
    coeff = mf.mo_coeff
    occ_a = np.asarray(mf.mo_occ, dtype=float) / 2.0
    occ_a_exc = occ_a.copy()
    occ_a_exc[4] -= 1.0
    occ_a_exc[5] += 1.0
    dma = (coeff * occ_a_exc) @ coeff.conj().T
    dmb = (coeff * occ_a) @ coeff.conj().T

    umf = scf.UHF(mol)
    e_elec, _ = umf.energy_elec(dm=np.asarray((dma, dmb)))
    assert d_exc["e_tot"] == pytest.approx(e_elec + mf.energy_nuc(), abs=1e-8)


# ---------------------------------------------------------------------------
# Saddle-point warnings
# ---------------------------------------------------------------------------

def test_thermo_warns_on_imaginary_modes(in_tmp, capsys):
    (in_tmp / "linear.xyz").write_text(LINEAR_H2O)
    assert run_cli(["thermo", "linear.xyz", "--basis", "sto-3g"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "not" in out


def test_energy_zpe_warns_on_imaginary_modes(in_tmp, capsys):
    (in_tmp / "linear.xyz").write_text(LINEAR_H2O)
    assert run_cli(["energy", "linear.xyz", "--basis", "sto-3g", "--zpe"]) == 0
    out = capsys.readouterr().out
    assert "imaginary" in out and "saddle point" in out


# ---------------------------------------------------------------------------
# vibmovie unit handling
# ---------------------------------------------------------------------------

def test_vibmovie_bohr_input_writes_angstrom_frames(in_tmp):
    (in_tmp / "h2b.xyz").write_text("2\nH2 at 1.4 Bohr\nH 0 0 0\nH 0 0 1.4\n")
    assert run_cli(["vibmovie", "h2b.xyz", "--unit", "Bohr",
                    "--basis", "sto-3g", "--nframes", "2", "--fps", "2"]) == 0
    lines = (in_tmp / "h2b_vibmovie" / "mode001.xyz").read_text().splitlines()
    z = float(lines[3].split()[3])  # frame 0, second H atom
    assert z == pytest.approx(1.4 * core.BOHR_TO_ANG, abs=1e-4)


# ---------------------------------------------------------------------------
# Module execution
# ---------------------------------------------------------------------------

def test_python_dash_m_pyscf_cli():
    proc = subprocess.run([sys.executable, "-m", "pyscf_cli", "--version"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "pyscf-cli" in proc.stdout
