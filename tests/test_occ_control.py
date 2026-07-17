"""Tests for explicit electron-configuration control (--occ-alpha/--occ-beta)
and the singlet/triplet estimates in --fixed-occ-decomp."""

import json

import pytest

from conftest import run_cli

BE_XYZ = "1\nberyllium atom\nBe 0 0 0\n"


@pytest.fixture
def be_xyz(in_tmp):
    (in_tmp / "be.xyz").write_text(BE_XYZ)
    return "be.xyz"


def _e_tot(json_file):
    return json.loads(open(json_file).read())["e_tot"]["hartree"]


def test_occ_alpha_reproduces_mom(be_xyz):
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--method", "uhf",
                    "--mom", "--json", "mom.json"]) == 0
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--method", "uhf",
                    "--occ-alpha", "1,3", "--json", "occ.json"]) == 0
    # MOM's HOMO->LUMO promotion is exactly the alpha {1,3} configuration
    assert _e_tot("occ.json") == pytest.approx(_e_tot("mom.json"), abs=1e-6)


def test_occ_triplet_matches_aufbau(be_xyz):
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--spin", "2",
                    "--method", "uhf", "--json", "aufbau.json"]) == 0
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--spin", "2",
                    "--method", "uhf", "--occ-alpha", "1,2,3",
                    "--occ-beta", "1", "--json", "occ.json"]) == 0
    assert _e_tot("occ.json") == pytest.approx(_e_tot("aufbau.json"), abs=1e-6)


def test_occ_validation_errors(be_xyz):
    # wrong electron count for the channel
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--method", "uhf",
                    "--occ-alpha", "1"]) == 2
    # duplicate index
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--method", "uhf",
                    "--occ-alpha", "1,1"]) == 2
    # requires uhf
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g",
                    "--occ-alpha", "1,3"]) == 2
    # not combinable with --mom
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--method", "uhf",
                    "--mom", "--occ-alpha", "1,3"]) == 2
    # --zpe needs the ground state
    assert run_cli(["energy", be_xyz, "--basis", "sto-3g", "--method", "uhf",
                    "--mom", "--zpe"]) == 2


def test_fixed_occ_spin_state_estimates(be_xyz, capsys):
    """Be/cc-pVDZ 2s->2p: K splits the frozen-orbital singlet and triplet."""
    assert run_cli(["energy", be_xyz, "--basis", "cc-pvdz",
                    "--fixed-occ-decomp", "--json", "f.json"]) == 0
    out = capsys.readouterr().out
    assert "Triplet" in out and "Singlet" in out
    data = json.loads(open("f.json").read())
    assert data["fixed_occ_K_eV"] == pytest.approx(1.4696, abs=2e-3)
    assert data["fixed_occ_triplet_estimate_eV"] == pytest.approx(2.4127, abs=5e-3)
    assert data["fixed_occ_singlet_estimate_eV"] == pytest.approx(5.3519, abs=5e-3)
