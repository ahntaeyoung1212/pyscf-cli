"""End-to-end smoke tests: every subcommand runs on a small molecule."""

import json
import os

import pytest

from conftest import run_cli


# ---------------------------------------------------------------------------
# Framework-level behavior
# ---------------------------------------------------------------------------

def test_no_command_shows_help(capsys):
    assert run_cli([]) == 0
    assert "energy" in capsys.readouterr().out


def test_info_runs(capsys):
    assert run_cli(["info"]) == 0
    out = capsys.readouterr().out
    assert "sto-3g" in out and "b3lyp" in out


def test_missing_file_exits_2(in_tmp):
    assert run_cli(["energy", "missing.xyz"]) == 2


def test_basis_typo_exits_2(h2o_xyz):
    assert run_cli(["energy", h2o_xyz, "--basis", "6-31g*8"]) == 2


def test_conflicting_xyz_args_exit_2(h2o_xyz):
    assert run_cli(["energy", h2o_xyz, "--xyz", "other.xyz"]) == 2


def test_examples_list_and_force(in_tmp, capsys):
    assert run_cli(["examples"]) == 0
    assert "h2o" in capsys.readouterr().out
    assert run_cli(["examples", "h2o"]) == 0
    assert run_cli(["examples", "h2o"]) == 0  # skip, no overwrite
    assert "skipped" in capsys.readouterr().out
    assert run_cli(["examples", "h2o", "--force"]) == 0
    assert run_cli(["examples", "nonexistent_molecule"]) == 2


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------

def test_energy_h2(h2_xyz, capsys):
    assert run_cli(["energy", h2_xyz]) == 0
    out = capsys.readouterr().out
    assert "Total Energy" in out and "eV" in out


def test_energy_json_output(h2_xyz):
    assert run_cli(["energy", h2_xyz, "--json", "result.json"]) == 0
    data = json.loads(open("result.json").read())
    assert data["e_tot"]["hartree"] == pytest.approx(-1.11675931, abs=1e-5)
    assert "eV" in data["e_tot"]


def test_energy_dry_run_generates_valid_python(h2o_xyz, capsys):
    assert run_cli(["energy", h2o_xyz, "--theory", "ccsd_t", "--dry-run"]) == 0
    script = capsys.readouterr().out
    compile(script, "generated.py", "exec")  # must be syntactically valid
    assert "gto.M" in script and "ccsd_t" in script.lower() or "CCSD" in script


def test_energy_pes(h2_xyz, capsys):
    assert run_cli(["energy", h2_xyz, "--pes",
                    "--rmin", "0.5", "--rmax", "1.1", "--npts", "3"]) == 0
    assert "PES scan" in capsys.readouterr().out


def test_energy_pes_rejects_polyatomic(h2o_xyz):
    assert run_cli(["energy", h2o_xyz, "--pes"]) == 2


def test_energy_decompose(h2_xyz, capsys):
    assert run_cli(["energy", h2_xyz, "--decompose-total-energy"]) == 0
    out = capsys.readouterr().out
    assert "Kinetic (T)" in out and "Sum" in out


def test_energy_decompose_rejects_dft(h2_xyz):
    assert run_cli(["energy", h2_xyz, "--theory", "dft",
                    "--decompose-total-energy"]) == 2


def test_energy_zpe(h2_xyz, capsys):
    assert run_cli(["energy", h2_xyz, "--zpe"]) == 0
    assert "ZPE" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# relax / vib / thermo
# ---------------------------------------------------------------------------

def test_relax_h2(h2_xyz, capsys):
    assert run_cli(["relax", h2_xyz]) == 0
    assert os.path.exists("h2-finish.xyz")
    assert os.path.exists("h2_relax.txt")
    assert "Final energy" in capsys.readouterr().out


def test_vib_h2o(h2o_xyz, capsys):
    assert run_cli(["vib", h2o_xyz, "--basis", "sto-3g", "--nmax", "1"]) == 0
    out = capsys.readouterr().out
    assert "vibrational" in out and "E_n (eV)" in out


def test_thermo_h2(h2_xyz, capsys):
    assert run_cli(["thermo", h2_xyz, "--basis", "sto-3g"]) == 0
    out = capsys.readouterr().out
    assert "G_tot" in out and "eV" in out


# ---------------------------------------------------------------------------
# dos / orbitals / vibmovie
# ---------------------------------------------------------------------------

def test_dos_h2o(h2o_xyz, capsys):
    assert run_cli(["dos", h2o_xyz, "--quiet"]) == 0
    for ext in ("pdf", "csv", "txt"):
        assert os.path.exists(f"DOS_h2o.{ext}")


def test_dos_uhf_spin_resolved(o2_xyz):
    assert run_cli(["dos", o2_xyz, "--spin", "2", "--method", "uhf",
                    "--quiet"]) == 0
    header = open("DOS_o2.csv").readline()
    assert "alpha_DOS" in header and "beta_DOS" in header


def test_orbitals_h2o(h2o_xyz, capsys):
    assert run_cli(["orbitals", h2o_xyz, "--homo", "--lumo",
                    "--nx", "20", "--ny", "20", "--nz", "20"]) == 0
    assert os.path.exists("h2o_orbitals/MO_005_HOMO.cube")
    assert os.path.exists("h2o_orbitals/MO_006_LUMO.cube")


def test_orbitals_requires_target(h2o_xyz):
    assert run_cli(["orbitals", h2o_xyz]) == 2


def test_vibmovie_h2(h2_xyz):
    assert run_cli(["vibmovie", h2_xyz, "--basis", "sto-3g",
                    "--nframes", "4", "--fps", "4"]) == 0
    assert os.path.exists("h2_vibmovie/mode001.gif")
    assert os.path.exists("h2_vibmovie/mode001.xyz")
