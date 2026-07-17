"""Tests for course-derived features: convert (SDF->XYZ), COOP/COHP,
honest ROHF labeling, and bundled atom samples."""

import os

import pytest

from conftest import run_cli
from pyscf_cli import core

TINY_SDF = """water
  course-test

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END
$$$$
"""


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def test_convert_sdf_to_xyz(in_tmp):
    (in_tmp / "SDF_water.sdf").write_text(TINY_SDF)
    assert run_cli(["convert", "SDF_water.sdf"]) == 0
    # course naming convention: SDF_name.sdf -> XYZ_name.xyz
    atoms = core.read_xyz("XYZ_water.xyz")
    assert [a[0] for a in atoms] == ["O", "H", "H"]
    assert atoms[1][2] == pytest.approx(0.7572)


def test_convert_explicit_output(in_tmp):
    (in_tmp / "mol.sdf").write_text(TINY_SDF)
    assert run_cli(["convert", "mol.sdf", "-o", "out.xyz"]) == 0
    assert os.path.exists("out.xyz")


def test_convert_rejects_garbage(in_tmp):
    (in_tmp / "bad.sdf").write_text("not\nan\nsdf\nfile at all\n")
    assert run_cli(["convert", "bad.sdf"]) == 2
    assert run_cli(["convert", "missing.sdf"]) == 2


# ---------------------------------------------------------------------------
# COOP/COHP
# ---------------------------------------------------------------------------

def test_dos_coop_cohp(h2o_xyz, capsys):
    assert run_cli(["dos", h2o_xyz, "--basis", "sto-3g",
                    "--coop", "--cohp", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "ICOOP(O,H)" in out and "ICOHP(O,H)" in out
    assert os.path.exists("OP_h2o_O_H.pdf")
    assert os.path.exists("HP_h2o_O_H.pdf")


def test_dos_coop_json_values(h2o_xyz):
    import json
    assert run_cli(["dos", h2o_xyz, "--basis", "sto-3g", "--coop",
                    "--quiet", "--json", "d.json"]) == 0
    data = json.loads(open("d.json").read())
    icoop = data["icoop"]["O,H"]
    assert 0.0 < icoop < 2.0  # bonding overall for water O-H


def test_dos_pair_validation(h2o_xyz):
    # element not in molecule
    assert run_cli(["dos", h2o_xyz, "--coop", "--pair", "C,O",
                    "--quiet"]) == 2
    # --pair without --coop/--cohp
    assert run_cli(["dos", h2o_xyz, "--pair", "O,H", "--quiet"]) == 2
    # malformed pair
    assert run_cli(["dos", h2o_xyz, "--coop", "--pair", "OH",
                    "--quiet"]) == 2


# ---------------------------------------------------------------------------
# Honest reference labels
# ---------------------------------------------------------------------------

def test_open_shell_rhf_is_labeled_rohf(in_tmp, capsys):
    (in_tmp / "oh.xyz").write_text("2\nOH radical\nO 0 0 0\nH 0 0 0.97\n")
    assert run_cli(["energy", "oh.xyz", "--basis", "sto-3g",
                    "--spin", "1", "--method", "rhf"]) == 0
    out = capsys.readouterr().out
    # PySCF promotes RHF to ROHF on open shells; the label must say so
    assert "Method        : ROHF" in out


# ---------------------------------------------------------------------------
# Bundled atom samples
# ---------------------------------------------------------------------------

def test_atom_samples_bundled(in_tmp, capsys):
    assert run_cli(["examples"]) == 0
    out = capsys.readouterr().out
    for name in ("li", "c", "n", "o", "f", "ne", "na", "cl"):
        assert f"\n  {name} " in out or f" {name}  " in out
