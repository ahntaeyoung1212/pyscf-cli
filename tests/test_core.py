"""Unit tests for pyscf_cli.core: parsing, validation, error messages."""

import pytest

from pyscf_cli import core


# ---------------------------------------------------------------------------
# read_xyz
# ---------------------------------------------------------------------------

def test_read_xyz_valid(tmp_path):
    f = tmp_path / "mol.xyz"
    f.write_text("2\ncomment\nH 0 0 0\nF 0 0 0.92\n")
    atoms = core.read_xyz(str(f))
    assert atoms == [("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 0.92)]


def test_read_xyz_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(core.InputError, match="not found"):
        core.read_xyz("nope.xyz")


def test_read_xyz_bad_atom_count_line(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("H 0 0 0\n")
    with pytest.raises(core.InputError, match="number of atoms"):
        core.read_xyz(str(f))


def test_read_xyz_too_few_atom_lines(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("3\ncomment\nH 0 0 0\n")
    with pytest.raises(core.InputError, match="declares 3 atoms"):
        core.read_xyz(str(f))


def test_read_xyz_non_numeric_coordinates(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("1\ncomment\nH a b c\n")
    with pytest.raises(core.InputError, match="numbers"):
        core.read_xyz(str(f))


def test_write_then_read_roundtrip(tmp_path):
    f = tmp_path / "out.xyz"
    atoms = [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692)]
    core.write_xyz(str(f), atoms, comment="test")
    back = core.read_xyz(str(f))
    for (e1, *xyz1), (e2, *xyz2) in zip(atoms, back):
        assert e1 == e2
        assert xyz1 == pytest.approx(xyz2, abs=1e-8)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_normalize_basis_known():
    assert core.normalize_basis("6-31G**") == "6-31g**"


def test_normalize_basis_typo_suggests():
    with pytest.raises(core.InputError, match="Did you mean '6-31g\\*'"):
        core.normalize_basis("6-31g*8")


def test_normalize_basis_unknown_passes_through(capsys):
    assert core.normalize_basis("ano-rcc") == "ano-rcc"
    assert "not in the common list" in capsys.readouterr().err


def test_normalize_xc_typo_suggests():
    with pytest.raises(core.InputError, match="Did you mean 'b3lyp'"):
        core.normalize_xc("b3lyq")


def test_resolve_method_auto():
    assert core.resolve_method("auto", 0) == "rhf"
    assert core.resolve_method("auto", 2) == "uhf"
    assert core.resolve_method("rohf", 2) == "rohf"


def test_build_mol_impossible_spin():
    atoms = [("O", 0.0, 0.0, 0.1173),
             ("H", 0.0, 0.7572, -0.4692),
             ("H", 0.0, -0.7572, -0.4692)]
    with pytest.raises(core.InputError, match="UNPAIRED"):
        core.build_mol(atoms, "sto-3g", spin=1)
