import pytest

from pyscf_cli import main as cli_main


def run_cli(argv):
    """Run pyscf-cli in-process and return its exit code."""
    return cli_main.main(argv)


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Run the test from an empty temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fetch_sample(name):
    assert cli_main.main(["examples", name]) == 0
    return f"{name}.xyz"


@pytest.fixture
def h2_xyz(in_tmp):
    return _fetch_sample("h2")


@pytest.fixture
def h2o_xyz(in_tmp):
    return _fetch_sample("h2o")


@pytest.fixture
def o2_xyz(in_tmp):
    return _fetch_sample("o2")
