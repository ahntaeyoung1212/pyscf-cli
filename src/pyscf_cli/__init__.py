"""pyscf-cli: educational command-line interface to PySCF."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pyscf-cli")
    except PackageNotFoundError:  # running from a source tree
        __version__ = "0.1.0.dev0"
except ImportError:  # pragma: no cover
    __version__ = "0.1.0.dev0"
