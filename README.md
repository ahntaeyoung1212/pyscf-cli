# pyscf-cli

**Run quantum chemistry calculations from a single XYZ file and one command line.**

`pyscf-cli` is an educational front-end to [PySCF](https://pyscf.org/):
students run Hartree-Fock, DFT, MP2, and CCSD(T) calculations — plus geometry
optimization, vibrational analysis, thermochemistry, molecular DOS plots, and
orbital visualization — without writing any Python.

> **Note:** pyscf-cli is an independent educational project developed at the
> Mochizuki group, Tokyo University of Science. It is **not** an official tool
> of the PySCF developers. Please cite PySCF itself in any academic work.

## Status

🚧 **Under development** (Phase 1: core infrastructure). Not yet on PyPI.

## Quick taste

```bash
pyscf-cli energy h2o.xyz --basis 6-31g**              # single-point RHF
pyscf-cli energy o2.xyz --spin 2 --method uhf         # triplet O2, UHF
pyscf-cli relax h2o.xyz --theory dft --xc b3lyp       # geometry optimization
pyscf-cli vib h2o.xyz --basis 6-31g                   # vibrational frequencies
pyscf-cli dos benzene.xyz --element-pdos              # molecular DOS/PDOS plot
pyscf-cli info basis                                  # what basis sets can I use?
```

## Development install

```bash
git clone <repo-url>
cd pyscf-cli
pip install -e ".[dev]"
pyscf-cli --help
```

Requires Python ≥ 3.9 on Linux or macOS (on Windows, use WSL or Google Colab —
PySCF does not support Windows natively).

## License

MIT. PySCF itself is licensed under Apache-2.0.

If you use pyscf-cli in teaching or research, please cite PySCF:
> Q. Sun *et al.*, "Recent developments in the PySCF program package",
> *J. Chem. Phys.* **153**, 024109 (2020).
