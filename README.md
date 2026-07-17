# pyscf-cli

**Run real quantum chemistry from a single XYZ file and one command line.**

[![CI](https://github.com/mochizuki-group/pyscf-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/mochizuki-group/pyscf-cli/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mochizuki-group/pyscf-cli/blob/main/examples/colab_quickstart.ipynb)

`pyscf-cli` is an educational front-end to [PySCF](https://pyscf.org/).
Students run Hartree–Fock, DFT, MP2, and CCSD(T) calculations — plus geometry
optimization, vibrational analysis, thermochemistry, molecular DOS plots,
orbital visualization, and Δ-SCF excited states — without writing any Python.
When they are ready to look under the hood, `--dry-run` prints the equivalent
PySCF script for any calculation.

> **Note:** pyscf-cli is an independent educational project developed at the
> Mochizuki group, Tokyo University of Science. It is **not** an official tool
> of the PySCF developers. Please cite PySCF itself in any academic work
> (see [Citation](#citation)).

日本語のREADMEは [README_ja.md](README_ja.md) をご覧ください。

## Installation

```bash
pip install pyscf-cli        # (after the PyPI release)
# development version:
pip install git+https://github.com/mochizuki-group/pyscf-cli.git
```

Requires Python ≥ 3.9 on **Linux or macOS**. PySCF does not support Windows
natively — Windows users should use WSL or simply run everything on
[Google Colab](https://colab.research.google.com/github/mochizuki-group/pyscf-cli/blob/main/examples/colab_quickstart.ipynb)
(`%pip install pyscf-cli` works there).

## Five-minute quickstart

```bash
pyscf-cli examples h2o          # copy a bundled sample molecule to ./h2o.xyz
pyscf-cli energy h2o.xyz        # RHF/STO-3G single point
```

```text
==============================================
 PySCF Calculation (pyscf-cli energy)
==============================================
XYZ file      : h2o.xyz
Basis         : sto-3g
...
Method        : RHF
Total Energy  : -2039.847777 eV
<S^2>         : 0.000000
Multiplicity  : 1.0

MO energies (Hartree / eV):
  MO  1: -20.242078  -550.8151
  ...
```

Not sure what to try? `pyscf-cli examples` lists the bundled molecules,
`pyscf-cli info` explains every basis set, functional, and level of theory,
and every subcommand has `--help`.

## What can it do?

| Command | What you get |
|---|---|
| `energy` | total energy (HF/DFT/MP2/CCSD/CCSD(T)), MO levels, ⟨S²⟩, diatomic PES scans, energy decompositions, Δ-SCF excited states |
| `relax` | geometry optimization (geomeTRIC); writes `<input>-finish.xyz` |
| `vib` | harmonic frequencies, imaginary-mode detection, quantized levels E_n |
| `thermo` | ZPE, E/H/G/S/Cp at chosen T and p |
| `dos` | molecular DOS/PDOS plots (s/p/d/f- and element-resolved, Löwdin/Mulliken, spin-resolved) |
| `orbitals` | cube files of MOs (HOMO/LUMO/any) for VESTA/Avogadro |
| `vibmovie` | animated GIF of each normal mode |
| `examples` | bundled sample molecules (H₂, H₂O, O₂, CO₂, NH₃, CH₄, benzene) |
| `info` | curated basis-set / functional cheat sheet |

A tour:

```bash
pyscf-cli energy o2.xyz --spin 2 --method uhf              # triplet O2 (check <S^2> = 2!)
pyscf-cli energy h2o.xyz --theory dft --xc b3lyp --basis 6-31g**
pyscf-cli energy h2.xyz --pes --rmin 0.4 --rmax 3.0        # bond dissociation curve
pyscf-cli relax h2o.xyz --basis 6-31g
pyscf-cli vib h2o.xyz --basis 6-31g
pyscf-cli thermo h2o.xyz --basis 6-31g --temp 298.15
pyscf-cli dos benzene.xyz --element-pdos --align homo
pyscf-cli orbitals h2o.xyz --homo --lumo                   # cube files for VESTA
pyscf-cli vibmovie h2o.xyz --basis 6-31g                   # GIF per normal mode
```

## Designed for teaching

- **Helpful errors.** Typos suggest fixes (`Unknown basis '6-31g*8'. Did you
  mean '6-31g*'?`); impossible charge/spin combinations explain what `--spin`
  means; saddle-point geometries trigger "this is not a minimum" warnings
  instead of silent nonsense.
- **`--dry-run` is the bridge to real PySCF.** It prints a runnable Python
  script equivalent to the CLI call — students go CLI → read the script →
  edit the script → graduate from the CLI.
- **`--json` for auto-grading.** Every calculation can emit machine-readable
  results (`--json result.json`, or `--json -` for pure JSON on stdout).
  Exit codes are meaningful: 0 = success, 2 = input error, 3 = SCF did not
  converge.
- **Electron-configuration control.** `--spin` sets the number of unpaired
  electrons; `--occ-alpha 1,3` occupies arbitrary MOs per spin channel and
  holds that configuration during the SCF (maximum overlap method); the
  printed ⟨S²⟩ verifies what you actually converged to.
- **Energy decompositions.** `--decompose-total-energy` splits E into
  T + V_ne + U + J + V_nn (the sum is verified to equal E_SCF);
  `--fixed-occ-decomp` analyzes a frozen-orbital promotion and prints the
  exchange integral K with the resulting singlet/triplet estimates.

See [docs/TEACHING_ja.md](docs/TEACHING_ja.md) for ready-to-use classroom
exercises (in Japanese).

## Citation

If you use pyscf-cli in teaching or research, please cite **PySCF**, which
does all the actual quantum chemistry:

> Q. Sun *et al.*, "Recent developments in the PySCF program package",
> *J. Chem. Phys.* **153**, 024109 (2020). DOI: 10.1063/5.0006074

A citable DOI for pyscf-cli itself (Zenodo) will be provided with the first
stable release.

## License

MIT © 2026 Yasuhide Mochizuki, Tokyo University of Science.
PySCF itself is licensed under Apache-2.0.
