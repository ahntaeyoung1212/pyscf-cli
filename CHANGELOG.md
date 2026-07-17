# Changelog

## 0.1.0 (unreleased)

First public release. Consolidates seven classroom scripts
(`calc_pyscf*.py`, Mochizuki group, Tokyo University of Science) into one
installable CLI.

### Features
- Subcommands: `energy`, `relax`, `vib`, `thermo`, `dos`, `orbitals`,
  `vibmovie`, `examples`, `info`
- `--dry-run`: print the equivalent runnable PySCF script (energy, relax,
  vib, thermo)
- `--json`: machine-readable output for every calculation (`-` = pure JSON
  on stdout); meaningful exit codes (0 ok / 2 input error / 3 SCF not
  converged)
- Explicit electron-configuration control: `--occ-alpha` / `--occ-beta`
  (MOM-constrained Δ-SCF) in addition to `--mom` single promotions
- `--fixed-occ-decomp` reports the open-shell exchange integral K and
  frozen-orbital singlet/triplet estimates
- Bundled sample molecules (`pyscf-cli examples`), curated basis/functional
  cheat sheet (`pyscf-cli info`), teaching-oriented error messages

### Fixes relative to the legacy scripts
- ZPE in `calc_pyscf.py --zpe` was overestimated by a factor of ~42.7
  (PySCF `freq_au` misread as Hartree); now computed from wavenumbers and
  consistent with `thermo`
- RHF `--fixed-occ-decomp` promoted-state energy was a fractional-occupation
  ensemble with self-interaction (~9.5 eV too high for H2O HOMO→LUMO);
  now the true frozen-orbital determinant
- Kohn-Sham calculations honor `--method` (RKS/ROKS/UKS); restricted
  open-shell Hessians are rejected up front (not implemented in PySCF)
- ROHF energy decomposition, imaginary-mode reporting, PES non-convergence
  warnings, Bohr-unit handling in `vibmovie`, and many smaller fixes
  (21 findings from a pre-release adversarial review)
