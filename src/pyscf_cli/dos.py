"""`pyscf-cli dos` — molecular DOS/PDOS plots from MO energies.

Port of calc_pyscf_dos.py.  MO energies are broadened (Gaussian or
histogram) into a molecular "density of states"; each MO is projected
onto AO angular-momentum channels (s/p/d/f), optionally element-resolved,
with Löwdin or Mulliken populations.  UHF gives spin-resolved plots
(beta plotted negative).
"""

from __future__ import annotations

import os

import numpy as np

from . import core
from .core import HARTREE_TO_EV, InputError
from .output import Report, use_headless_matplotlib

ANGULAR_CHANNELS = ("s", "p", "d", "f")
VESTA_FALLBACK_COLORS = {
    "H": "#ffcccc",
    "C": "#4c4c4c",
    "N": "#b0b9e6",
    "O": "#fe0300",
    "F": "#b0b9e6",
    "P": "#c09cc2",
    "S": "#fffa00",
    "Cl": "#31fc02",
    "Br": "#7e3102",
    "I": "#940094",
}


def register(subparsers):
    parser = subparsers.add_parser(
        "dos",
        help="molecular DOS/PDOS plot from broadened MO energies",
        description=(
            "Broaden the MO spectrum into a molecular DOS and project each MO "
            "onto AO angular-momentum channels (and optionally elements). "
            "Writes a plot, a CSV of all curves, and a text summary."
        ),
    )
    core.add_common_arguments(parser)
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Gaussian broadening width in eV (default: 0.3)")
    parser.add_argument("--binwidth", "--bin-width", dest="binwidth",
                        type=float, default=None,
                        help="use a histogram DOS with this bin width in eV "
                             "instead of Gaussian broadening")
    parser.add_argument("--npts", type=int, default=2000,
                        help="number of energy grid points (default: 2000)")
    parser.add_argument("--padding", type=float, default=3.0,
                        help="energy padding in eV (default: 3.0)")
    parser.add_argument("--emin", type=float, default=None,
                        help="minimum plotted energy in eV")
    parser.add_argument("--emax", type=float, default=None,
                        help="maximum plotted energy in eV")
    parser.add_argument("--xrange", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                        default=None, help="x-axis range in eV, e.g. --xrange -10 10")
    parser.add_argument("--yrange", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                        default=None, help="y-axis range in states/eV")
    parser.add_argument("--align", default="absolute", choices=["homo", "absolute"],
                        help="energy reference (default: absolute)")
    parser.add_argument("--projection", default="lowdin",
                        choices=["lowdin", "mulliken"],
                        help="AO population analysis for PDOS weights (default: lowdin)")
    parser.add_argument("--no-spin-degeneracy", dest="spin_degeneracy",
                        action="store_false",
                        help="do not count restricted MOs as doubly degenerate")
    parser.set_defaults(spin_degeneracy=True)
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="plot file (default: DOS_<input>.pdf)")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="CSV file (default: DOS_<input>.csv)")
    parser.add_argument("--txt", default=None, metavar="FILE",
                        help="text summary file (default: DOS_<input>.txt)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="raster output resolution (default: 200)")
    parser.add_argument("--element-pdos", action="store_true",
                        help="element- and angular-momentum-resolved PDOS "
                             "with VESTA element colors")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the MO weight table")
    parser.set_defaults(func=run)
    return parser


# ---------------------------------------------------------------------------
# MO channels and AO groupings (identical to legacy)
# ---------------------------------------------------------------------------

def split_mo_channels(mf):
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = getattr(mf, "mo_occ", None)

    if isinstance(mo_coeff, (tuple, list)):
        labels = ("alpha", "beta")
        return [
            {
                "label": labels[i],
                "energy": np.asarray(mo_energy[i]),
                "coeff": np.asarray(mo_coeff[i]),
                "occ": None if mo_occ is None else np.asarray(mo_occ[i]),
                "state_factor": 1.0,
            }
            for i in range(len(mo_coeff))
        ]

    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)

    if mo_energy.ndim == 2 and mo_coeff.ndim == 3 and mo_energy.shape[0] == 2:
        labels = ("alpha", "beta")
        return [
            {
                "label": labels[i],
                "energy": mo_energy[i],
                "coeff": mo_coeff[i],
                "occ": None if mo_occ is None else np.asarray(mo_occ[i]),
                "state_factor": 1.0,
            }
            for i in range(2)
        ]

    return [
        {
            "label": "restricted",
            "energy": mo_energy,
            "coeff": mo_coeff,
            "occ": None if mo_occ is None else np.asarray(mo_occ),
            "state_factor": 2.0,
        }
    ]


def ao_angular_groups(mol):
    groups = {channel: [] for channel in ANGULAR_CHANNELS}
    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        shell_name = label[2]
        angular = next((c for c in shell_name if c in ANGULAR_CHANNELS), None)
        if angular is not None:
            groups[angular].append(ao_index)
    return {ch: np.asarray(idx, dtype=int) for ch, idx in groups.items()}


def ao_element_groups(mol):
    groups = {}
    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        element = mol.atom_symbol(label[0])
        groups.setdefault(element, []).append(ao_index)
    return {
        element: np.asarray(indices, dtype=int)
        for element, indices in sorted(groups.items())
    }


def ao_element_angular_groups(mol):
    groups = {}
    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        element = mol.atom_symbol(label[0])
        shell_name = label[2]
        angular = next((c for c in shell_name if c in ANGULAR_CHANNELS), None)
        if angular is None:
            continue
        groups.setdefault((element, angular), []).append(ao_index)

    angular_order = {a: i for i, a in enumerate(ANGULAR_CHANNELS)}
    ordered = sorted(
        groups,
        key=lambda k: (k[0], angular_order.get(k[1], len(ANGULAR_CHANNELS))),
    )
    return {key: np.asarray(groups[key], dtype=int) for key in ordered}


def lowdin_projector(mol):
    overlap = mol.intor_symmetric("int1e_ovlp")
    eigval, eigvec = np.linalg.eigh(overlap)
    eigval = np.clip(eigval, 0.0, None)
    return (eigvec * np.sqrt(eigval)) @ eigvec.T


def mo_group_weights(mol, coeff, groups, projection):
    """Per-MO weights for arbitrary AO index groups (Löwdin or Mulliken)."""
    if projection == "lowdin":
        projected = lowdin_projector(mol) @ coeff
        return {
            group: np.sum(np.abs(projected[indices, :]) ** 2, axis=0)
            if len(indices) > 0 else np.zeros(coeff.shape[1])
            for group, indices in groups.items()
        }

    overlap_coeff = mol.intor_symmetric("int1e_ovlp") @ coeff
    return {
        group: np.einsum(
            "mi,mi->i",
            np.conjugate(coeff[indices, :]),
            overlap_coeff[indices, :],
        ).real
        if len(indices) > 0 else np.zeros(coeff.shape[1])
        for group, indices in groups.items()
    }


def vesta_element_colors(elements):
    import matplotlib.pyplot as plt

    try:
        from pymatgen.vis.structure_vtk import EL_COLORS

        color_table = EL_COLORS.get("VESTA", {})
        colors = {}
        for element in elements:
            rgb = color_table.get(element)
            colors[element] = (
                None if rgb is None else tuple(ch / 255.0 for ch in rgb)
            )
        if any(color is not None for color in colors.values()):
            fallback = plt.get_cmap("tab20")
            for i, element in enumerate(elements):
                if colors[element] is None:
                    colors[element] = fallback(i % fallback.N)
            return colors
    except Exception:
        pass

    fallback = plt.get_cmap("tab20")
    return {
        element: VESTA_FALLBACK_COLORS.get(element, fallback(i % fallback.N))
        for i, element in enumerate(elements)
    }


# ---------------------------------------------------------------------------
# DOS construction (identical math to legacy)
# ---------------------------------------------------------------------------

def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def histogram_edges(emin, emax, binwidth):
    n_bins = max(1, int(np.ceil((emax - emin) / binwidth - 1.0e-12)))
    edges = emin + np.arange(n_bins + 1) * binwidth
    edges[-1] = emax
    return edges


def homo_energy_ev(channels):
    occupied_energies = []
    for channel in channels:
        if channel["occ"] is None:
            continue
        occupied = channel["energy"][channel["occ"] > 1.0e-8]
        if occupied.size:
            occupied_energies.append(np.max(occupied))
    if not occupied_energies:
        return 0.0
    return max(occupied_energies) * HARTREE_TO_EV


def build_dos(mf, args):
    channels = split_mo_channels(mf)
    groups = ao_angular_groups(mf.mol)
    element_groups = (
        ao_element_angular_groups(mf.mol) if args.element_pdos
        else ao_element_groups(mf.mol)
    )
    element_labels = list(element_groups)

    reference_ev = homo_energy_ev(channels) if args.align == "homo" else 0.0
    all_energies = np.concatenate([channel["energy"] for channel in channels])
    all_energies_ev = all_energies * HARTREE_TO_EV - reference_ev

    use_histogram = args.binwidth is not None
    padding = args.padding if use_histogram else max(args.padding, 6.0 * args.sigma)
    emin = np.min(all_energies_ev) - padding if args.emin is None else args.emin
    emax = np.max(all_energies_ev) + padding if args.emax is None else args.emax

    if use_histogram:
        bin_edges = histogram_edges(emin, emax, args.binwidth)
        bin_widths = np.diff(bin_edges)
        grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        bin_edges = None
        bin_widths = None
        grid = np.linspace(emin, emax, args.npts)

    dos = np.zeros_like(grid)
    pdos = {channel: np.zeros_like(grid) for channel in ANGULAR_CHANNELS}
    element_pdos = {element: np.zeros_like(grid) for element in element_labels}
    spin_dos = {channel["label"]: np.zeros_like(grid) for channel in channels}
    spin_pdos = {
        channel["label"]: {a: np.zeros_like(grid) for a in ANGULAR_CHANNELS}
        for channel in channels
    }
    spin_element_pdos = {
        channel["label"]: {e: np.zeros_like(grid) for e in element_labels}
        for channel in channels
    }
    mo_rows = []

    for channel in channels:
        spin_label = channel["label"]
        weights = mo_group_weights(mf.mol, channel["coeff"], groups, args.projection)
        element_weights = mo_group_weights(
            mf.mol, channel["coeff"], element_groups, args.projection
        )
        state_factor = channel["state_factor"] if args.spin_degeneracy else 1.0
        energies_ev = channel["energy"] * HARTREE_TO_EV - reference_ev

        if use_histogram:
            total_weights = np.full_like(energies_ev, state_factor, dtype=float)
            hist, _ = np.histogram(energies_ev, bins=bin_edges, weights=total_weights)
            channel_dos = hist / bin_widths
            dos += channel_dos
            spin_dos[spin_label] += channel_dos

            for angular in ANGULAR_CHANNELS:
                hist, _ = np.histogram(
                    energies_ev, bins=bin_edges,
                    weights=state_factor * weights[angular],
                )
                channel_pdos = hist / bin_widths
                pdos[angular] += channel_pdos
                spin_pdos[spin_label][angular] += channel_pdos

            for element in element_labels:
                hist, _ = np.histogram(
                    energies_ev, bins=bin_edges,
                    weights=state_factor * element_weights[element],
                )
                channel_pdos = hist / bin_widths
                element_pdos[element] += channel_pdos
                spin_element_pdos[spin_label][element] += channel_pdos

        for mo_index, energy_ev in enumerate(energies_ev):
            if not use_histogram:
                broadening = gaussian(grid, energy_ev, args.sigma)
                channel_dos = state_factor * broadening
                dos += channel_dos
                spin_dos[spin_label] += channel_dos
                for angular in ANGULAR_CHANNELS:
                    channel_pdos = state_factor * weights[angular][mo_index] * broadening
                    pdos[angular] += channel_pdos
                    spin_pdos[spin_label][angular] += channel_pdos
                for element in element_labels:
                    channel_pdos = (
                        state_factor * element_weights[element][mo_index] * broadening
                    )
                    element_pdos[element] += channel_pdos
                    spin_element_pdos[spin_label][element] += channel_pdos

            occ = None if channel["occ"] is None else channel["occ"][mo_index]
            mo_rows.append(
                {
                    "spin": channel["label"],
                    "mo": mo_index + 1,
                    "energy_ev": energy_ev,
                    "occ": occ,
                    "weights": {a: weights[a][mo_index] for a in ANGULAR_CHANNELS},
                }
            )

    return (
        grid, dos, pdos, element_pdos, spin_dos, spin_pdos, spin_element_pdos,
        mo_rows, reference_ev, bin_widths, element_labels,
    )


# ---------------------------------------------------------------------------
# Plotting and CSV (identical layout to legacy)
# ---------------------------------------------------------------------------

def plot_dos(grid, dos, pdos, element_pdos, spin_dos, spin_pdos,
             spin_element_pdos, element_labels, args, title, bin_widths, method):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    colors = {"s": "#1f77b4", "p": "#d62728", "d": "#2ca02c", "f": "#9467bd"}
    linestyles = {"s": "-", "p": "--", "d": "-.", "f": ":"}

    if args.element_pdos:
        unique_elements = []
        for element, _angular in element_labels:
            if element not in unique_elements:
                unique_elements.append(element)
    else:
        unique_elements = list(element_labels)
    element_colors = vesta_element_colors(unique_elements)
    spin_resolved = method == "uhf" and {"alpha", "beta"}.issubset(spin_dos)

    if spin_resolved:
        spin_signs = {"alpha": 1.0, "beta": -1.0}

        for spin_label in ("alpha", "beta"):
            signed_dos = spin_signs[spin_label] * spin_dos[spin_label]
            if args.binwidth is None:
                ax.plot(grid, signed_dos, color="black", lw=1.8,
                        ls="-" if spin_label == "alpha" else "--",
                        label=f"{spin_label} total")
            else:
                ax.bar(grid, signed_dos, width=0.92 * bin_widths, align="center",
                       color="0.82" if spin_label == "alpha" else "0.72",
                       edgecolor="black", linewidth=0.6,
                       label=f"{spin_label} total")

            if args.element_pdos:
                for element, angular in element_labels:
                    signed_pdos = (
                        spin_signs[spin_label]
                        * spin_element_pdos[spin_label][(element, angular)]
                    )
                    if np.max(np.abs(signed_pdos)) <= 1.0e-10:
                        continue
                    plot_kwargs = dict(
                        color=element_colors.get(element), ls=linestyles[angular],
                        lw=1.2, alpha=0.8 if spin_label == "alpha" else 0.55,
                        label=f"{spin_label} {element} {angular}-PDOS",
                    )
                    if args.binwidth is None:
                        ax.plot(grid, signed_pdos, **plot_kwargs)
                    else:
                        ax.step(grid, signed_pdos, where="mid", **plot_kwargs)
            else:
                for angular in ANGULAR_CHANNELS:
                    signed_pdos = (
                        spin_signs[spin_label] * spin_pdos[spin_label][angular]
                    )
                    if np.max(np.abs(signed_pdos)) <= 1.0e-10:
                        continue
                    plot_kwargs = dict(
                        color=colors[angular], ls=linestyles[angular], lw=1.2,
                        alpha=1.0 if spin_label == "alpha" else 0.72,
                        label=f"{spin_label} {angular}-PDOS",
                    )
                    if args.binwidth is None:
                        ax.plot(grid, signed_pdos, **plot_kwargs)
                    else:
                        ax.step(grid, signed_pdos, where="mid", **plot_kwargs)
        ax.axhline(0.0, color="0.55", lw=0.8)
    elif args.binwidth is None:
        ax.plot(grid, dos, color="black", lw=1.8, label="total")
    else:
        ax.bar(grid, dos, width=0.92 * bin_widths, align="center", color="0.82",
               edgecolor="black", linewidth=0.6, label="total")

    if not spin_resolved and not args.element_pdos:
        for angular in ANGULAR_CHANNELS:
            if np.max(np.abs(pdos[angular])) > 1.0e-10:
                plot_kwargs = dict(color=colors[angular], ls=linestyles[angular],
                                   lw=1.4, label=f"{angular}-PDOS")
                if args.binwidth is None:
                    ax.plot(grid, pdos[angular], **plot_kwargs)
                else:
                    ax.step(grid, pdos[angular], where="mid", **plot_kwargs)

    if not spin_resolved and args.element_pdos:
        for element, angular in element_labels:
            if np.max(np.abs(element_pdos[(element, angular)])) > 1.0e-10:
                plot_kwargs = dict(color=element_colors.get(element),
                                   ls=linestyles[angular], lw=1.4, alpha=0.85,
                                   label=f"{element} {angular}-PDOS")
                if args.binwidth is None:
                    ax.plot(grid, element_pdos[(element, angular)], **plot_kwargs)
                else:
                    ax.step(grid, element_pdos[(element, angular)], where="mid",
                            **plot_kwargs)

    if args.align == "homo":
        ax.axvline(0.0, color="0.55", lw=0.9, ls=":")
        ax.set_xlabel("Energy - HOMO (eV)")
    else:
        ax.set_xlabel("Orbital energy (eV)")

    if args.xrange is not None:
        ax.set_xlim(args.xrange)
    if args.yrange is not None:
        ax.set_ylim(args.yrange)

    if spin_resolved:
        ax.set_ylabel("DOS (states/eV; beta plotted negative)")
    else:
        ax.set_ylabel("DOS (states/eV)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)


def write_csv(grid, dos, pdos, element_pdos, spin_dos, spin_pdos,
              spin_element_pdos, element_labels, csv_file, method,
              include_element_pdos):
    columns = [grid, dos]
    headers = ["energy_eV", "total_DOS"]

    if not include_element_pdos:
        columns.extend(pdos[channel] for channel in ANGULAR_CHANNELS)
        headers.extend(["s_PDOS", "p_PDOS", "d_PDOS", "f_PDOS"])
    else:
        for element, angular in element_labels:
            columns.append(element_pdos[(element, angular)])
            headers.append(f"{element}_{angular}_PDOS")

    if method == "uhf" and {"alpha", "beta"}.issubset(spin_dos):
        for spin_label in ("alpha", "beta"):
            columns.append(spin_dos[spin_label])
            headers.append(f"{spin_label}_DOS")
            if include_element_pdos:
                for element, angular in element_labels:
                    columns.append(spin_element_pdos[spin_label][(element, angular)])
                    headers.append(f"{spin_label}_{element}_{angular}_PDOS")
            else:
                for angular in ANGULAR_CHANNELS:
                    columns.append(spin_pdos[spin_label][angular])
                    headers.append(f"{spin_label}_{angular}_PDOS")

    data = np.column_stack(columns)
    np.savetxt(csv_file, data, delimiter=",", header=",".join(headers), comments="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args):
    core.finalize_common_args(args)
    use_headless_matplotlib()

    if args.binwidth is not None and args.binwidth <= 0.0:
        raise InputError("--binwidth must be positive")
    if args.xrange is not None:
        if args.emin is not None or args.emax is not None:
            raise InputError("--xrange cannot be used together with --emin or --emax")
        args.emin, args.emax = args.xrange
    if args.emin is not None and args.emax is not None and args.emin >= args.emax:
        raise InputError("energy range must satisfy minimum < maximum")
    if args.yrange is not None and args.yrange[0] >= args.yrange[1]:
        raise InputError("--yrange must satisfy YMIN < YMAX")

    input_root = os.path.splitext(os.path.basename(args.xyz))[0]
    output_file = args.output or f"DOS_{input_root}.pdf"
    root, _ = os.path.splitext(output_file)
    csv_file = args.csv or root + ".csv"
    txt_file = args.txt or root + ".txt"
    args.output = output_file

    mol = core.build_mol(args.atoms, args.basis, args.charge, args.spin, args.unit)
    mf, e_tot, info = core.run_theory(mol, args.theory, args.method, args.xc)

    (grid, dos, pdos, element_pdos, spin_dos, spin_pdos, spin_element_pdos,
     mo_rows, reference_ev, bin_widths, element_labels) = build_dos(mf, args)

    title = f"{os.path.basename(args.xyz)}  {args.theory.upper()}/{args.basis}"
    plot_dos(grid, dos, pdos, element_pdos, spin_dos, spin_pdos,
             spin_element_pdos, element_labels, args, title, bin_widths,
             args.method)
    write_csv(grid, dos, pdos, element_pdos, spin_dos, spin_pdos,
              spin_element_pdos, element_labels, csv_file, args.method,
              args.element_pdos)

    r = Report("PySCF Molecular DOS (pyscf-cli dos)")
    r.kv("XYZ file", args.xyz, key="xyz")
    r.kv("Method", args.method.upper(), key="method")
    r.kv("Theory", args.theory, key="theory")
    if args.theory == "dft":
        r.kv("XC functional", args.xc, key="xc")
    r.kv("Basis", args.basis, key="basis")
    r.energy("Total Energy", e_tot, key="e_tot")
    r.kv("Charge", args.charge, key="charge")
    r.kv("Spin (2S)", args.spin, key="spin")
    r.kv("Projection", args.projection, key="projection")
    if args.binwidth is None:
        r.kv("DOS mode", "gaussian broadening", key="dos_mode")
        r.kv("Gaussian sigma", f"{args.sigma:.3f} eV", key="sigma_eV")
    else:
        r.kv("DOS mode", "histogram", key="dos_mode")
        r.kv("Bin width", f"{args.binwidth:.3f} eV", key="binwidth_eV")
    if args.align == "homo":
        r.kv("Energy zero", f"HOMO = {reference_ev:.6f} eV", key="reference_eV")
    else:
        r.kv("Energy zero", "absolute MO energy", key="reference_eV")
    r.kv("Plot output", args.output, key="plot_file")
    r.kv("CSV output", csv_file, key="csv_file")
    r.kv("Text output", txt_file, key="txt_file")
    if args.element_pdos:
        labels = [f"{element}({angular})" for element, angular in element_labels]
        r.kv("Element PDOS", ", ".join(labels))

    if not args.quiet:
        energy_header = "E-HOMO(eV)" if args.align == "homo" else "energy(eV)"
        r.line()
        r.line("MO projected weights:")
        r.line(f"spin         MO    {energy_header:>10s}    occ"
               "        s        p        d        f")
        r.line("-" * 74)
        for row in mo_rows:
            occ_text = "NA" if row["occ"] is None else f"{row['occ']:.3g}"
            r.line(
                f"{row['spin']:<10s} {row['mo']:3d} "
                f"{row['energy_ev']:12.5f} {occ_text:>6s} "
                f"{row['weights']['s']:8.4f} "
                f"{row['weights']['p']:8.4f} "
                f"{row['weights']['d']:8.4f} "
                f"{row['weights']['f']:8.4f}"
            )

    r.add("mo_rows", [
        {
            "spin": row["spin"],
            "mo": row["mo"],
            "energy_eV": float(row["energy_ev"]),
            "occ": None if row["occ"] is None else float(row["occ"]),
            "weights": {a: float(w) for a, w in row["weights"].items()},
        }
        for row in mo_rows
    ])
    r.rule("=")
    r.emit(txt_path=txt_file, json_target=args.json)
    return 0
