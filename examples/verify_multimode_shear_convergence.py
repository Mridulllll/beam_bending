#!/usr/bin/env python3

"""
Modal convergence study for multimode shear buckling.

Two cases are evaluated:

1. homogeneous isotropic AlSi10Mg plate
2. homogenized Gyroid RD12 sandwich

For each retained Navier basis size M x N, solve pure in-plane shear:

    Nx_ref  = 0
    Ny_ref  = 0
    Nxy_ref = 1

and report:

    - critical shear resultant Nxy_cr
    - dominant mode
    - relative change from previous basis
    - dominant eigenvector participation

The purpose is to establish a converged modal truncation before
using the multimode formulation for TPMS buckling interaction studies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
)

from host.generalized_stiffness import (
    Layer,
    assemble_generalized_stiffness,
)

from host.hsdt import (
    extract_hsdt_stiffness,
)

from host.multimode_buckling import (
    solve_multimode_buckling,
)

from host.sandwich import (
    build_three_layer_sandwich,
)

from host.sandwich_warping import (
    material_informed_generalized_warping,
)

from host.tpms_core import (
    representative_transverse_shear_modulus,
)

from host.tpms_io import (
    load_tpms_homogenization_json,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


QUERY_DIR = Path(
    r"C:\AbaqusScratch\UCs\Shells\TPMS_mapped_thickness"
    r"\_stiffness_database\queries"
)

GYROID_JSON = (
    QUERY_DIR
    / "sizing_shape_Gyroid_rd12_compression_itag00.json"
)


def build_homogeneous_plate():
    E = 70000.0
    nu = 0.33

    h = 1.0

    C = sigma_zz_zero_stiffness(
        isotropic_stiffness_3d(
            E=E,
            nu=nu,
        )
    )

    K14 = assemble_generalized_stiffness(
        [
            Layer(
                "homogeneous_plate",
                -0.5 * h,
                +0.5 * h,
                C,
            )
        ],
        reddy_baseline_generalized_warping(
            h
        ),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(
        K14
    )


def build_gyroid_sandwich():
    record = load_tpms_homogenization_json(
        GYROID_JSON
    )

    E_face = 70000.0
    nu_face = 0.33

    hf = 0.5
    hc = 10.0

    h = (
        2.0 * hf
        + hc
    )

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

    G_face = (
        E_face
        / (
            2.0
            * (1.0 + nu_face)
        )
    )

    G_core = (
        representative_transverse_shear_modulus(
            record.core,
            relative_tolerance=0.05,
        )
    )

    warping = (
        material_informed_generalized_warping(
            h=h,
            hc=hc,
            G_face=G_face,
            G_core=G_core,
        )
    )

    sandwich = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=record.C_hom,
        C_top=C_face,
        hf_bottom=hf,
        hc=hc,
        hf_top=hf,
        warping=warping,
    )

    return sandwich.K11


def dominant_participation(
    result,
    count=5,
):
    amplitudes = np.abs(
        result.eigenvector
    )

    order = np.argsort(
        amplitudes
    )[::-1]

    entries = []

    for index in order[:count]:
        mode = result.modes[
            int(index)
        ]

        entries.append(
            (
                mode.m,
                mode.n,
                float(
                    amplitudes[index]
                ),
            )
        )

    return entries


def run_convergence_case(
    name,
    K_hsdt,
    a,
    b,
    basis_sizes,
):
    rows = []

    previous = None

    print()
    print("=" * 88)
    print(name)
    print("=" * 88)

    print(
        "{:>8s} {:>16s} {:>16s} {:>14s} {:>24s}".format(
            "basis",
            "Nxy_cr [N/mm]",
            "rel.change [%]",
            "dominant",
            "top participation",
        )
    )

    print("-" * 88)

    for size in basis_sizes:
        result = solve_multimode_buckling(
            K_hsdt,
            a=a,
            b=b,
            Nx_ref=0.0,
            Ny_ref=0.0,
            Nxy_ref=1.0,
            max_m=size,
            max_n=size,
        )

        value = float(
            result.load_factor
        )

        if previous is None:
            rel_change = np.nan
        else:
            rel_change = (
                abs(
                    value - previous
                )
                / abs(value)
                * 100.0
            )

        dominant = (
            result.dominant_mode.m,
            result.dominant_mode.n,
        )

        participation = dominant_participation(
            result,
            count=5,
        )

        participation_text = "; ".join(
            "({},{})={:.3f}".format(
                m,
                n,
                amplitude,
            )
            for m, n, amplitude
            in participation
        )

        print(
            "{:8d} {:16.6f} {:16s} {:>14s} {:>24s}".format(
                size,
                value,
                (
                    "-"
                    if np.isnan(rel_change)
                    else "{:.6f}".format(
                        rel_change
                    )
                ),
                "({},{})".format(
                    dominant[0],
                    dominant[1],
                ),
                participation_text,
            )
        )

        rows.append(
            [
                size,
                value,
                rel_change,
                dominant[0],
                dominant[1],
            ]
        )

        previous = value

    return np.asarray(
        rows,
        dtype=float,
    )


def save_case(
    output_dir,
    stem,
    rows,
):
    csv_path = (
        output_dir
        / "{}_shear_convergence.csv".format(
            stem
        )
    )

    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header=(
            "basis_size,"
            "Nxy_cr_N_per_mm,"
            "relative_change_percent,"
            "dominant_m,"
            "dominant_n"
        ),
        comments="",
    )

    fig, ax = plt.subplots()

    ax.plot(
        rows[:, 0],
        rows[:, 1],
        marker="o",
    )

    ax.set_xlabel(
        "Maximum Navier index M = N"
    )

    ax.set_ylabel(
        r"$N_{xy,cr}$ [N/mm]"
    )

    ax.set_title(
        "{}: shear-buckling modal convergence".format(
            stem.replace(
                "_",
                " ",
            ).title()
        )
    )

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "{}_shear_convergence.png".format(
            stem
        ),
        dpi=200,
        bbox_inches="tight",
    )

    return csv_path


def main():
    a = 100.0
    b = 100.0

    basis_sizes = [
        4,
        6,
        8,
        10,
        12,
        16,
    ]

    homogeneous = (
        build_homogeneous_plate()
    )

    gyroid = (
        build_gyroid_sandwich()
    )

    homogeneous_rows = (
        run_convergence_case(
            "Homogeneous AlSi10Mg plate",
            homogeneous,
            a,
            b,
            basis_sizes,
        )
    )

    gyroid_rows = (
        run_convergence_case(
            "Gyroid RD12 HOST sandwich",
            gyroid,
            a,
            b,
            basis_sizes,
        )
    )

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "multimode_buckling"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    homogeneous_csv = save_case(
        output_dir,
        "homogeneous",
        homogeneous_rows,
    )

    gyroid_csv = save_case(
        output_dir,
        "gyroid_rd12",
        gyroid_rows,
    )

    print()
    print("=" * 88)
    print("Convergence summary")
    print("=" * 88)

    for label, rows in (
        (
            "Homogeneous",
            homogeneous_rows,
        ),
        (
            "Gyroid RD12",
            gyroid_rows,
        ),
    ):
        last_change = rows[
            -1,
            2,
        ]

        print(
            "{:<20s}: final Nxy_cr = {:12.6f} N/mm, "
            "last change = {:.6f}%".format(
                label,
                rows[-1, 1],
                last_change,
            )
        )

    print()
    print(
        "Homogeneous CSV:",
        homogeneous_csv,
    )

    print(
        "Gyroid CSV     :",
        gyroid_csv,
    )


if __name__ == "__main__":
    main()