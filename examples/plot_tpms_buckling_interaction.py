#!/usr/bin/env python3

"""
Compare TPMS sandwich buckling interaction envelopes.

Default comparison:

    Gyroid RD12 initial shear state, itag00
    Gyroid RD12 optimized shear state, itag18

Interaction spaces:

    1. Nx - Nxy
    2. Nx - Ny

The material-informed HOST sandwich model is used with a converged
16 x 16 Navier modal basis.

Outputs:
    - CSV data
    - Nx-Nxy interaction figure
    - Nx-Ny interaction figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from host.buckling_interaction import (
    nx_nxy_interaction,
    nx_ny_interaction,
)

from host.constitutive import (
    isotropic_stiffness_3d,
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


QUERY_DIR = Path(
    r"C:\AbaqusScratch\UCs\Shells\TPMS_mapped_thickness"
    r"\_stiffness_database\queries"
)

INITIAL_JSON = (
    QUERY_DIR
    / "sizing_shape_Gyroid_rd12_shear_itag00.json"
)

OPTIMIZED_JSON = (
    QUERY_DIR
    / "sizing_shape_Gyroid_rd12_shear_itag18.json"
)


def build_host_sandwich(
    json_path,
):
    """
    Build the material-informed HOST sandwich from one homogenization JSON.
    """

    record = load_tpms_homogenization_json(
        json_path
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

    return record, sandwich


def interaction_rows(
    points,
):
    """
    Convert interaction points to a numeric array.

    Columns:

        parameter
        lambda_cr
        Nx_cr
        Ny_cr
        Nxy_cr
        dominant_m
        dominant_n
        dominant_amplitude
    """

    rows = []

    for point in points:
        result = point.result

        amplitudes = np.abs(
            result.eigenvector
        )

        index = int(
            np.argmax(
                amplitudes
            )
        )

        dominant = result.modes[
            index
        ]

        rows.append(
            [
                point.parameter,
                point.load_factor,
                point.Nx_cr,
                point.Ny_cr,
                point.Nxy_cr,
                dominant.m,
                dominant.n,
                amplitudes[index],
            ]
        )

    return np.asarray(
        rows,
        dtype=float,
    )


def save_csv(
    path,
    rows,
):
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=(
            "parameter_deg,"
            "lambda_cr,"
            "Nx_cr_N_per_mm,"
            "Ny_cr_N_per_mm,"
            "Nxy_cr_N_per_mm,"
            "dominant_m,"
            "dominant_n,"
            "dominant_amplitude"
        ),
        comments="",
    )


def print_mode_transitions(
    label,
    rows,
):
    """
    Print locations where the dominant mode changes.
    """

    print()
    print(label)
    print("-" * 72)

    previous = None

    for row in rows:
        angle = row[0]

        mode = (
            int(row[5]),
            int(row[6]),
        )

        if mode != previous:
            print(
                "parameter={:8.3f} deg   dominant mode=({},{})".format(
                    angle,
                    mode[0],
                    mode[1],
                )
            )

            previous = mode


def main():
    a = 100.0
    b = 100.0

    max_m = 16
    max_n = 16

    record_initial, initial = (
        build_host_sandwich(
            INITIAL_JSON
        )
    )

    record_optimized, optimized = (
        build_host_sandwich(
            OPTIMIZED_JSON
        )
    )

    print()
    print("=" * 76)
    print("TPMS buckling interaction comparison")
    print("=" * 76)

    print(
        "Initial   : {} RD={:.6f}".format(
            record_initial.topology,
            record_initial.relative_density_actual,
        )
    )

    print(
        "Optimized : {} RD={:.6f}".format(
            record_optimized.topology,
            record_optimized.relative_density_actual,
        )
    )

    print(
        "Modal basis: {} x {}".format(
            max_m,
            max_n,
        )
    )

    # --------------------------------------------------------------
    # Nx-Nxy interaction
    # --------------------------------------------------------------

    shear_angles = np.linspace(
        -90.0,
        90.0,
        73,
    )

    initial_nx_nxy = nx_nxy_interaction(
        initial.K11,
        a,
        b,
        angles_deg=shear_angles,
        max_m=max_m,
        max_n=max_n,
    )

    optimized_nx_nxy = nx_nxy_interaction(
        optimized.K11,
        a,
        b,
        angles_deg=shear_angles,
        max_m=max_m,
        max_n=max_n,
    )

    rows_initial_nx_nxy = interaction_rows(
        initial_nx_nxy
    )

    rows_optimized_nx_nxy = interaction_rows(
        optimized_nx_nxy
    )

    # --------------------------------------------------------------
    # Nx-Ny interaction
    # --------------------------------------------------------------

    biaxial_angles = np.linspace(
        0.0,
        90.0,
        37,
    )

    initial_nx_ny = nx_ny_interaction(
        initial.K11,
        a,
        b,
        angles_deg=biaxial_angles,
        max_m=max_m,
        max_n=max_n,
    )

    optimized_nx_ny = nx_ny_interaction(
        optimized.K11,
        a,
        b,
        angles_deg=biaxial_angles,
        max_m=max_m,
        max_n=max_n,
    )

    rows_initial_nx_ny = interaction_rows(
        initial_nx_ny
    )

    rows_optimized_nx_ny = interaction_rows(
        optimized_nx_ny
    )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "tpms_buckling_interaction"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    save_csv(
        output_dir
        / "gyroid_rd12_initial_nx_nxy.csv",
        rows_initial_nx_nxy,
    )

    save_csv(
        output_dir
        / "gyroid_rd12_shear18_nx_nxy.csv",
        rows_optimized_nx_nxy,
    )

    save_csv(
        output_dir
        / "gyroid_rd12_initial_nx_ny.csv",
        rows_initial_nx_ny,
    )

    save_csv(
        output_dir
        / "gyroid_rd12_shear18_nx_ny.csv",
        rows_optimized_nx_ny,
    )

    # --------------------------------------------------------------
    # Nx-Nxy figure
    # --------------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        rows_initial_nx_nxy[:, 2],
        rows_initial_nx_nxy[:, 4],
        marker="o",
        markersize=3,
        label="Initial",
    )

    ax.plot(
        rows_optimized_nx_nxy[:, 2],
        rows_optimized_nx_nxy[:, 4],
        marker="o",
        markersize=3,
        label="Shear optimized itag18",
    )

    ax.set_xlabel(
        r"$N_{x,cr}$ [N/mm]"
    )

    ax.set_ylabel(
        r"$N_{xy,cr}$ [N/mm]"
    )

    ax.set_title(
        "Gyroid RD12 buckling interaction: "
        r"$N_x-N_{xy}$"
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.axvline(
        0.0,
        linewidth=0.8,
    )

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "gyroid_rd12_initial_vs_shear18_nx_nxy.png",
        dpi=250,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Nx-Ny figure
    # --------------------------------------------------------------

    fig2, ax2 = plt.subplots()

    ax2.plot(
        rows_initial_nx_ny[:, 2],
        rows_initial_nx_ny[:, 3],
        marker="o",
        markersize=3,
        label="Initial",
    )

    ax2.plot(
        rows_optimized_nx_ny[:, 2],
        rows_optimized_nx_ny[:, 3],
        marker="o",
        markersize=3,
        label="Shear optimized itag18",
    )

    ax2.set_xlabel(
        r"$N_{x,cr}$ [N/mm]"
    )

    ax2.set_ylabel(
        r"$N_{y,cr}$ [N/mm]"
    )

    ax2.set_title(
        "Gyroid RD12 buckling interaction: "
        r"$N_x-N_y$"
    )

    ax2.grid(True)
    ax2.legend()

    fig2.tight_layout()

    fig2.savefig(
        output_dir
        / "gyroid_rd12_initial_vs_shear18_nx_ny.png",
        dpi=250,
        bbox_inches="tight",
    )

    # --------------------------------------------------------------
    # Mode transitions
    # --------------------------------------------------------------

    print_mode_transitions(
        "Initial Nx-Nxy mode transitions",
        rows_initial_nx_nxy,
    )

    print_mode_transitions(
        "Optimized Nx-Nxy mode transitions",
        rows_optimized_nx_nxy,
    )

    print_mode_transitions(
        "Initial Nx-Ny mode transitions",
        rows_initial_nx_ny,
    )

    print_mode_transitions(
        "Optimized Nx-Ny mode transitions",
        rows_optimized_nx_ny,
    )

    print()
    print(
        "Output directory:",
        output_dir,
    )


if __name__ == "__main__":
    main()