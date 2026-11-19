#!/usr/bin/env python3

"""
Compare Nx-Nxy buckling interaction boundaries for two homogenized
TPMS sandwich-core states.

Curves:

    - itag00 Reddy
    - itag00 directional material-informed HOST
    - itag15 Reddy
    - itag15 directional material-informed HOST

Outputs:

    verification/directional_buckling_interaction/
        gyroid_rd12_compression_nx_nxy_physical.png
        gyroid_rd12_compression_nx_nxy_normalized.png
        gyroid_rd12_compression_nx_nxy.csv

The normalized figure uses each formulation's own pure-Nx and
pure-Nxy critical loads. It therefore highlights changes in interaction
shape separately from changes in absolute buckling resistance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from host.buckling_interaction import (
    directional_nx_nxy_interaction,
    nx_nxy_interaction,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.generalized_stiffness import (
    assemble_directional_generalized_stiffness,
)

from host.hsdt import (
    extract_directional_hsdt_stiffness,
)

from host.sandwich import (
    build_three_layer_sandwich,
)

from host.sandwich_warping import (
    material_informed_directional_generalized_warping,
)

from host.tpms_io import (
    load_tpms_homogenization_json,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def build_stiffnesses(
    json_file,
    *,
    E_face,
    nu_face,
    hf,
    hc,
):
    record = load_tpms_homogenization_json(
        json_file
    )

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

    h = (
        2.0 * hf
        + hc
    )

    G_face = (
        E_face
        / (
            2.0
            * (1.0 + nu_face)
        )
    )

    G_xz = float(
        record.core.G_xz
    )

    G_yz = float(
        record.core.G_yz
    )

    reddy = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=record.C_hom,
        C_top=C_face,
        hf_bottom=hf,
        hc=hc,
        hf_top=hf,
        warping=reddy_baseline_generalized_warping(
            h
        ),
    )

    directional_warping = (
        material_informed_directional_generalized_warping(
            h=h,
            hc=hc,
            G_face_xz=G_face,
            G_face_yz=G_face,
            G_core_xz=G_xz,
            G_core_yz=G_yz,
        )
    )

    K15 = assemble_directional_generalized_stiffness(
        reddy.layers,
        directional_warping,
        quadrature_order=10,
    )

    K12 = extract_directional_hsdt_stiffness(
        K15
    )

    return (
        record,
        reddy.K11,
        K12,
    )


def evaluate_scalar(
    K11,
    *,
    a,
    b,
    angles,
    max_mode,
):
    return nx_nxy_interaction(
        K11,
        a=a,
        b=b,
        angles_deg=angles,
        max_m=max_mode,
        max_n=max_mode,
    )


def evaluate_directional(
    K12,
    *,
    a,
    b,
    angles,
    max_mode,
):
    return directional_nx_nxy_interaction(
        K12,
        a=a,
        b=b,
        angles_deg=angles,
        max_m=max_mode,
        max_n=max_mode,
    )


def result_arrays(points):
    return {
        "angle": np.asarray(
            [
                point.parameter
                for point in points
            ],
            dtype=float,
        ),
        "Nx": np.asarray(
            [
                point.Nx_cr
                for point in points
            ],
            dtype=float,
        ),
        "Nxy": np.asarray(
            [
                point.Nxy_cr
                for point in points
            ],
            dtype=float,
        ),
        "lambda": np.asarray(
            [
                point.load_factor
                for point in points
            ],
            dtype=float,
        ),
        "m": np.asarray(
            [
                point.result.dominant_mode.m
                for point in points
            ],
            dtype=int,
        ),
        "n": np.asarray(
            [
                point.result.dominant_mode.n
                for point in points
            ],
            dtype=int,
        ),
    }


def endpoint_scales(data):
    angles = data["angle"]

    i_nx = int(
        np.argmin(
            np.abs(
                angles
            )
        )
    )

    i_pos_shear = int(
        np.argmin(
            np.abs(
                angles - 90.0
            )
        )
    )

    return (
        abs(
            data["Nx"][i_nx]
        ),
        abs(
            data["Nxy"][i_pos_shear]
        ),
    )


def print_mode_transitions(
    label,
    data,
):
    print()
    print(
        "{} dominant-mode transitions".format(
            label
        )
    )
    print("-" * 72)

    previous = None

    for angle, m, n in zip(
        data["angle"],
        data["m"],
        data["n"],
    ):
        mode = (
            int(m),
            int(n),
        )

        if mode != previous:
            print(
                "theta = {:7.2f} deg -> mode ({},{})".format(
                    angle,
                    mode[0],
                    mode[1],
                )
            )

            previous = mode


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "itag00",
    )

    parser.add_argument(
        "itag15",
    )

    parser.add_argument(
        "--face-E",
        type=float,
        default=70000.0,
    )

    parser.add_argument(
        "--face-nu",
        type=float,
        default=0.33,
    )

    parser.add_argument(
        "--hf",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--hc",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--a",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--b",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--max-mode",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--angle-step",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "verification/"
            "directional_buckling_interaction"
        ),
    )

    args = parser.parse_args()

    E_face = float(
        args.face_E
    )

    nu_face = float(
        args.face_nu
    )

    hf = float(
        args.hf
    )

    hc = float(
        args.hc
    )

    a = float(
        args.a
    )

    b = float(
        args.b
    )

    max_mode = int(
        args.max_mode
    )

    step = float(
        args.angle_step
    )

    if step <= 0.0:
        raise ValueError(
            "angle-step must be positive."
        )

    count = int(
        round(
            180.0 / step
        )
    ) + 1

    angles = np.linspace(
        -90.0,
        90.0,
        count,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Build both TPMS states.
    # --------------------------------------------------------------

    rec00, K11_00, K12_00 = build_stiffnesses(
        args.itag00,
        E_face=E_face,
        nu_face=nu_face,
        hf=hf,
        hc=hc,
    )

    rec15, K11_15, K12_15 = build_stiffnesses(
        args.itag15,
        E_face=E_face,
        nu_face=nu_face,
        hf=hf,
        hc=hc,
    )

    print()
    print("=" * 78)
    print("TPMS directional Nx-Nxy interaction study")
    print("=" * 78)

    print(
        "plate                       : {:.3f} x {:.3f} mm".format(
            a,
            b,
        )
    )

    print(
        "modal truncation            : {} x {}".format(
            max_mode,
            max_mode,
        )
    )

    print(
        "angle step                  : {:.3f} deg".format(
            step
        )
    )

    print()

    print(
        "itag00 Gxz/Gyz [MPa]        : {:.6f} / {:.6f}".format(
            rec00.core.G_xz,
            rec00.core.G_yz,
        )
    )

    print(
        "itag15 Gxz/Gyz [MPa]        : {:.6f} / {:.6f}".format(
            rec15.core.G_xz,
            rec15.core.G_yz,
        )
    )

    # --------------------------------------------------------------
    # Evaluate curves.
    # --------------------------------------------------------------

    print()
    print("Solving itag00 Reddy ...")

    p00_reddy = evaluate_scalar(
        K11_00,
        a=a,
        b=b,
        angles=angles,
        max_mode=max_mode,
    )

    print("Solving itag00 directional HOST ...")

    p00_dir = evaluate_directional(
        K12_00,
        a=a,
        b=b,
        angles=angles,
        max_mode=max_mode,
    )

    print("Solving itag15 Reddy ...")

    p15_reddy = evaluate_scalar(
        K11_15,
        a=a,
        b=b,
        angles=angles,
        max_mode=max_mode,
    )

    print("Solving itag15 directional HOST ...")

    p15_dir = evaluate_directional(
        K12_15,
        a=a,
        b=b,
        angles=angles,
        max_mode=max_mode,
    )

    curves = {
        "itag00_reddy": result_arrays(
            p00_reddy
        ),
        "itag00_directional": result_arrays(
            p00_dir
        ),
        "itag15_reddy": result_arrays(
            p15_reddy
        ),
        "itag15_directional": result_arrays(
            p15_dir
        ),
    }

    # --------------------------------------------------------------
    # Endpoint summary.
    # --------------------------------------------------------------

    print()
    print("-" * 78)
    print("Endpoint critical loads")
    print("-" * 78)

    for name, data in curves.items():
        Nx0, Nxy0 = endpoint_scales(
            data
        )

        print(
            "{:<24s} Nx = {:11.4f}, Nxy = {:11.4f} N/mm".format(
                name,
                Nx0,
                Nxy0,
            )
        )

    # --------------------------------------------------------------
    # Mode transition summary.
    # --------------------------------------------------------------

    for name, data in curves.items():
        print_mode_transitions(
            name,
            data,
        )

    # --------------------------------------------------------------
    # CSV.
    # --------------------------------------------------------------

    csv_path = (
        output_dir
        / "gyroid_rd12_compression_nx_nxy.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "state",
                "theory",
                "angle_deg",
                "Nx_cr_N_per_mm",
                "Nxy_cr_N_per_mm",
                "load_factor",
                "dominant_m",
                "dominant_n",
            ]
        )

        metadata = [
            (
                "itag00",
                "Reddy",
                curves[
                    "itag00_reddy"
                ],
            ),
            (
                "itag00",
                "Directional HOST",
                curves[
                    "itag00_directional"
                ],
            ),
            (
                "itag15",
                "Reddy",
                curves[
                    "itag15_reddy"
                ],
            ),
            (
                "itag15",
                "Directional HOST",
                curves[
                    "itag15_directional"
                ],
            ),
        ]

        for state, theory, data in metadata:
            for i in range(
                len(
                    data["angle"]
                )
            ):
                writer.writerow(
                    [
                        state,
                        theory,
                        data["angle"][i],
                        data["Nx"][i],
                        data["Nxy"][i],
                        data["lambda"][i],
                        data["m"][i],
                        data["n"][i],
                    ]
                )

    # --------------------------------------------------------------
    # Physical interaction plot.
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.0, 6.5)
    )

    ax.plot(
        curves["itag00_reddy"]["Nx"],
        curves["itag00_reddy"]["Nxy"],
        linestyle="--",
        label="itag00 Reddy",
    )

    ax.plot(
        curves["itag00_directional"]["Nx"],
        curves["itag00_directional"]["Nxy"],
        linestyle="-",
        label="itag00 directional HOST",
    )

    ax.plot(
        curves["itag15_reddy"]["Nx"],
        curves["itag15_reddy"]["Nxy"],
        linestyle="--",
        label="itag15 Reddy",
    )

    ax.plot(
        curves["itag15_directional"]["Nx"],
        curves["itag15_directional"]["Nxy"],
        linestyle="-",
        label="itag15 directional HOST",
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.axvline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        r"$N_x$ [N/mm]"
    )

    ax.set_ylabel(
        r"$N_{xy}$ [N/mm]"
    )

    ax.set_title(
        "Gyroid RD12 compression optimization\n"
        "Nx-Nxy buckling interaction"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    physical_path = (
        output_dir
        / "gyroid_rd12_compression_nx_nxy_physical.png"
    )

    fig.savefig(
        physical_path,
        dpi=300,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------------
    # Normalized interaction plot.
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.0, 6.5)
    )

    for key, label, linestyle in [
        (
            "itag00_reddy",
            "itag00 Reddy",
            "--",
        ),
        (
            "itag00_directional",
            "itag00 directional HOST",
            "-",
        ),
        (
            "itag15_reddy",
            "itag15 Reddy",
            "--",
        ),
        (
            "itag15_directional",
            "itag15 directional HOST",
            "-",
        ),
    ]:
        data = curves[
            key
        ]

        Nx0, Nxy0 = endpoint_scales(
            data
        )

        ax.plot(
            data["Nx"] / Nx0,
            data["Nxy"] / Nxy0,
            linestyle=linestyle,
            label=label,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.axvline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        r"$N_x/N_{x,\mathrm{cr}}$"
    )

    ax.set_ylabel(
        r"$N_{xy}/N_{xy,\mathrm{cr}}$"
    )

    ax.set_title(
        "Normalized Nx-Nxy buckling interaction"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    normalized_path = (
        output_dir
        / "gyroid_rd12_compression_nx_nxy_normalized.png"
    )

    fig.savefig(
        normalized_path,
        dpi=300,
    )

    plt.close(
        fig
    )

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)

    print(
        "CSV        : {}".format(
            csv_path
        )
    )

    print(
        "Physical   : {}".format(
            physical_path
        )
    )

    print(
        "Normalized : {}".format(
            normalized_path
        )
    )


if __name__ == "__main__":
    main()