#!/usr/bin/env python3

"""
Quantitative verification of the five-field HOST HSDT against the
classical Kirchhoff-Love thin-plate solution.

Outputs:
    - console table
    - CSV file
    - convergence figure

Study parameter:
    h/a

The expected asymptotic behavior is

    w_HSDT / w_KL -> 1

as

    h/a -> 0.
"""

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

from host.kirchhoff import (
    center_deflection_uniform_pressure,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def build_hsdt_plate_stiffness(
    E,
    nu,
    h,
):
    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    C5 = sigma_zz_zero_stiffness(
        C3
    )

    layers = [
        Layer(
            name="homogeneous_plate",
            z_bottom=-0.5 * h,
            z_top=+0.5 * h,
            stiffness=C5,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(
        K14
    )


def main():
    E = 70.0e9
    nu = 0.33

    a = 1.0
    b = 1.0

    q0 = 1000.0

    thickness_ratios = np.array(
        [
            1.0e-1,
            7.5e-2,
            5.0e-2,
            2.5e-2,
            1.0e-2,
            7.5e-3,
            5.0e-3,
            2.5e-3,
            1.0e-3,
        ],
        dtype=float,
    )

    rows = []

    print()
    print(
        "{:>12s} {:>16s} {:>16s} {:>16s} {:>16s}".format(
            "h/a",
            "w_HSDT [m]",
            "w_KL [m]",
            "w_HSDT/w_KL",
            "rel. error [%]",
        )
    )

    print("-" * 84)

    for h_over_a in thickness_ratios:
        h = h_over_a * a

        K_hsdt = build_hsdt_plate_stiffness(
            E,
            nu,
            h,
        )

        w_hsdt = (
            center_deflection_uniform_pressure_hsdt(
                K_hsdt,
                q0,
                a,
                b,
                max_mode=31,
            )
        )

        w_kl = (
            center_deflection_uniform_pressure(
                q0,
                E,
                nu,
                h,
                a,
                b,
                max_mode=31,
            )
        )

        ratio = w_hsdt / w_kl

        rel_error_percent = (
            100.0
            * (w_hsdt - w_kl)
            / w_kl
        )

        rows.append(
            [
                h_over_a,
                w_hsdt,
                w_kl,
                ratio,
                rel_error_percent,
            ]
        )

        print(
            "{:12.5e} {:16.8e} {:16.8e} {:16.8f} {:16.8f}".format(
                h_over_a,
                w_hsdt,
                w_kl,
                ratio,
                rel_error_percent,
            )
        )

    rows = np.asarray(
        rows,
        dtype=float,
    )

    repo_root = Path(__file__).resolve().parents[1]

    output_dir = (
        repo_root
        / "verification"
        / "hsdt_kirchhoff"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "hsdt_kirchhoff_convergence.csv"
    )

    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header=(
            "h_over_a,"
            "w_hsdt_m,"
            "w_kirchhoff_m,"
            "w_ratio,"
            "relative_error_percent"
        ),
        comments="",
    )

    fig, ax = plt.subplots()

    ax.semilogx(
        rows[:, 0],
        rows[:, 3],
        marker="o",
    )

    ax.axhline(
        1.0,
        linewidth=1.0,
    )

    ax.set_xlabel("Thickness ratio h/a")
    ax.set_ylabel(r"$w_{\mathrm{HSDT}}/w_{\mathrm{KL}}$")
    ax.set_title(
        "HSDT to Kirchhoff-Love convergence"
    )

    ax.grid(True)

    fig.tight_layout()

    figure_path = (
        output_dir
        / "hsdt_kirchhoff_convergence.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        "CSV:",
        csv_path,
    )

    print(
        "Figure:",
        figure_path,
    )


if __name__ == "__main__":
    main()