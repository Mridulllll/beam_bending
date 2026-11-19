#!/usr/bin/env python3

"""
First HOST sandwich benchmark.

Three-layer sandwich:

    Al face / compliant homogenized surrogate core / Al face

The core is initially isotropic so that the mechanics can be understood
before inserting actual TPMS homogenization tensors.

Parameter:

    E_core / E_face

Outputs:

    center deflection
    normalized deflection
    first-mode energy decomposition
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.energy import (
    modal_energy_decomposition,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
    solve_navier_mode_basis_aware,
)

from host.sandwich import (
    build_three_layer_sandwich,
)


def main():
    E_face = 70.0e9
    nu_face = 0.33

    nu_core = 0.25

    hf = 0.5e-3
    hc = 10.0e-3

    a = 0.1
    b = 0.1

    q0 = 10.0e3

    ratios = np.logspace(
        -4,
        -1,
        13,
    )

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

    rows = []

    print()
    print(
        "{:>12s} {:>14s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            "Ec/Ef",
            "Ec [GPa]",
            "w_center [mm]",
            "U_bend/U",
            "U_shear/U",
            "U_warp/U",
        )
    )

    print("-" * 90)

    for ratio in ratios:
        E_core = (
            ratio
            * E_face
        )

        C_core = isotropic_stiffness_3d(
            E=E_core,
            nu=nu_core,
        )

        sandwich = build_three_layer_sandwich(
            C_bottom=C_face,
            C_core=C_core,
            C_top=C_face,
            hf_bottom=hf,
            hc=hc,
            hf_top=hf,
        )

        w_center = (
            center_deflection_uniform_pressure_hsdt(
                sandwich.K11,
                q0=q0,
                a=a,
                b=b,
                max_mode=21,
            )
        )

        q11 = solve_navier_mode_basis_aware(
            sandwich.K11,
            q0=q0,
            a=a,
            b=b,
            m=1,
            n=1,
        )

        energy = modal_energy_decomposition(
            sandwich.K11,
            q11,
            a=a,
            b=b,
            m=1,
            n=1,
        )

        U = energy.total

        bend_fraction = (
            energy.self_terms[
                "bending"
            ]
            / U
        )

        shear_fraction = (
            energy.self_terms[
                "shear"
            ]
            / U
        )

        warping_fraction = (
            energy.self_terms[
                "warping"
            ]
            / U
        )

        rows.append(
            [
                ratio,
                E_core,
                w_center,
                bend_fraction,
                shear_fraction,
                warping_fraction,
            ]
        )

        print(
            "{:12.4e} {:14.6f} {:14.6f} {:14.6f} {:14.6f} {:14.6f}".format(
                ratio,
                E_core / 1.0e9,
                w_center * 1.0e3,
                bend_fraction,
                shear_fraction,
                warping_fraction,
            )
        )

    rows = np.asarray(
        rows,
        dtype=float,
    )

    repo_root = Path(
        __file__
    ).resolve().parents[1]

    output_dir = (
        repo_root
        / "verification"
        / "sandwich_core_stiffness"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "core_stiffness_sweep.csv"
    )

    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header=(
            "Ec_over_Ef,"
            "Ec_Pa,"
            "w_center_m,"
            "bending_self_fraction,"
            "shear_self_fraction,"
            "warping_self_fraction"
        ),
        comments="",
    )

    fig, ax = plt.subplots()

    ax.loglog(
        rows[:, 0],
        np.abs(
            rows[:, 2]
        ) * 1.0e3,
        marker="o",
    )

    ax.set_xlabel(
        r"$E_c/E_f$"
    )

    ax.set_ylabel(
        "Center deflection [mm]"
    )

    ax.set_title(
        "Sandwich compliance versus core stiffness"
    )

    ax.grid(
        True,
        which="both",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "core_stiffness_vs_deflection.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig2, ax2 = plt.subplots()

    ax2.semilogx(
        rows[:, 0],
        rows[:, 3],
        marker="o",
        label="Bending self-energy",
    )

    ax2.semilogx(
        rows[:, 0],
        rows[:, 4],
        marker="o",
        label="Shear self-energy",
    )

    ax2.semilogx(
        rows[:, 0],
        rows[:, 5],
        marker="o",
        label="Warping self-energy",
    )

    ax2.set_xlabel(
        r"$E_c/E_f$"
    )

    ax2.set_ylabel(
        "Fraction of total modal energy"
    )

    ax2.set_title(
        "First-mode energy contributions"
    )

    ax2.grid(
        True,
        which="both",
    )

    ax2.legend()

    fig2.tight_layout()

    fig2.savefig(
        output_dir
        / "core_stiffness_energy_contributions.png",
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        "CSV:",
        csv_path,
    )


if __name__ == "__main__":
    main()