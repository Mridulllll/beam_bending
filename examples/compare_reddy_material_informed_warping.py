#!/usr/bin/env python3

"""
Compare the global Reddy HSDT warping function with the
material-informed sandwich-specific warping function.

The same symmetric three-layer sandwich is analysed with both
kinematic assumptions.

Parameter:
    E_core / E_face

Reported quantities:
    - center deflection
    - deflection ratio
    - effective first-mode transverse stiffness
    - higher-order modal softening
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from host.condensation import (
    condense_to_transverse_stiffness,
    higher_order_softening_fraction,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
    navier_modal_stiffness_basis_aware,
)

from host.sandwich import (
    build_three_layer_sandwich,
)

from host.sandwich_warping import (
    material_informed_generalized_warping,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def main():
    E_face = 70.0e9
    nu_face = 0.33

    nu_core = 0.25

    hf = 0.5e-3
    hc = 10.0e-3

    h = 2.0 * hf + hc

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

    G_face = (
        E_face
        / (2.0 * (1.0 + nu_face))
    )

    rows = []

    print()
    print(
        "{:>11s} {:>13s} {:>13s} {:>12s} {:>13s} {:>13s}".format(
            "Ec/Ef",
            "w_R [mm]",
            "w_MI [mm]",
            "w_MI/w_R",
            "soft_R [%]",
            "soft_MI [%]",
        )
    )

    print("-" * 84)

    for ratio in ratios:
        E_core = (
            ratio
            * E_face
        )

        C_core = isotropic_stiffness_3d(
            E=E_core,
            nu=nu_core,
        )

        G_core = (
            E_core
            / (2.0 * (1.0 + nu_core))
        )

        # ----------------------------------------------------------
        # Global Reddy warping
        # ----------------------------------------------------------

        reddy_warping = (
            reddy_baseline_generalized_warping(
                h
            )
        )

        sandwich_reddy = (
            build_three_layer_sandwich(
                C_bottom=C_face,
                C_core=C_core,
                C_top=C_face,
                hf_bottom=hf,
                hc=hc,
                hf_top=hf,
                warping=reddy_warping,
            )
        )

        # ----------------------------------------------------------
        # Material-informed warping
        # ----------------------------------------------------------

        mi_warping = (
            material_informed_generalized_warping(
                h=h,
                hc=hc,
                G_face=G_face,
                G_core=G_core,
            )
        )

        sandwich_mi = (
            build_three_layer_sandwich(
                C_bottom=C_face,
                C_core=C_core,
                C_top=C_face,
                hf_bottom=hf,
                hc=hc,
                hf_top=hf,
                warping=mi_warping,
            )
        )

        # ----------------------------------------------------------
        # Center deflection
        # ----------------------------------------------------------

        w_reddy = (
            center_deflection_uniform_pressure_hsdt(
                sandwich_reddy.K11,
                q0=q0,
                a=a,
                b=b,
                max_mode=21,
            )
        )

        w_mi = (
            center_deflection_uniform_pressure_hsdt(
                sandwich_mi.K11,
                q0=q0,
                a=a,
                b=b,
                max_mode=21,
            )
        )

        # ----------------------------------------------------------
        # First Navier mode
        # ----------------------------------------------------------

        Kmn_reddy = (
            navier_modal_stiffness_basis_aware(
                sandwich_reddy.K11,
                a=a,
                b=b,
                m=1,
                n=1,
            )
        )

        Kmn_mi = (
            navier_modal_stiffness_basis_aware(
                sandwich_mi.K11,
                a=a,
                b=b,
                m=1,
                n=1,
            )
        )

        k_reddy = (
            condense_to_transverse_stiffness(
                Kmn_reddy
            )
        )

        k_mi = (
            condense_to_transverse_stiffness(
                Kmn_mi
            )
        )

        soft_reddy = (
            100.0
            * higher_order_softening_fraction(
                Kmn_reddy
            )
        )

        soft_mi = (
            100.0
            * higher_order_softening_fraction(
                Kmn_mi
            )
        )

        rows.append(
            [
                ratio,
                w_reddy,
                w_mi,
                w_mi / w_reddy,
                k_reddy,
                k_mi,
                soft_reddy,
                soft_mi,
            ]
        )

        print(
            "{:11.4e} {:13.6f} {:13.6f} {:12.6f} {:13.6f} {:13.6f}".format(
                ratio,
                w_reddy * 1.0e3,
                w_mi * 1.0e3,
                w_mi / w_reddy,
                soft_reddy,
                soft_mi,
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
        / "sandwich_warping_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output_dir
        / "reddy_vs_material_informed.csv",
        rows,
        delimiter=",",
        header=(
            "Ec_over_Ef,"
            "w_reddy_m,"
            "w_material_informed_m,"
            "w_ratio_MI_over_Reddy,"
            "k_eff_reddy,"
            "k_eff_material_informed,"
            "softening_reddy_percent,"
            "softening_material_informed_percent"
        ),
        comments="",
    )

    fig, ax = plt.subplots()

    ax.semilogx(
        rows[:, 0],
        rows[:, 2] / rows[:, 1],
        marker="o",
    )

    ax.axhline(
        1.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        r"$E_c/E_f$"
    )

    ax.set_ylabel(
        r"$w_{\mathrm{MI}}/w_{\mathrm{Reddy}}$"
    )

    ax.set_title(
        "Effect of sandwich-specific warping"
    )

    ax.grid(
        True,
        which="both",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "deflection_ratio.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig2, ax2 = plt.subplots()

    ax2.semilogx(
        rows[:, 0],
        rows[:, 6],
        marker="o",
        label="Reddy",
    )

    ax2.semilogx(
        rows[:, 0],
        rows[:, 7],
        marker="o",
        label="Material-informed",
    )

    ax2.set_xlabel(
        r"$E_c/E_f$"
    )

    ax2.set_ylabel(
        "Higher-order softening [%]"
    )

    ax2.set_title(
        "Higher-order modal relaxation"
    )

    ax2.grid(
        True,
        which="both",
    )

    ax2.legend()

    fig2.tight_layout()

    fig2.savefig(
        output_dir
        / "softening_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()