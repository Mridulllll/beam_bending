#!/usr/bin/env python3

import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.fsdt_reference import (
    build_energy_equivalent_fsdt_section,
    center_deflection_uniform_pressure_fsdt,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
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
        / (
            2.0
            * (1.0 + nu_face)
        )
    )

    print()
    print(
        "{:>11s} {:>13s} {:>13s} {:>13s} {:>12s} {:>12s}".format(
            "Ec/Ef",
            "FSDT [mm]",
            "Reddy [mm]",
            "MI [mm]",
            "R/FSDT",
            "MI/FSDT",
        )
    )

    print("-" * 84)

    for ratio in ratios:
        E_core = (
            ratio
            * E_face
        )

        C_core = (
            isotropic_stiffness_3d(
                E=E_core,
                nu=nu_core,
            )
        )

        G_core = (
            E_core
            / (
                2.0
                * (1.0 + nu_core)
            )
        )

        reddy = (
            build_three_layer_sandwich(
                C_bottom=C_face,
                C_core=C_core,
                C_top=C_face,
                hf_bottom=hf,
                hc=hc,
                hf_top=hf,
                warping=(
                    reddy_baseline_generalized_warping(
                        h
                    )
                ),
            )
        )

        mi = (
            build_three_layer_sandwich(
                C_bottom=C_face,
                C_core=C_core,
                C_top=C_face,
                hf_bottom=hf,
                hc=hc,
                hf_top=hf,
                warping=(
                    material_informed_generalized_warping(
                        h=h,
                        hc=hc,
                        G_face=G_face,
                        G_core=G_core,
                    )
                ),
            )
        )

        fsdt_section = (
            build_energy_equivalent_fsdt_section(
                reddy.layers
            )
        )

        w_fsdt = (
            center_deflection_uniform_pressure_fsdt(
                fsdt_section,
                q0=q0,
                a=a,
                b=b,
                max_mode=21,
            )
        )

        w_reddy = (
            center_deflection_uniform_pressure_hsdt(
                reddy.K11,
                q0=q0,
                a=a,
                b=b,
                max_mode=21,
            )
        )

        w_mi = (
            center_deflection_uniform_pressure_hsdt(
                mi.K11,
                q0=q0,
                a=a,
                b=b,
                max_mode=21,
            )
        )

        print(
            "{:11.4e} {:13.6f} {:13.6f} {:13.6f} {:12.6f} {:12.6f}".format(
                ratio,
                w_fsdt * 1.0e3,
                w_reddy * 1.0e3,
                w_mi * 1.0e3,
                w_reddy / w_fsdt,
                w_mi / w_fsdt,
            )
        )


if __name__ == "__main__":
    main()