#!/usr/bin/env python3

import numpy as np

from host.condensation import (
    condense_to_transverse_stiffness,
    higher_order_softening_fraction,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.navier import (
    navier_modal_stiffness_basis_aware,
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

    ratios = np.logspace(
        -4,
        -1,
        13,
    )

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

    print()
    print(
        "{:>12s} {:>16s} {:>16s} {:>16s} {:>16s}".format(
            "Ec/Ef",
            "K_WW raw",
            "k_W effective",
            "k_eff/K_raw",
            "softening [%]",
        )
    )

    print("-" * 84)

    for ratio in ratios:
        C_core = isotropic_stiffness_3d(
            E=ratio * E_face,
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

        Kmn = navier_modal_stiffness_basis_aware(
            sandwich.K11,
            a=a,
            b=b,
            m=1,
            n=1,
        )

        raw = float(
            Kmn[2, 2]
        )

        effective = (
            condense_to_transverse_stiffness(
                Kmn
            )
        )

        ratio_eff = (
            effective / raw
        )

        softening = (
            100.0
            * higher_order_softening_fraction(
                Kmn
            )
        )

        print(
            "{:12.4e} {:16.8e} {:16.8e} {:16.8f} {:16.8f}".format(
                ratio,
                raw,
                effective,
                ratio_eff,
                softening,
            )
        )


if __name__ == "__main__":
    main()