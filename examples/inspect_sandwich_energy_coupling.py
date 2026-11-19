#!/usr/bin/env python3

import numpy as np

from host.constitutive import isotropic_stiffness_3d
from host.energy import (
    modal_energy_decomposition,
    reconstructed_total_energy,
)
from host.navier import solve_navier_mode_basis_aware
from host.sandwich import build_three_layer_sandwich


def main():
    E_face = 70.0e9
    nu_face = 0.33
    nu_core = 0.25

    hf = 0.5e-3
    hc = 10.0e-3

    a = 0.1
    b = 0.1
    q0 = 10.0e3

    ratios = [
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
    ]

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

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

        reconstructed = reconstructed_total_energy(
            energy
        )

        print()
        print("=" * 72)
        print("Ec/Ef = {:.4e}".format(ratio))
        print("=" * 72)

        print("\nSelf terms:")
        for name, value in energy.self_terms.items():
            print(
                "  {:12s}: {: .8e}   ({: .6f} U)".format(
                    name,
                    value,
                    value / energy.total,
                )
            )

        print("\nCoupling terms:")
        for names, value in energy.coupling_terms.items():
            print(
                "  {:12s}-{:12s}: {: .8e}   ({: .6f} U)".format(
                    names[0],
                    names[1],
                    value,
                    value / energy.total,
                )
            )

        print()
        print(
            "Total energy       : {:.8e}".format(
                energy.total
            )
        )
        print(
            "Reconstructed total: {:.8e}".format(
                reconstructed
            )
        )
        print(
            "Relative residual  : {:.3e}".format(
                abs(
                    reconstructed - energy.total
                )
                / max(
                    abs(energy.total),
                    1.0e-30,
                )
            )
        )


if __name__ == "__main__":
    main()