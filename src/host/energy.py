"""
Energy decomposition for the five-field HOST HSDT.

The generalized strain vector is partitioned into

    membrane : 0:3
    bending  : 3:6
    warping  : 6:9
    shear    : 9:11

For one Navier mode, the actual spatial sine/cosine basis families
are treated independently so that the energy decomposition remains
consistent with the basis-aware Navier formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from host.navier import (
    generalized_strain_basis_operators,
    navier_mode,
)


GROUPS = {
    "membrane": (0, 3),
    "bending": (3, 6),
    "warping": (6, 9),
    "shear": (9, 11),
}


@dataclass(frozen=True)
class ModalEnergyDecomposition:
    self_terms: dict
    coupling_terms: dict
    total: float


def modal_energy_decomposition(
    K_hsdt: np.ndarray,
    modal_amplitudes: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
) -> ModalEnergyDecomposition:
    """
    Exact generalized strain-energy decomposition for one Navier mode.

    The energy associated with one spatial basis family r is

        U_r = 1/2 * (a*b/4) * E_r^T K E_r.

    Self terms are

        U_ii = 1/2 * factor * E_i^T K_ii E_i,

    and coupling terms are

        U_ij = factor * E_i^T K_ij E_j,

    for i != j.

    Summation of every self and coupling contribution exactly
    reproduces the total modal strain energy.
    """

    K = np.asarray(
        K_hsdt,
        dtype=float,
    )

    q = np.asarray(
        modal_amplitudes,
        dtype=float,
    ).reshape(-1)

    if K.shape != (11, 11):
        raise ValueError(
            "HSDT stiffness must have shape (11, 11)."
        )

    if q.shape != (5,):
        raise ValueError(
            "Navier modal amplitude vector must contain 5 components."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    operators = (
        generalized_strain_basis_operators(
            mode.alpha,
            mode.beta,
        )
    )

    area_factor = (
        float(a)
        * float(b)
        / 4.0
    )

    self_terms = {
        name: 0.0
        for name in GROUPS
    }

    coupling_terms = {}

    group_names = list(
        GROUPS.keys()
    )

    total = 0.0

    for G in operators.values():
        E = G @ q

        total += (
            0.5
            * area_factor
            * float(
                E @ K @ E
            )
        )

        for name in group_names:
            i0, i1 = GROUPS[name]

            Ei = E[i0:i1]
            Kii = K[i0:i1, i0:i1]

            self_terms[name] += (
                0.5
                * area_factor
                * float(
                    Ei @ Kii @ Ei
                )
            )

        for i, name_i in enumerate(
            group_names
        ):
            i0, i1 = GROUPS[name_i]
            Ei = E[i0:i1]

            for name_j in group_names[
                i + 1:
            ]:
                j0, j1 = GROUPS[
                    name_j
                ]

                Ej = E[j0:j1]

                Kij = K[
                    i0:i1,
                    j0:j1,
                ]

                key = (
                    name_i,
                    name_j,
                )

                coupling_terms.setdefault(
                    key,
                    0.0,
                )

                coupling_terms[key] += (
                    area_factor
                    * float(
                        Ei @ Kij @ Ej
                    )
                )

    return ModalEnergyDecomposition(
        self_terms=self_terms,
        coupling_terms=coupling_terms,
        total=float(total),
    )


def reconstructed_total_energy(
    decomposition: ModalEnergyDecomposition,
) -> float:
    """
    Reconstruct total strain energy from all decomposed contributions.
    """

    return float(
        sum(
            decomposition.self_terms.values()
        )
        + sum(
            decomposition.coupling_terms.values()
        )
    )