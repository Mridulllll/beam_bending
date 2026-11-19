"""
Reduced one-dimensional HOST beam bending formulation.

Starting from the beam-section constitutive relation

    [N]   [K00 K0t] [eps0]
    [r] = [Kt0 Ktt] [ e  ]

with

    r = [M, H, Q]^T

and

    e = [kappa, chi, beta]^T,

the free-axial-force condition N = 0 is condensed exactly.

The reduced bending/shear constitutive relation is

    r = K3 e,

where

    K3 = Ktt - Kt0 K00^{-1} K0t.

Kinematics:

    theta = w'
    kappa = -w''
    chi   = beta'

Equilibrium:

    M'' + q = 0
    H' - Q  = 0

using the sign convention that positive q acts in the positive
transverse displacement direction.

A convenient first-order state is

    y = [w, theta, beta, M, V, H]^T,

where

    V = M'.

For a uniform beam section this gives a constant-coefficient
state-space problem that can be propagated exactly using a matrix
exponential.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class ReducedHOSTBeamSection:
    """
    Axially condensed HOST beam section.

    Attributes
    ----------
    K3
        Reduced constitutive matrix for

            [kappa, chi, beta]

        ->

            [M, H, Q].

    axial_stiffness
        Original K00 term.

    axial_coupling
        Original K0t row coupling eps0 to
        [kappa, chi, beta].
    """

    K3: np.ndarray
    axial_stiffness: float
    axial_coupling: np.ndarray


def _validate_K4(
    K4: np.ndarray,
) -> np.ndarray:
    K4 = np.asarray(
        K4,
        dtype=float,
    )

    if K4.shape != (
        4,
        4,
    ):
        raise ValueError(
            "K4 must have shape (4, 4)."
        )

    if not np.allclose(
        K4,
        K4.T,
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError(
            "K4 must be symmetric."
        )

    if not np.all(
        np.isfinite(
            K4
        )
    ):
        raise ValueError(
            "K4 contains non-finite values."
        )

    return K4


def _validate_K3(
    K3: np.ndarray,
) -> np.ndarray:
    K3 = np.asarray(
        K3,
        dtype=float,
    )

    if K3.shape != (
        3,
        3,
    ):
        raise ValueError(
            "K3 must have shape (3, 3)."
        )

    if not np.allclose(
        K3,
        K3.T,
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError(
            "K3 must be symmetric."
        )

    eigenvalues = np.linalg.eigvalsh(
        K3
    )

    if np.min(
        eigenvalues
    ) <= 0.0:
        raise ValueError(
            "K3 must be positive definite."
        )

    return K3


def condense_zero_axial_force(
    K4: np.ndarray,
) -> ReducedHOSTBeamSection:
    """
    Condense eps0 under the free-axial-force condition

        N = 0.

    Given

        N = K00 eps0 + K0t e,

    then

        eps0 = -K00^{-1} K0t e.

    The reduced section stiffness is

        K3 =
        Ktt - Kt0 K00^{-1} K0t.
    """

    K4 = _validate_K4(
        K4
    )

    K00 = float(
        K4[0, 0]
    )

    if K00 <= 0.0:
        raise ValueError(
            "Axial stiffness K00 must be positive."
        )

    K0t = np.asarray(
        K4[
            0,
            1:4,
        ],
        dtype=float,
    )

    Kt0 = np.asarray(
        K4[
            1:4,
            0,
        ],
        dtype=float,
    )

    Ktt = np.asarray(
        K4[
            1:4,
            1:4,
        ],
        dtype=float,
    )

    K3 = (
        Ktt
        - np.outer(
            Kt0,
            K0t,
        )
        / K00
    )

    K3 = 0.5 * (
        K3
        + K3.T
    )

    _validate_K3(
        K3
    )

    return ReducedHOSTBeamSection(
        K3=K3,
        axial_stiffness=K00,
        axial_coupling=K0t.copy(),
    )


def recover_axial_strain(
    section: ReducedHOSTBeamSection,
    reduced_strain: np.ndarray,
    *,
    axial_force: float = 0.0,
) -> float:
    """
    Recover eps0 for a prescribed axial resultant N.

    For

        N = K00 eps0 + K0t e,

    the exact recovery is

        eps0 =
        (N - K0t e) / K00.
    """

    e = np.asarray(
        reduced_strain,
        dtype=float,
    )

    if e.shape != (
        3,
    ):
        raise ValueError(
            "reduced_strain must have shape (3,)."
        )

    return float(
        (
            float(axial_force)
            - section.axial_coupling
            @ e
        )
        / section.axial_stiffness
    )


def recover_full_generalized_strain(
    section: ReducedHOSTBeamSection,
    reduced_strain: np.ndarray,
    *,
    axial_force: float = 0.0,
) -> np.ndarray:
    """
    Return

        [eps0, kappa, chi, beta]^T.
    """

    e = np.asarray(
        reduced_strain,
        dtype=float,
    )

    eps0 = recover_axial_strain(
        section,
        e,
        axial_force=axial_force,
    )

    return np.concatenate(
        (
            np.asarray(
                [
                    eps0,
                ],
                dtype=float,
            ),
            e,
        )
    )


def reduced_resultants(
    section: ReducedHOSTBeamSection,
    reduced_strain: np.ndarray,
) -> np.ndarray:
    """
    Return

        [M, H, Q]^T = K3 [kappa, chi, beta]^T

    for N = 0.
    """

    e = np.asarray(
        reduced_strain,
        dtype=float,
    )

    if e.shape != (
        3,
    ):
        raise ValueError(
            "reduced_strain must have shape (3,)."
        )

    return (
        section.K3
        @ e
    )


def state_generalized_quantities(
    K3: np.ndarray,
    state: np.ndarray,
):
    """
    Recover the generalized strains and Q from the state

        y = [w, theta, beta, M, V, H].

    Since beta is simultaneously one of the generalized strains,
    Q is determined algebraically so that

        [kappa, chi, beta]^T
        =
        inv(K3) [M, H, Q]^T.

    Returns
    -------
    kappa, chi, beta, M, V, H, Q
    """

    K3 = _validate_K3(
        K3
    )

    y = np.asarray(
        state,
        dtype=float,
    )

    if y.shape != (
        6,
    ):
        raise ValueError(
            "state must have shape (6,)."
        )

    beta = float(
        y[2]
    )

    M = float(
        y[3]
    )

    V = float(
        y[4]
    )

    H = float(
        y[5]
    )

    compliance = np.linalg.inv(
        K3
    )

    S20 = float(
        compliance[2, 0]
    )

    S21 = float(
        compliance[2, 1]
    )

    S22 = float(
        compliance[2, 2]
    )

    if S22 <= 0.0:
        raise ValueError(
            "Reduced beam compliance S22 must be positive."
        )

    Q = (
        beta
        - S20 * M
        - S21 * H
    ) / S22

    resultants = np.asarray(
        [
            M,
            H,
            Q,
        ],
        dtype=float,
    )

    strains = np.linalg.solve(
        K3,
        resultants,
    )

    kappa = float(
        strains[0]
    )

    chi = float(
        strains[1]
    )

    beta_recovered = float(
        strains[2]
    )

    return {
        "kappa": kappa,
        "chi": chi,
        "beta": beta_recovered,
        "M": M,
        "V": V,
        "H": H,
        "Q": float(Q),
    }


def state_derivative(
    K3: np.ndarray,
    state: np.ndarray,
    *,
    q: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the HOST beam first-order state equations.

    State ordering:

        0 w
        1 theta = w'
        2 beta
        3 M
        4 V = M'
        5 H

    Differential equations:

        w'     = theta
        theta' = -kappa
        beta'  = chi
        M'     = V
        V'     = -q
        H'     = Q
    """

    y = np.asarray(
        state,
        dtype=float,
    )

    if y.shape != (
        6,
    ):
        raise ValueError(
            "state must have shape (6,)."
        )

    values = state_generalized_quantities(
        K3,
        y,
    )

    dydx = np.zeros(
        6,
        dtype=float,
    )

    dydx[0] = y[1]

    dydx[1] = (
        -values[
            "kappa"
        ]
    )

    dydx[2] = values[
        "chi"
    ]

    dydx[3] = y[4]

    dydx[4] = (
        -float(q)
    )

    dydx[5] = values[
        "Q"
    ]

    return dydx


def beam_state_space(
    K3: np.ndarray,
):
    """
    Return constant state-space matrices

        y' = A y + b q

    for a uniform HOST beam.

    The matrices are constructed directly from the constitutive
    state equations to avoid duplicating their algebra.
    """

    K3 = _validate_K3(
        K3
    )

    A = np.zeros(
        (
            6,
            6,
        ),
        dtype=float,
    )

    for j in range(
        6
    ):
        basis = np.zeros(
            6,
            dtype=float,
        )

        basis[j] = 1.0

        A[:, j] = state_derivative(
            K3,
            basis,
            q=0.0,
        )

    zero = np.zeros(
        6,
        dtype=float,
    )

    b = state_derivative(
        K3,
        zero,
        q=1.0,
    )

    return (
        A,
        b,
    )


def propagate_uniform_segment(
    K3: np.ndarray,
    state_start: np.ndarray,
    length: float,
    *,
    q: float = 0.0,
) -> np.ndarray:
    """
    Propagate the HOST beam state exactly over a uniform segment.

    For constant K3 and constant distributed load q,

        y' = A y + b q

    is integrated through an augmented matrix exponential.

    No longitudinal discretization is introduced.
    """

    K3 = _validate_K3(
        K3
    )

    y0 = np.asarray(
        state_start,
        dtype=float,
    )

    if y0.shape != (
        6,
    ):
        raise ValueError(
            "state_start must have shape (6,)."
        )

    length = float(
        length
    )

    if length < 0.0:
        raise ValueError(
            "Segment length must be non-negative."
        )

    if length == 0.0:
        return y0.copy()

    A, b = beam_state_space(
        K3
    )

    augmented = np.zeros(
        (
            7,
            7,
        ),
        dtype=float,
    )

    augmented[
        0:6,
        0:6,
    ] = A

    augmented[
        0:6,
        6,
    ] = (
        b
        * float(q)
    )

    transfer = expm(
        augmented
        * length
    )

    augmented_state = np.ones(
        7,
        dtype=float,
    )

    augmented_state[
        0:6
    ] = y0

    result = (
        transfer
        @ augmented_state
    )

    return np.asarray(
        result[
            0:6
        ],
        dtype=float,
    )