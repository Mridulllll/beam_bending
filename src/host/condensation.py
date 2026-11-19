"""
Static condensation utilities for the HOST Navier formulation.

For one mode:

    q = [U, V, W, X, Y]^T

is partitioned into

    q_c = [U, V, W]^T
    q_h = [X, Y]^T.

The higher-order amplitudes X,Y can be statically condensed:

    K_eff = K_cc - K_ch K_hh^{-1} K_hc.

This gives the effective classical modal stiffness after allowing
higher-order shear/warping relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModalCondensation:
    K_full: np.ndarray

    K_cc: np.ndarray
    K_ch: np.ndarray
    K_hc: np.ndarray
    K_hh: np.ndarray

    relaxation: np.ndarray
    K_effective: np.ndarray


def condense_higher_order_modal_variables(
    K_modal: np.ndarray,
) -> ModalCondensation:
    """
    Condense X,Y from the five-variable Navier modal system.

    Ordering:

        [U, V, W, X, Y]
    """

    K = np.asarray(
        K_modal,
        dtype=float,
    )

    if K.shape != (5, 5):
        raise ValueError(
            "Navier modal stiffness must have shape (5, 5)."
        )

    classical = np.array(
        [0, 1, 2],
        dtype=int,
    )

    higher = np.array(
        [3, 4],
        dtype=int,
    )

    K_cc = K[
        np.ix_(
            classical,
            classical,
        )
    ]

    K_ch = K[
        np.ix_(
            classical,
            higher,
        )
    ]

    K_hc = K[
        np.ix_(
            higher,
            classical,
        )
    ]

    K_hh = K[
        np.ix_(
            higher,
            higher,
        )
    ]

    try:
        relaxation = (
            K_ch
            @ np.linalg.solve(
                K_hh,
                K_hc,
            )
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Higher-order modal stiffness block is singular."
        ) from exc

    K_effective = (
        K_cc
        - relaxation
    )

    K_effective = 0.5 * (
        K_effective
        + K_effective.T
    )

    return ModalCondensation(
        K_full=K.copy(),
        K_cc=K_cc.copy(),
        K_ch=K_ch.copy(),
        K_hc=K_hc.copy(),
        K_hh=K_hh.copy(),
        relaxation=relaxation,
        K_effective=K_effective,
    )


def condense_to_transverse_stiffness(
    K_modal: np.ndarray,
) -> float:
    """
    Condense all modal variables except W.

    Starting ordering:

        [U, V, W, X, Y].

    Returns the scalar effective transverse modal stiffness k_W such that

        k_W W = f_W

    after U,V,X,Y have been allowed to equilibrate.
    """

    K = np.asarray(
        K_modal,
        dtype=float,
    )

    if K.shape != (5, 5):
        raise ValueError(
            "Navier modal stiffness must have shape (5, 5)."
        )

    w_index = 2

    other = np.array(
        [0, 1, 3, 4],
        dtype=int,
    )

    Kww = float(
        K[
            w_index,
            w_index,
        ]
    )

    Kwo = K[
        w_index,
        other,
    ].reshape(1, -1)

    Kow = K[
        other,
        w_index,
    ].reshape(-1, 1)

    Koo = K[
        np.ix_(
            other,
            other,
        )
    ]

    correction = (
        Kwo
        @ np.linalg.solve(
            Koo,
            Kow,
        )
    ).item()

    return (
        Kww
        - correction
    )


def higher_order_softening_fraction(
    K_modal: np.ndarray,
) -> float:
    """
    Relative reduction of the W modal stiffness due to allowing
    U,V,X,Y to relax:

        s = 1 - k_W_eff / K_WW.

    For a symmetric transverse bending problem, U,V usually decouple,
    so this predominantly measures X,Y higher-order relaxation.
    """

    K = np.asarray(
        K_modal,
        dtype=float,
    )

    raw = float(
        K[2, 2]
    )

    effective = (
        condense_to_transverse_stiffness(
            K
        )
    )

    if raw <= 0.0:
        raise ValueError(
            "Raw transverse modal stiffness must be positive."
        )

    return (
        1.0
        - effective / raw
    )