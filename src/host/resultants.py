"""
Generalized stress resultants for the five-field HOST HSDT.

Generalized strain ordering:

    E_H =
    [eps0_xx,
     eps0_yy,
     gamma0_xy,

     kappa_xx,
     kappa_yy,
     kappa_xy,

     chi_xx,
     chi_yy,
     chi_xy,

     beta_x,
     beta_y]^T

The energetically conjugate generalized resultants are

    R_H =
    [N_xx,
     N_yy,
     N_xy,

     M_xx,
     M_yy,
     M_xy,

     P_xx,
     P_yy,
     P_xy,

     Q_x,
     Q_y]^T

with

    R_H = K_H E_H.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HSDT_RESULTANT_SIZE = 11


@dataclass(frozen=True)
class HSDTResultants:
    N: np.ndarray
    M: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    full: np.ndarray


def generalized_resultants(
    K_hsdt: np.ndarray,
    E_hsdt: np.ndarray,
) -> HSDTResultants:
    """
    Evaluate the generalized stress resultants

        R = K_H E_H.
    """

    K_hsdt = np.asarray(
        K_hsdt,
        dtype=float,
    )

    E_hsdt = np.asarray(
        E_hsdt,
        dtype=float,
    ).reshape(-1)

    if K_hsdt.shape != (11, 11):
        raise ValueError(
            "HSDT stiffness must have shape (11, 11)."
        )

    if E_hsdt.shape != (11,):
        raise ValueError(
            "HSDT generalized strain vector must contain 11 components."
        )

    R = K_hsdt @ E_hsdt

    return HSDTResultants(
        N=R[0:3].copy(),
        M=R[3:6].copy(),
        P=R[6:9].copy(),
        Q=R[9:11].copy(),
        full=R.copy(),
    )


def internal_virtual_work_density(
    delta_E_hsdt: np.ndarray,
    resultants: HSDTResultants,
) -> float:
    """
    Internal virtual-work density

        delta(U) = delta(E_H)^T R_H.
    """

    delta_E_hsdt = np.asarray(
        delta_E_hsdt,
        dtype=float,
    ).reshape(-1)

    if delta_E_hsdt.shape != (11,):
        raise ValueError(
            "Virtual generalized strain vector must contain 11 components."
        )

    return float(
        delta_E_hsdt @ resultants.full
    )