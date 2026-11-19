"""
Five-field HSDT baseline extracted from the complete HOST formulation.

Generalized HSDT strain ordering:

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

The quasi-3D eta sector is excluded from this baseline.
"""

from __future__ import annotations

import numpy as np


HSDT_GENERALIZED_SIZE = 11


def extract_hsdt_stiffness(
    K_generalized: np.ndarray,
) -> np.ndarray:
    """
    Extract the five-field HSDT 11x11 constitutive matrix from the
    complete 14x14 generalized HOST stiffness matrix.
    """

    K_generalized = np.asarray(
        K_generalized,
        dtype=float,
    )

    if K_generalized.shape != (14, 14):
        raise ValueError(
            "Complete generalized stiffness must have shape (14, 14)."
        )

    return K_generalized[:11, :11].copy()


def hsdt_strain_energy_density(
    E_hsdt: np.ndarray,
    K_hsdt: np.ndarray,
) -> float:
    """
    Five-field HSDT strain energy per unit midsurface area:

        U = 1/2 E_H^T K_H E_H.
    """

    E_hsdt = np.asarray(
        E_hsdt,
        dtype=float,
    ).reshape(-1)

    K_hsdt = np.asarray(
        K_hsdt,
        dtype=float,
    )

    if E_hsdt.shape != (11,):
        raise ValueError(
            "HSDT generalized strain vector must have 11 components."
        )

    if K_hsdt.shape != (11, 11):
        raise ValueError(
            "HSDT stiffness must have shape (11, 11)."
        )

    return 0.5 * float(
        E_hsdt @ K_hsdt @ E_hsdt
    )

# ==================================================================
# Directional five-field HSDT
# ==================================================================

DIRECTIONAL_HSDT_GENERALIZED_SIZE = 12


def extract_directional_hsdt_stiffness(
    K_generalized: np.ndarray,
) -> np.ndarray:
    """
    Extract the directional five-field HSDT constitutive matrix from
    the complete 15x15 directional HOST generalized stiffness.

    Directional five-field ordering:

        [eps0_xx,
         eps0_yy,
         gamma0_xy,

         kappa_xx,
         kappa_yy,
         kappa_xy,

         chi_xx,
         chi_yy,
         chi_xy_x,
         chi_xy_y,

         beta_x,
         beta_y]^T

    The quasi-3D eta sector is excluded.
    """

    K_generalized = np.asarray(
        K_generalized,
        dtype=float,
    )

    if K_generalized.shape != (15, 15):
        raise ValueError(
            "Complete directional generalized stiffness must "
            "have shape (15, 15)."
        )

    return K_generalized[:12, :12].copy()


def directional_hsdt_strain_energy_density(
    E_hsdt: np.ndarray,
    K_hsdt: np.ndarray,
) -> float:
    """
    Directional five-field HSDT strain energy per unit midsurface area:

        U = 1/2 E_H^T K_H E_H.
    """

    E_hsdt = np.asarray(
        E_hsdt,
        dtype=float,
    ).reshape(-1)

    K_hsdt = np.asarray(
        K_hsdt,
        dtype=float,
    )

    if E_hsdt.shape != (12,):
        raise ValueError(
            "Directional HSDT generalized strain vector must "
            "have 12 components."
        )

    if K_hsdt.shape != (12, 12):
        raise ValueError(
            "Directional HSDT stiffness must have shape (12, 12)."
        )

    return 0.5 * float(
        E_hsdt @ K_hsdt @ E_hsdt
    )