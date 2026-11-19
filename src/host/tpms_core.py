"""
Homogenized TPMS-core constitutive support for HOST.

Permanent HOST Voigt ordering:

    [xx, yy, zz, xy, xz, yz]

The input tensor may come directly from periodic RVE homogenization.

For the five-field HOST baseline:
    - sigma_zz = 0 is condensed,
    - xz and yz transverse shear stiffnesses are retained.

The complete 6x6 homogenized tensor is preserved so the analytical
sandwich model does not reduce the TPMS core to a scalar modulus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from host.constitutive import (
    sigma_zz_zero_stiffness,
    validate_stiffness_matrix,
)


@dataclass(frozen=True)
class HomogenizedTPMSCore:
    C_3d: np.ndarray
    C_hsdt: np.ndarray

    G_xy: float
    G_xz: float
    G_yz: float

    shear_anisotropy_xz_yz: float


def build_homogenized_tpms_core(
    C_hom: np.ndarray,
) -> HomogenizedTPMSCore:
    """
    Prepare a homogenized TPMS stiffness tensor for HOST.

    Parameters
    ----------
    C_hom
        Full homogenized 6x6 stiffness tensor in HOST ordering

            [xx, yy, zz, xy, xz, yz].

    Returns
    -------
    HomogenizedTPMSCore
        Original 3-D tensor, five-field reduced tensor, and useful
        transverse-shear diagnostics.
    """

    C3 = validate_stiffness_matrix(
        C_hom,
        positive_definite=True,
    )

    C5 = sigma_zz_zero_stiffness(
        C3
    )

    G_xy = float(
        C5[3, 3]
    )

    G_xz = float(
        C5[4, 4]
    )

    G_yz = float(
        C5[5, 5]
    )

    if min(
        G_xy,
        G_xz,
        G_yz,
    ) <= 0.0:
        raise ValueError(
            "Homogenized shear stiffnesses must be positive."
        )

    shear_anisotropy = (
        max(
            G_xz,
            G_yz,
        )
        / min(
            G_xz,
            G_yz,
        )
    )

    return HomogenizedTPMSCore(
        C_3d=C3.copy(),
        C_hsdt=C5.copy(),
        G_xy=G_xy,
        G_xz=G_xz,
        G_yz=G_yz,
        shear_anisotropy_xz_yz=float(
            shear_anisotropy
        ),
    )


def representative_transverse_shear_modulus(
    core: HomogenizedTPMSCore,
    *,
    relative_tolerance: float = 0.05,
) -> float:
    """
    Return a scalar transverse shear modulus for the current
    single-warping-function HOST baseline.

    If G_xz and G_yz differ by more than relative_tolerance, an
    exception is raised rather than silently averaging anisotropic
    behavior.

    A future orthotropic extension can introduce separate f_x(z)
    and f_y(z) warping functions.
    """

    Gx = float(
        core.G_xz
    )

    Gy = float(
        core.G_yz
    )

    relative_difference = (
        abs(
            Gx - Gy
        )
        / max(
            abs(Gx),
            abs(Gy),
        )
    )

    if relative_difference > relative_tolerance:
        raise ValueError(
            "G_xz and G_yz differ by {:.2f}%. "
            "The current scalar-warping HOST baseline is not "
            "appropriate; direction-dependent warping is required."
            .format(
                100.0
                * relative_difference
            )
        )

    return 0.5 * (
        Gx + Gy
    )