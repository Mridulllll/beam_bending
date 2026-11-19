"""
Three-dimensional constitutive utilities for the HOST sandwich theory.

Permanent engineering-strain/stress ordering:

    [xx, yy, zz, xy, xz, yz]

i.e.

    strain =
    [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_xz, gamma_yz]^T

    stress =
    [sig_xx, sig_yy, sig_zz, tau_xy, tau_xz, tau_yz]^T

This ordering must be used consistently throughout the repository.
"""

from __future__ import annotations

import numpy as np


VOIGT_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")

# In-plane components relevant to classical A/B/D theory:
# [eps_xx, eps_yy, gamma_xy]
INPLANE_INDICES = (0, 1, 3)


def validate_stiffness_matrix(
    C: np.ndarray,
    *,
    symmetry_tol: float = 1.0e-10,
    positive_definite: bool = True,
) -> np.ndarray:
    """
    Validate a 6x6 three-dimensional constitutive stiffness matrix.

    Parameters
    ----------
    C
        Constitutive matrix in HOST Voigt ordering:
        [xx, yy, zz, xy, xz, yz].
    symmetry_tol
        Relative tolerance for symmetry verification.
    positive_definite
        If True, require all eigenvalues to be strictly positive.

    Returns
    -------
    np.ndarray
        Validated floating-point copy of C.
    """

    C = np.asarray(C, dtype=float)

    if C.shape != (6, 6):
        raise ValueError(
            "Constitutive stiffness matrix must have shape (6, 6); "
            "got {}.".format(C.shape)
        )

    scale = max(1.0, float(np.linalg.norm(C, ord=np.inf)))
    skew = float(np.linalg.norm(C - C.T, ord=np.inf))

    if skew > symmetry_tol * scale:
        raise ValueError(
            "Constitutive stiffness matrix is not symmetric within "
            "the requested tolerance."
        )

    if positive_definite:
        eigvals = np.linalg.eigvalsh(0.5 * (C + C.T))

        if np.min(eigvals) <= 0.0:
            raise ValueError(
                "Constitutive stiffness matrix must be positive definite."
            )

    return C.copy()


def isotropic_stiffness_3d(
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Return the full isotropic 3-D stiffness matrix.

    HOST ordering:
        [xx, yy, zz, xy, xz, yz]

    Engineering shear strains are used.
    """

    E = float(E)
    nu = float(nu)

    if E <= 0.0:
        raise ValueError("Young's modulus must be positive.")

    if not (-1.0 < nu < 0.5):
        raise ValueError(
            "Poisson's ratio for isotropic linear elasticity must satisfy "
            "-1 < nu < 0.5."
        )

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    C = np.zeros((6, 6), dtype=float)

    C[0, 0] = lam + 2.0 * mu
    C[1, 1] = lam + 2.0 * mu
    C[2, 2] = lam + 2.0 * mu

    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam

    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu

    return validate_stiffness_matrix(C)


def inplane_submatrix(C: np.ndarray) -> np.ndarray:
    """
    Extract the [xx, yy, xy] constitutive submatrix.

    No plane-stress reduction is performed here. This function merely
    extracts components from the supplied 3-D constitutive tensor.

    The distinction between:
        - direct 3-D constitutive integration,
        - plane-stress condensation,
        - quasi-3-D theory

    is deliberately kept explicit.
    """

    C = validate_stiffness_matrix(
        C,
        positive_definite=False,
    )

    idx = np.asarray(INPLANE_INDICES, dtype=int)

    return C[np.ix_(idx, idx)].copy()

def plane_stress_condensation(C):
    """
    Condense a 3-D constitutive matrix to the in-plane plane-stress law.

    Permanent 3-D Voigt ordering:

        [xx, yy, zz, xy, xz, yz]

    Retained in-plane components:

        a = [xx, yy, xy]

    Eliminated transverse components:

        b = [zz, xz, yz]

    Imposing

        sigma_b = 0

    gives

        epsilon_b = -C_bb^{-1} C_ba epsilon_a

    and therefore

        C_ps = C_aa - C_ab C_bb^{-1} C_ba.

    This is a Schur-complement condensation and remains valid for a
    general anisotropic symmetric 3-D constitutive tensor.
    """

    import numpy as np

    C = np.asarray(C, dtype=float)

    if C.shape != (6, 6):
        raise ValueError(
            "3-D stiffness matrix must have shape (6, 6)."
        )

    a = np.array(
        [0, 1, 3],
        dtype=int,
    )

    b = np.array(
        [2, 4, 5],
        dtype=int,
    )

    Caa = C[np.ix_(a, a)]
    Cab = C[np.ix_(a, b)]
    Cba = C[np.ix_(b, a)]
    Cbb = C[np.ix_(b, b)]

    try:
        correction = (
            Cab
            @ np.linalg.solve(
                Cbb,
                Cba,
            )
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Transverse constitutive block is singular; "
            "plane-stress condensation is not possible."
        ) from exc

    Cps = Caa - correction

    return 0.5 * (
        Cps + Cps.T
    )


def sigma_zz_zero_stiffness(C):
    """
    Reduce a full 3-D stiffness tensor for the five-field HSDT by imposing

        sigma_zz = 0

    while RETAINING transverse shear stiffnesses xz and yz.

    Permanent HOST Voigt ordering:

        [xx, yy, zz, xy, xz, yz]

    Active strain/stress components:

        a = [xx, yy, xy, xz, yz]

    Eliminated component:

        b = [zz]

    From

        sigma_zz = C_ba epsilon_a + C_bb epsilon_zz = 0

    follows

        epsilon_zz = -C_bb^{-1} C_ba epsilon_a

    and the reduced active constitutive matrix

        C_red = C_aa - C_ab C_bb^{-1} C_ba.

    The result is embedded back into a 6x6 matrix with a zero zz
    row and column so it can be used directly by the existing
    generalized-stiffness machinery.

    This is the appropriate constitutive reduction for the five-field
    HSDT baseline. It is NOT the quasi-3D constitutive model.
    """

    import numpy as np

    C = np.asarray(C, dtype=float)

    if C.shape != (6, 6):
        raise ValueError(
            "3-D stiffness matrix must have shape (6, 6)."
        )

    active = np.array(
        [0, 1, 3, 4, 5],
        dtype=int,
    )

    zz = np.array(
        [2],
        dtype=int,
    )

    Caa = C[np.ix_(active, active)]
    Cab = C[np.ix_(active, zz)]
    Cba = C[np.ix_(zz, active)]
    Cbb = C[np.ix_(zz, zz)]

    try:
        correction = (
            Cab
            @ np.linalg.solve(
                Cbb,
                Cba,
            )
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "C_zz block is singular; sigma_zz=0 condensation "
            "is not possible."
        ) from exc

    Cred = Caa - correction
    Cred = 0.5 * (Cred + Cred.T)

    Ceff = np.zeros(
        (6, 6),
        dtype=float,
    )

    Ceff[np.ix_(active, active)] = Cred

    return Ceff
