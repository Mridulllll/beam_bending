"""
Generalized through-thickness stiffness for the HOST sandwich theory.

The linear strain energy per unit midsurface area is

    U = 1/2 E^T K_H E

with

    K_H = integral F(z)^T C(z) F(z) dz.

The implementation integrates each physical layer separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


MatrixFunction = Callable[[float], np.ndarray]
ScalarFunction = Callable[[float], float]


@dataclass(frozen=True)
class Layer:
    """
    One physical sandwich layer.

    Parameters
    ----------
    name
        Human-readable layer identifier.
    z_bottom, z_top
        Layer coordinates relative to the sandwich reference plane.
    stiffness
        Either a constant 6x6 matrix or a function C(z).

        A function is particularly useful for a graded TPMS core:

            C_core(z) = C_hom(rho(z)).
    """

    name: str
    z_bottom: float
    z_top: float
    stiffness: object

    def __post_init__(self):
        if not self.z_bottom < self.z_top:
            raise ValueError(
                "Layer '{}' requires z_bottom < z_top.".format(
                    self.name
                )
            )


@dataclass(frozen=True)
class WarpingFunctions:
    """
    Through-thickness kinematic functions.

    f  : higher-order in-plane warping function
    df : derivative df/dz

    g  : transverse-normal function
    dg : derivative dg/dz

    No particular functional form is imposed by the theory.
    """

    f: ScalarFunction
    df: ScalarFunction
    g: ScalarFunction
    dg: ScalarFunction


def thickness_operator(
    z: float,
    warping: WarpingFunctions,
) -> np.ndarray:
    """
    Evaluate the 6x14 generalized thickness operator F(z).

    Generalized strain ordering:

        0  eps0_xx
        1  eps0_yy
        2  gamma0_xy

        3  kappa_xx
        4  kappa_yy
        5  kappa_xy

        6  chi_xx
        7  chi_yy
        8  chi_xy

        9  beta_x
        10 beta_y

        11 eta_x
        12 eta_y

        13 eta
    """

    z = float(z)

    f = float(warping.f(z))
    df = float(warping.df(z))
    g = float(warping.g(z))
    dg = float(warping.dg(z))

    F = np.zeros((6, 14), dtype=float)

    # eps_xx
    F[0, 0] = 1.0
    F[0, 3] = z
    F[0, 6] = f

    # eps_yy
    F[1, 1] = 1.0
    F[1, 4] = z
    F[1, 7] = f

    # eps_zz
    F[2, 13] = dg

    # gamma_xy
    F[3, 2] = 1.0
    F[3, 5] = z
    F[3, 8] = f

    # gamma_xz
    F[4, 9] = df
    F[4, 11] = g

    # gamma_yz
    F[5, 10] = df
    F[5, 12] = g

    return F


def _evaluate_stiffness(
    stiffness: object,
    z: float,
) -> np.ndarray:
    """
    Evaluate either a constant constitutive matrix or C(z).
    """

    if callable(stiffness):
        C = stiffness(float(z))
    else:
        C = stiffness

    C = np.asarray(C, dtype=float)

    if C.shape != (6, 6):
        raise ValueError(
            "Layer stiffness must evaluate to shape (6, 6); "
            "got {}.".format(C.shape)
        )

    return C


def integrate_layer_generalized_stiffness(
    layer: Layer,
    warping: WarpingFunctions,
    *,
    quadrature_order: int = 8,
) -> np.ndarray:
    """
    Numerically integrate one layer contribution

        K_layer = integral F^T C F dz

    using Gauss-Legendre quadrature.

    Each physical layer is integrated separately, so discontinuities
    between face sheets and core do not contaminate the quadrature.
    """

    if quadrature_order < 1:
        raise ValueError("quadrature_order must be at least 1.")

    xi, wi = np.polynomial.legendre.leggauss(quadrature_order)

    z0 = float(layer.z_bottom)
    z1 = float(layer.z_top)

    jac = 0.5 * (z1 - z0)
    center = 0.5 * (z1 + z0)

    K = np.zeros((14, 14), dtype=float)

    for xi_i, wi_i in zip(xi, wi):
        z = center + jac * float(xi_i)

        F = thickness_operator(z, warping)
        C = _evaluate_stiffness(layer.stiffness, z)

        K += (
            float(wi_i)
            * jac
            * (F.T @ C @ F)
        )

    return 0.5 * (K + K.T)


def assemble_generalized_stiffness(
    layers: Sequence[Layer],
    warping: WarpingFunctions,
    *,
    quadrature_order: int = 8,
) -> np.ndarray:
    """
    Assemble the total 14x14 generalized sandwich stiffness.
    """

    if len(layers) == 0:
        raise ValueError("At least one sandwich layer is required.")

    K = np.zeros((14, 14), dtype=float)

    for layer in layers:
        K += integrate_layer_generalized_stiffness(
            layer,
            warping,
            quadrature_order=quadrature_order,
        )

    return 0.5 * (K + K.T)


def classical_abd(
    layers: Sequence[Layer],
    *,
    quadrature_order: int = 8,
):
    """
    Independently calculate the classical A, B and D matrices.

    The relevant 3-D HOST indices are

        [xx, yy, xy] -> [0, 1, 3].

    This routine is intentionally independent of the 14x14 generalized
    stiffness construction and is therefore suitable for verification.
    """

    if len(layers) == 0:
        raise ValueError("At least one sandwich layer is required.")

    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    D = np.zeros((3, 3), dtype=float)

    idx = np.asarray((0, 1, 3), dtype=int)

    for layer in layers:
        xi, wi = np.polynomial.legendre.leggauss(quadrature_order)

        z0 = float(layer.z_bottom)
        z1 = float(layer.z_top)

        jac = 0.5 * (z1 - z0)
        center = 0.5 * (z1 + z0)

        for xi_i, wi_i in zip(xi, wi):
            z = center + jac * float(xi_i)

            C = _evaluate_stiffness(
                layer.stiffness,
                z,
            )

            Q = C[np.ix_(idx, idx)]

            weight = float(wi_i) * jac

            A += weight * Q
            B += weight * z * Q
            D += weight * z**2 * Q

    return A, B, D


def extract_abd_from_generalized_stiffness(
    K: np.ndarray,
):
    """
    Extract A/B/D sub-blocks from the generalized 14x14 matrix.

    Ordering:

        membrane : 0:3
        bending  : 3:6
    """

    K = np.asarray(K, dtype=float)

    if K.shape != (14, 14):
        raise ValueError(
            "Generalized stiffness must have shape (14, 14)."
        )

    A = K[0:3, 0:3].copy()
    B = K[0:3, 3:6].copy()
    D = K[3:6, 3:6].copy()

    return A, B, D


def generalized_strain_energy_density(
    Egen: np.ndarray,
    K: np.ndarray,
) -> float:
    """
    Return strain energy per unit midsurface area:

        U = 1/2 E^T K E.
    """

    Egen = np.asarray(Egen, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)

    if Egen.shape != (14,):
        raise ValueError(
            "Generalized strain vector must contain 14 components."
        )

    if K.shape != (14, 14):
        raise ValueError(
            "Generalized stiffness must have shape (14, 14)."
        )

    return 0.5 * float(Egen @ K @ Egen)

# ==================================================================
# Directional HOST generalized stiffness
# ==================================================================


@dataclass(frozen=True)
class DirectionalWarpingFunctions:
    """
    Direction-dependent through-thickness kinematic functions.

    fx  : x-directed in-plane warping function
    dfx : derivative dfx/dz

    fy  : y-directed in-plane warping function
    dfy : derivative dfy/dz

    g   : transverse-normal function
    dg  : derivative dg/dz
    """

    fx: ScalarFunction
    dfx: ScalarFunction

    fy: ScalarFunction
    dfy: ScalarFunction

    g: ScalarFunction
    dg: ScalarFunction


def directional_thickness_operator(
    z: float,
    warping: DirectionalWarpingFunctions,
) -> np.ndarray:
    """
    Evaluate the 6x15 directional HOST thickness operator F(z).

    Generalized strain ordering:

        0   eps0_xx
        1   eps0_yy
        2   gamma0_xy

        3   kappa_xx
        4   kappa_yy
        5   kappa_xy

        6   chi_xx
        7   chi_yy

        8   chi_xy_x = beta_x,y
        9   chi_xy_y = beta_y,x

        10  beta_x
        11  beta_y

        12  eta_x
        13  eta_y

        14  eta
    """

    z = float(z)

    fx = float(
        warping.fx(z)
    )

    fy = float(
        warping.fy(z)
    )

    dfx = float(
        warping.dfx(z)
    )

    dfy = float(
        warping.dfy(z)
    )

    g = float(
        warping.g(z)
    )

    dg = float(
        warping.dg(z)
    )

    F = np.zeros(
        (6, 15),
        dtype=float,
    )

    # --------------------------------------------------------------
    # eps_xx
    # --------------------------------------------------------------

    F[0, 0] = 1.0
    F[0, 3] = z
    F[0, 6] = fx

    # --------------------------------------------------------------
    # eps_yy
    # --------------------------------------------------------------

    F[1, 1] = 1.0
    F[1, 4] = z
    F[1, 7] = fy

    # --------------------------------------------------------------
    # eps_zz
    # --------------------------------------------------------------

    F[2, 14] = dg

    # --------------------------------------------------------------
    # gamma_xy
    #
    # Directional warping requires independent
    #
    #     beta_x,y
    #     beta_y,x
    #
    # terms.
    # --------------------------------------------------------------

    F[3, 2] = 1.0
    F[3, 5] = z

    F[3, 8] = fx
    F[3, 9] = fy

    # --------------------------------------------------------------
    # gamma_xz
    # --------------------------------------------------------------

    F[4, 10] = dfx
    F[4, 12] = g

    # --------------------------------------------------------------
    # gamma_yz
    # --------------------------------------------------------------

    F[5, 11] = dfy
    F[5, 13] = g

    return F


def integrate_layer_directional_generalized_stiffness(
    layer: Layer,
    warping: DirectionalWarpingFunctions,
    *,
    quadrature_order: int = 8,
) -> np.ndarray:
    """
    Integrate one physical layer contribution to the directional
    15x15 HOST generalized stiffness:

        K_layer =
        integral F_dir(z)^T C(z) F_dir(z) dz.
    """

    if quadrature_order < 1:
        raise ValueError(
            "quadrature_order must be at least 1."
        )

    xi, wi = (
        np.polynomial.legendre.leggauss(
            quadrature_order
        )
    )

    z0 = float(
        layer.z_bottom
    )

    z1 = float(
        layer.z_top
    )

    jac = 0.5 * (
        z1 - z0
    )

    center = 0.5 * (
        z1 + z0
    )

    K = np.zeros(
        (15, 15),
        dtype=float,
    )

    for xi_i, wi_i in zip(
        xi,
        wi,
    ):
        z = (
            center
            + jac * float(
                xi_i
            )
        )

        F = directional_thickness_operator(
            z,
            warping,
        )

        C = _evaluate_stiffness(
            layer.stiffness,
            z,
        )

        K += (
            float(wi_i)
            * jac
            * (
                F.T
                @ C
                @ F
            )
        )

    return 0.5 * (
        K + K.T
    )


def assemble_directional_generalized_stiffness(
    layers: Sequence[Layer],
    warping: DirectionalWarpingFunctions,
    *,
    quadrature_order: int = 8,
) -> np.ndarray:
    """
    Assemble the complete directional 15x15 HOST stiffness.
    """

    if len(layers) == 0:
        raise ValueError(
            "At least one sandwich layer is required."
        )

    K = np.zeros(
        (15, 15),
        dtype=float,
    )

    for layer in layers:
        K += (
            integrate_layer_directional_generalized_stiffness(
                layer,
                warping,
                quadrature_order=quadrature_order,
            )
        )

    return 0.5 * (
        K + K.T
    )


def scalar_to_directional_warping(
    warping: WarpingFunctions,
) -> DirectionalWarpingFunctions:
    """
    Embed the historical scalar warping functions into the
    directional formulation:

        fx  = fy  = f
        dfx = dfy = df

    This is the exact scalar limit of the directional theory.
    """

    return DirectionalWarpingFunctions(
        fx=warping.f,
        dfx=warping.df,

        fy=warping.f,
        dfy=warping.df,

        g=warping.g,
        dg=warping.dg,
    )


def directional_generalized_strain_energy_density(
    Egen: np.ndarray,
    K: np.ndarray,
) -> float:
    """
    Directional HOST generalized strain energy per unit area:

        U = 1/2 E^T K E.
    """

    Egen = np.asarray(
        Egen,
        dtype=float,
    ).reshape(-1)

    K = np.asarray(
        K,
        dtype=float,
    )

    if Egen.shape != (
        15,
    ):
        raise ValueError(
            "Directional generalized strain vector must "
            "contain 15 components."
        )

    if K.shape != (
        15,
        15,
    ):
        raise ValueError(
            "Directional generalized stiffness must "
            "have shape (15, 15)."
        )

    return 0.5 * float(
        Egen
        @ K
        @ Egen
    )
