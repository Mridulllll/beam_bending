"""
Generalized strain representations for HOST sandwich theory.

Two formulations are retained:

1. Scalar-warping HOST
   -------------------
   Uses one common in-plane warping function

       fx(z) = fy(z) = f(z)

   and the historical 14-component generalized strain vector.

2. Directional-warping HOST
   ------------------------
   Uses independent functions

       fx(z), fy(z)

   and therefore requires separate generalized twisting-warping
   components

       chi_xy_x = beta_x,y
       chi_xy_y = beta_y,x.

   The resulting generalized strain vector contains 15 components.

The directional formulation reduces exactly to the scalar formulation
when fx = fy.
"""

from dataclasses import dataclass, replace

import sympy as sp

from host.kinematics import (
    SymbolicKinematics,
    build_symbolic_kinematics,
)


@dataclass(frozen=True)
class GeneralizedStrainSystem:
    """
    Historical scalar-warping 14-component HOST system.
    """

    kin: SymbolicKinematics
    E: sp.Matrix
    F: sp.Matrix
    reconstructed_strain: sp.Matrix


@dataclass(frozen=True)
class DirectionalGeneralizedStrainSystem:
    """
    Directional 15-component HOST generalized strain system.
    """

    kin: SymbolicKinematics
    E: sp.Matrix
    F: sp.Matrix
    reconstructed_strain: sp.Matrix


def _scalarized_kinematics() -> SymbolicKinematics:
    """
    Return the directional symbolic kinematics restricted to

        fy(z) = fx(z).

    This preserves the exact historical scalar HOST formulation while
    allowing build_symbolic_kinematics() itself to remain directional.
    """

    kin = build_symbolic_kinematics()

    scalar_v = sp.simplify(
        kin.v.subs(
            kin.fy,
            kin.fx,
        )
    )

    scalar_strain = sp.Matrix(
        [
            sp.simplify(
                expression.subs(
                    kin.fy,
                    kin.fx,
                )
            )
            for expression in kin.strain
        ]
    )

    return replace(
        kin,
        fy=kin.fx,
        v=scalar_v,
        strain=scalar_strain,
    )


def build_generalized_strain_system() -> GeneralizedStrainSystem:
    """
    Construct the historical 14-component scalar HOST operator.

    This function intentionally preserves the existing public API.

    Generalized strain ordering:

        0   eps0_xx
        1   eps0_yy
        2   gamma0_xy

        3   kappa_xx
        4   kappa_yy
        5   kappa_xy

        6   chi_xx
        7   chi_yy
        8   chi_xy

        9   beta_x
        10  beta_y

        11  eta_x
        12  eta_y

        13  eta
    """

    kin = _scalarized_kinematics()

    x = kin.x
    y = kin.y
    z = kin.z

    # --------------------------------------------------------------
    # Membrane strains
    # --------------------------------------------------------------

    eps0_xx = sp.diff(
        kin.u0,
        x,
    )

    eps0_yy = sp.diff(
        kin.v0,
        y,
    )

    gamma0_xy = (
        sp.diff(
            kin.u0,
            y,
        )
        + sp.diff(
            kin.v0,
            x,
        )
    )

    # --------------------------------------------------------------
    # Classical bending curvatures
    # --------------------------------------------------------------

    kappa_xx = -sp.diff(
        kin.w0,
        x,
        2,
    )

    kappa_yy = -sp.diff(
        kin.w0,
        y,
        2,
    )

    kappa_xy = -2 * sp.diff(
        kin.w0,
        x,
        y,
    )

    # --------------------------------------------------------------
    # Scalar higher-order warping strains
    # --------------------------------------------------------------

    chi_xx = sp.diff(
        kin.beta_x,
        x,
    )

    chi_yy = sp.diff(
        kin.beta_y,
        y,
    )

    chi_xy = (
        sp.diff(
            kin.beta_x,
            y,
        )
        + sp.diff(
            kin.beta_y,
            x,
        )
    )

    eta_x = sp.diff(
        kin.eta,
        x,
    )

    eta_y = sp.diff(
        kin.eta,
        y,
    )

    E = sp.Matrix(
        [
            eps0_xx,
            eps0_yy,
            gamma0_xy,

            kappa_xx,
            kappa_yy,
            kappa_xy,

            chi_xx,
            chi_yy,
            chi_xy,

            kin.beta_x,
            kin.beta_y,

            eta_x,
            eta_y,

            kin.eta,
        ]
    )

    f = kin.fx
    g = kin.g

    fp = sp.diff(
        f,
        z,
    )

    gp = sp.diff(
        g,
        z,
    )

    F = sp.zeros(
        6,
        14,
    )

    # eps_xx
    F[0, 0] = 1
    F[0, 3] = z
    F[0, 6] = f

    # eps_yy
    F[1, 1] = 1
    F[1, 4] = z
    F[1, 7] = f

    # eps_zz
    F[2, 13] = gp

    # gamma_xy
    F[3, 2] = 1
    F[3, 5] = z
    F[3, 8] = f

    # gamma_xz
    F[4, 9] = fp
    F[4, 11] = g

    # gamma_yz
    F[5, 10] = fp
    F[5, 12] = g

    reconstructed = sp.simplify(
        F * E
    )

    return GeneralizedStrainSystem(
        kin=kin,
        E=E,
        F=F,
        reconstructed_strain=reconstructed,
    )


def build_directional_generalized_strain_system(
) -> DirectionalGeneralizedStrainSystem:
    """
    Construct the 15-component directional HOST strain operator.

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

    kin = build_symbolic_kinematics()

    x = kin.x
    y = kin.y
    z = kin.z

    # --------------------------------------------------------------
    # Membrane strains
    # --------------------------------------------------------------

    eps0_xx = sp.diff(
        kin.u0,
        x,
    )

    eps0_yy = sp.diff(
        kin.v0,
        y,
    )

    gamma0_xy = (
        sp.diff(
            kin.u0,
            y,
        )
        + sp.diff(
            kin.v0,
            x,
        )
    )

    # --------------------------------------------------------------
    # Classical bending curvatures
    # --------------------------------------------------------------

    kappa_xx = -sp.diff(
        kin.w0,
        x,
        2,
    )

    kappa_yy = -sp.diff(
        kin.w0,
        y,
        2,
    )

    kappa_xy = -2 * sp.diff(
        kin.w0,
        x,
        y,
    )

    # --------------------------------------------------------------
    # Directional higher-order warping strains
    # --------------------------------------------------------------

    chi_xx = sp.diff(
        kin.beta_x,
        x,
    )

    chi_yy = sp.diff(
        kin.beta_y,
        y,
    )

    chi_xy_x = sp.diff(
        kin.beta_x,
        y,
    )

    chi_xy_y = sp.diff(
        kin.beta_y,
        x,
    )

    eta_x = sp.diff(
        kin.eta,
        x,
    )

    eta_y = sp.diff(
        kin.eta,
        y,
    )

    E = sp.Matrix(
        [
            eps0_xx,
            eps0_yy,
            gamma0_xy,

            kappa_xx,
            kappa_yy,
            kappa_xy,

            chi_xx,
            chi_yy,
            chi_xy_x,
            chi_xy_y,

            kin.beta_x,
            kin.beta_y,

            eta_x,
            eta_y,

            kin.eta,
        ]
    )

    fx = kin.fx
    fy = kin.fy
    g = kin.g

    dfx = sp.diff(
        fx,
        z,
    )

    dfy = sp.diff(
        fy,
        z,
    )

    dg = sp.diff(
        g,
        z,
    )

    F = sp.zeros(
        6,
        15,
    )

    # eps_xx
    F[0, 0] = 1
    F[0, 3] = z
    F[0, 6] = fx

    # eps_yy
    F[1, 1] = 1
    F[1, 4] = z
    F[1, 7] = fy

    # eps_zz
    F[2, 14] = dg

    # gamma_xy
    F[3, 2] = 1
    F[3, 5] = z

    F[3, 8] = fx
    F[3, 9] = fy

    # gamma_xz
    F[4, 10] = dfx
    F[4, 12] = g

    # gamma_yz
    F[5, 11] = dfy
    F[5, 13] = g

    reconstructed = sp.simplify(
        F * E
    )

    return DirectionalGeneralizedStrainSystem(
        kin=kin,
        E=E,
        F=F,
        reconstructed_strain=reconstructed,
    )