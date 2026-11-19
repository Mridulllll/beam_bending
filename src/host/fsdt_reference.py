"""
Independent FSDT / classical sandwich reference theory.

This module deliberately does not use the five-field HOST warping
kinematics. It provides an independent shear-deformable sandwich
benchmark.

Transverse kinematics:

    w(x,y)
    theta_x(x,y)
    theta_y(x,y)

with engineering transverse shear strains

    gamma_xz = theta_x + w,x
    gamma_yz = theta_y + w,y

and bending curvatures

    kappa_xx = theta_x,x
    kappa_yy = theta_y,y
    kappa_xy = theta_x,y + theta_y,x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from host.generalized_stiffness import (
    Layer,
)


@dataclass(frozen=True)
class FSDTSection:
    D: np.ndarray
    S: np.ndarray


def _evaluate_stiffness(
    stiffness,
    z,
):
    if callable(stiffness):
        C = stiffness(float(z))
    else:
        C = stiffness

    C = np.asarray(
        C,
        dtype=float,
    )

    if C.shape != (6, 6):
        raise ValueError(
            "Layer stiffness must have shape (6, 6)."
        )

    return C


def build_fsdt_section(
    layers,
    *,
    shear_correction: float = 1.0,
    quadrature_order: int = 10,
) -> FSDTSection:
    """
    Integrate independent bending and transverse shear stiffnesses.

    Permanent HOST ordering:

        [xx, yy, zz, xy, xz, yz].

    Bending sector:
        [xx, yy, xy]

    Shear sector:
        [xz, yz]

    For this reference implementation, shear_correction defaults to 1.
    It is exposed explicitly rather than hidden.
    """

    if shear_correction <= 0.0:
        raise ValueError(
            "shear_correction must be positive."
        )

    D = np.zeros(
        (3, 3),
        dtype=float,
    )

    S = np.zeros(
        (2, 2),
        dtype=float,
    )

    inplane = np.asarray(
        [0, 1, 3],
        dtype=int,
    )

    shear = np.asarray(
        [4, 5],
        dtype=int,
    )

    xi, wi = np.polynomial.legendre.leggauss(
        quadrature_order
    )

    for layer in layers:
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

        for xi_i, wi_i in zip(
            xi,
            wi,
        ):
            z = (
                center
                + jac * float(xi_i)
            )

            C = _evaluate_stiffness(
                layer.stiffness,
                z,
            )

            Q = C[
                np.ix_(
                    inplane,
                    inplane,
                )
            ]

            G = C[
                np.ix_(
                    shear,
                    shear,
                )
            ]

            weight = (
                float(wi_i)
                * jac
            )

            D += (
                weight
                * z**2
                * Q
            )

            S += (
                shear_correction
                * weight
                * G
            )

    return FSDTSection(
        D=0.5 * (D + D.T),
        S=0.5 * (S + S.T),
    )


def fsdt_modal_stiffness(
    section: FSDTSection,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Construct the independent 3x3 FSDT Navier stiffness.

    Modal amplitudes:

        q = [W, Theta_x, Theta_y]^T

    with

        w       = W sin(alpha*x) sin(beta*y)

        theta_x = Theta_x cos(alpha*x) sin(beta*y)

        theta_y = Theta_y sin(alpha*x) cos(beta*y).

    Bending normal curvatures belong to the sin-sin family,
    twisting curvature belongs to cos-cos,
    gamma_xz belongs to cos-sin,
    gamma_yz belongs to sin-cos.
    """

    a = float(a)
    b = float(b)

    if a <= 0.0 or b <= 0.0:
        raise ValueError(
            "Plate dimensions must be positive."
        )

    if m < 1 or n < 1:
        raise ValueError(
            "Mode numbers must be positive."
        )

    alpha = (
        float(m)
        * np.pi
        / a
    )

    beta = (
        float(n)
        * np.pi
        / b
    )

    # q = [W, Theta_x, Theta_y]
    G_ss = np.zeros(
        (3, 3),
        dtype=float,
    )

    # kappa_xx
    G_ss[0, 1] = -alpha

    # kappa_yy
    G_ss[1, 2] = -beta

    G_cc = np.zeros(
        (3, 3),
        dtype=float,
    )

    # kappa_xy
    G_cc[2, 1] = beta
    G_cc[2, 2] = alpha

    H_cs = np.zeros(
        (2, 3),
        dtype=float,
    )

    # gamma_xz = theta_x + w,x
    H_cs[0, 0] = alpha
    H_cs[0, 1] = 1.0

    H_sc = np.zeros(
        (2, 3),
        dtype=float,
    )

    # gamma_yz = theta_y + w,y
    H_sc[1, 0] = beta
    H_sc[1, 2] = 1.0

    area_factor = (
        a * b / 4.0
    )

    K = area_factor * (
        G_ss.T
        @ section.D
        @ G_ss

        + G_cc.T
        @ section.D
        @ G_cc

        + H_cs.T
        @ section.S
        @ H_cs

        + H_sc.T
        @ section.S
        @ H_sc
    )

    return 0.5 * (
        K + K.T
    )


def uniform_pressure_modal_load(
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    FSDT modal load vector for uniform pressure.
    """

    f = np.zeros(
        3,
        dtype=float,
    )

    if (
        m % 2 == 0
        or n % 2 == 0
    ):
        return f

    qmn = (
        16.0
        * float(q0)
        / (
            float(m)
            * float(n)
            * np.pi**2
        )
    )

    f[0] = (
        float(a)
        * float(b)
        / 4.0
        * qmn
    )

    return f


def solve_fsdt_mode(
    section: FSDTSection,
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    K = fsdt_modal_stiffness(
        section,
        a,
        b,
        m,
        n,
    )

    f = uniform_pressure_modal_load(
        q0,
        a,
        b,
        m,
        n,
    )

    return np.linalg.solve(
        K,
        f,
    )


def center_deflection_uniform_pressure_fsdt(
    section: FSDTSection,
    q0: float,
    a: float,
    b: float,
    max_mode: int = 21,
) -> float:
    """
    Center deflection from odd Navier modes.
    """

    w = 0.0

    for m in range(
        1,
        max_mode + 1,
        2,
    ):
        for n in range(
            1,
            max_mode + 1,
            2,
        ):
            modal = solve_fsdt_mode(
                section,
                q0,
                a,
                b,
                m,
                n,
            )

            w += (
                modal[0]
                * np.sin(
                    m * np.pi / 2.0
                )
                * np.sin(
                    n * np.pi / 2.0
                )
            )

    return float(w)

def energy_equivalent_shear_stiffness(
    layers,
    *,
    stress_shape=None,
    quadrature_order: int = 20,
):
    """
    Energy-equivalent transverse shear stiffness.

    For each transverse direction, assume

        tau(z) = Q * phi(z) / integral(phi dz)

    and equate

        U_s = 1/2 integral[tau(z)^2 / G(z)] dz

    with

        U_s = Q^2 / (2 S_eff).

    This gives

        S_eff =
        [integral phi(z) dz]^2
        /
        integral [phi(z)^2 / G(z)] dz.

    The result correctly treats compliant layers in series and is
    therefore appropriate as a classical sandwich reference.

    If stress_shape is None,

        phi(z) = 1

    is used.
    """

    if stress_shape is None:
        stress_shape = lambda z: 1.0

    numerator = 0.0

    denominator_x = 0.0
    denominator_y = 0.0

    xi, wi = np.polynomial.legendre.leggauss(
        quadrature_order
    )

    for layer in layers:
        z0 = float(layer.z_bottom)
        z1 = float(layer.z_top)

        jac = 0.5 * (z1 - z0)
        center = 0.5 * (z1 + z0)

        for xi_i, wi_i in zip(
            xi,
            wi,
        ):
            z = (
                center
                + jac * float(xi_i)
            )

            C = _evaluate_stiffness(
                layer.stiffness,
                z,
            )

            phi = float(
                stress_shape(z)
            )

            weight = (
                float(wi_i)
                * jac
            )

            Gxz = float(C[4, 4])
            Gyz = float(C[5, 5])

            if Gxz <= 0.0 or Gyz <= 0.0:
                raise ValueError(
                    "Transverse shear moduli must be positive."
                )

            numerator += (
                weight
                * phi
            )

            denominator_x += (
                weight
                * phi**2
                / Gxz
            )

            denominator_y += (
                weight
                * phi**2
                / Gyz
            )

    Sx = (
        numerator**2
        / denominator_x
    )

    Sy = (
        numerator**2
        / denominator_y
    )

    return np.array(
        [
            [Sx, 0.0],
            [0.0, Sy],
        ],
        dtype=float,
    )


def build_energy_equivalent_fsdt_section(
    layers,
    *,
    stress_shape=None,
    quadrature_order: int = 20,
):
    """
    Classical sandwich FSDT reference using:

        - standard integrated bending stiffness D,
        - energy-equivalent transverse shear stiffness S_eff.

    Unlike the naive integral(G dz), this captures shear localization
    in a compliant sandwich core.
    """

    base = build_fsdt_section(
        layers,
        shear_correction=1.0,
        quadrature_order=quadrature_order,
    )

    S_eff = energy_equivalent_shear_stiffness(
        layers,
        stress_shape=stress_shape,
        quadrature_order=quadrature_order,
    )

    return FSDTSection(
        D=base.D,
        S=S_eff,
    )
