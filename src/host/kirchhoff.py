"""
Classical Kirchhoff-Love reference solutions.

These routines are intentionally independent of the HOST/HSDT
implementation and are used as analytical verification benchmarks.
"""

from __future__ import annotations

import numpy as np


def isotropic_bending_rigidity(
    E: float,
    nu: float,
    h: float,
) -> float:
    """
    Classical isotropic thin-plate bending rigidity

        D = E h^3 / [12 (1 - nu^2)].
    """

    E = float(E)
    nu = float(nu)
    h = float(h)

    if E <= 0.0:
        raise ValueError(
            "Young's modulus must be positive."
        )

    if h <= 0.0:
        raise ValueError(
            "Plate thickness must be positive."
        )

    if not (-1.0 < nu < 0.5):
        raise ValueError(
            "Poisson ratio must satisfy -1 < nu < 0.5."
        )

    return (
        E
        * h**3
        / (
            12.0
            * (1.0 - nu**2)
        )
    )


def uniform_pressure_coefficient(
    q0: float,
    m: int,
    n: int,
) -> float:
    """
    Double-sine Fourier coefficient for uniform pressure.
    """

    if m < 1 or n < 1:
        raise ValueError(
            "Mode numbers must be positive."
        )

    if m % 2 == 0 or n % 2 == 0:
        return 0.0

    return (
        16.0
        * float(q0)
        / (
            np.pi**2
            * float(m)
            * float(n)
        )
    )


def modal_amplitude(
    q0: float,
    D: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> float:
    """
    Kirchhoff-Love modal deflection amplitude W_mn.

    Governing modal equation:

        D (alpha^2 + beta^2)^2 W_mn = q_mn.
    """

    alpha = (
        float(m)
        * np.pi
        / float(a)
    )

    beta = (
        float(n)
        * np.pi
        / float(b)
    )

    qmn = uniform_pressure_coefficient(
        q0,
        m,
        n,
    )

    return (
        qmn
        / (
            float(D)
            * (
                alpha**2
                + beta**2
            )**2
        )
    )


def center_deflection_uniform_pressure(
    q0: float,
    E: float,
    nu: float,
    h: float,
    a: float,
    b: float,
    max_mode: int = 31,
) -> float:
    """
    Center deflection of a simply supported rectangular isotropic
    Kirchhoff-Love plate under uniform pressure.

    Odd modes are summed through max_mode.
    """

    D = isotropic_bending_rigidity(
        E,
        nu,
        h,
    )

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
            Wmn = modal_amplitude(
                q0,
                D,
                a,
                b,
                m,
                n,
            )

            sign_x = (
                np.sin(
                    m * np.pi / 2.0
                )
            )

            sign_y = (
                np.sin(
                    n * np.pi / 2.0
                )
            )

            w += (
                Wmn
                * sign_x
                * sign_y
            )

    return float(w)