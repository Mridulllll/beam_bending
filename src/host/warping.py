"""
Through-thickness warping functions for the HOST sandwich theory.

The first physical baseline is the Reddy-type cubic HSDT function

    f(z) = z - 4*z^3/(3*h^2)

with

    df/dz = 1 - 4*z^2/h^2.

Therefore

    df/dz = 0

at z = +/- h/2, so transverse shear strains vanish at the
external surfaces in the five-field HSDT baseline.

The quasi-3D transverse-normal function g(z) is deliberately
kept separate from the baseline formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from host.generalized_stiffness import WarpingFunctions


@dataclass(frozen=True)
class HSDTWarping:
    """
    Five-field higher-order shear-deformation warping functions.
    """

    h: float
    f: Callable[[float], float]
    df: Callable[[float], float]


def reddy_cubic_warping(
    h: float,
) -> HSDTWarping:
    """
    Classical cubic Reddy-type warping function.

    Parameters
    ----------
    h
        Total sandwich thickness.

    Returns
    -------
    HSDTWarping
        f(z) and df/dz.
    """

    h = float(h)

    if h <= 0.0:
        raise ValueError(
            "Total sandwich thickness h must be positive."
        )

    def f(z: float) -> float:
        z = float(z)

        return (
            z
            - 4.0 * z**3 / (3.0 * h**2)
        )

    def df(z: float) -> float:
        z = float(z)

        return (
            1.0
            - 4.0 * z**2 / h**2
        )

    return HSDTWarping(
        h=h,
        f=f,
        df=df,
    )


def reddy_baseline_generalized_warping(
    h: float,
) -> WarpingFunctions:
    """
    Adapter for the existing 14-component generalized framework.

    The five-field baseline uses

        eta = 0,

    so g and dg do not contribute physically. They are set to zero
    here in order to retain compatibility with the generalized
    stiffness infrastructure.
    """

    reddy = reddy_cubic_warping(h)

    return WarpingFunctions(
        f=reddy.f,
        df=reddy.df,
        g=lambda z: 0.0,
        dg=lambda z: 0.0,
    )