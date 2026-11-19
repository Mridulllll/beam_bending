"""
Linear buckling of simply supported HOST sandwich plates.

The current implementation considers uniform normal membrane
prestresses:

    Nx_ref >= 0
    Ny_ref >= 0

where positive values denote compression.

Pure in-plane shear buckling is deliberately excluded here because
Nxy couples different Navier modes and therefore requires a multimode
eigenvalue formulation.

For one (m,n) mode:

    K_elastic q = lambda K_geometric q

and, after static condensation to the transverse modal amplitude W,

    lambda_cr =
        k_W_eff
        /
        [(a*b/4) * (Nx_ref*alpha^2 + Ny_ref*beta^2)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from host.condensation import (
    condense_to_transverse_stiffness,
)

from host.navier import (
    directional_navier_modal_stiffness,
    navier_modal_stiffness_basis_aware,
    navier_mode,
)


@dataclass(frozen=True)
class BucklingModeResult:
    m: int
    n: int

    alpha: float
    beta: float

    elastic_transverse_stiffness: float
    geometric_coefficient: float

    load_factor: float


def geometric_modal_coefficient(
    a: float,
    b: float,
    m: int,
    n: int,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
) -> float:
    """
    Geometric stiffness coefficient associated with W^2.

    Positive Nx_ref and Ny_ref denote compressive membrane resultants
    with units force/length.

    The coefficient has units force/length when W is represented in
    the current modal energy convention.
    """

    Nx_ref = float(Nx_ref)
    Ny_ref = float(Ny_ref)

    if Nx_ref < 0.0 or Ny_ref < 0.0:
        raise ValueError(
            "Current buckling convention requires nonnegative "
            "compressive reference resultants."
        )

    if Nx_ref == 0.0 and Ny_ref == 0.0:
        raise ValueError(
            "At least one compressive reference resultant is required."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    area_factor = (
        float(a)
        * float(b)
        / 4.0
    )

    return area_factor * (
        Nx_ref * mode.alpha**2
        + Ny_ref * mode.beta**2
    )


def critical_load_factor_mode(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
) -> BucklingModeResult:
    """
    Critical multiplier for one simply supported Navier mode.
    """

    Kmn = (
        navier_modal_stiffness_basis_aware(
            K_hsdt,
            a=a,
            b=b,
            m=m,
            n=n,
        )
    )

    k_eff = (
        condense_to_transverse_stiffness(
            Kmn
        )
    )

    k_geo = geometric_modal_coefficient(
        a,
        b,
        m,
        n,
        Nx_ref=Nx_ref,
        Ny_ref=Ny_ref,
    )

    if k_eff <= 0.0:
        raise ValueError(
            "Condensed elastic transverse stiffness must be positive."
        )

    if k_geo <= 0.0:
        raise ValueError(
            "Geometric modal coefficient must be positive."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    return BucklingModeResult(
        m=m,
        n=n,
        alpha=mode.alpha,
        beta=mode.beta,
        elastic_transverse_stiffness=float(
            k_eff
        ),
        geometric_coefficient=float(
            k_geo
        ),
        load_factor=float(
            k_eff / k_geo
        ),
    )


def find_critical_buckling_mode(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Search integer Navier modes for the minimum buckling multiplier.
    """

    if max_m < 1 or max_n < 1:
        raise ValueError(
            "Mode-search limits must be positive."
        )

    results = []

    for m in range(
        1,
        max_m + 1,
    ):
        for n in range(
            1,
            max_n + 1,
        ):
            results.append(
                critical_load_factor_mode(
                    K_hsdt,
                    a,
                    b,
                    m,
                    n,
                    Nx_ref=Nx_ref,
                    Ny_ref=Ny_ref,
                )
            )

    return min(
        results,
        key=lambda result: result.load_factor,
    )


def critical_uniaxial_load_x(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Critical Nx in force/length.

    The reference resultant is chosen as

        Nx_ref = 1
        Ny_ref = 0,

    so the eigenvalue itself equals Nx_cr.
    """

    return find_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=1.0,
        Ny_ref=0.0,
        max_m=max_m,
        max_n=max_n,
    )


def critical_equal_biaxial_load(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Equal biaxial compression:

        Nx = Ny = N_cr.
    """

    return find_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=1.0,
        Ny_ref=1.0,
        max_m=max_m,
        max_n=max_n,
    )

# ==================================================================
# Directional-warping single-mode buckling
# ==================================================================

def critical_directional_load_factor_mode(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
) -> BucklingModeResult:
    """
    Critical buckling multiplier for one directional-HSDT Navier mode.

    The elastic modal stiffness is obtained from the 12-component
    directional five-field theory. The geometric stiffness remains
    governed by the classical membrane-prestress contribution of w0.

    Positive Nx_ref and Ny_ref denote compression.
    """

    Kmn = directional_navier_modal_stiffness(
        K_hsdt,
        a=a,
        b=b,
        m=m,
        n=n,
    )

    k_eff = condense_to_transverse_stiffness(
        Kmn
    )

    k_geo = geometric_modal_coefficient(
        a,
        b,
        m,
        n,
        Nx_ref=Nx_ref,
        Ny_ref=Ny_ref,
    )

    if k_eff <= 0.0:
        raise ValueError(
            "Condensed directional elastic transverse stiffness "
            "must be positive."
        )

    if k_geo <= 0.0:
        raise ValueError(
            "Geometric modal coefficient must be positive."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    return BucklingModeResult(
        m=m,
        n=n,

        alpha=mode.alpha,
        beta=mode.beta,

        elastic_transverse_stiffness=float(
            k_eff
        ),

        geometric_coefficient=float(
            k_geo
        ),

        load_factor=float(
            k_eff / k_geo
        ),
    )


def find_directional_critical_buckling_mode(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Search integer Navier modes for the minimum directional-HSDT
    buckling multiplier.
    """

    if max_m < 1 or max_n < 1:
        raise ValueError(
            "Mode-search limits must be positive."
        )

    results = []

    for m in range(
        1,
        max_m + 1,
    ):
        for n in range(
            1,
            max_n + 1,
        ):
            results.append(
                critical_directional_load_factor_mode(
                    K_hsdt,
                    a,
                    b,
                    m,
                    n,
                    Nx_ref=Nx_ref,
                    Ny_ref=Ny_ref,
                )
            )

    return min(
        results,
        key=lambda result: result.load_factor,
    )


def directional_critical_uniaxial_load_x(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Directional-HSDT critical uniaxial Nx resultant.
    """

    return find_directional_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=1.0,
        Ny_ref=0.0,
        max_m=max_m,
        max_n=max_n,
    )


def directional_critical_uniaxial_load_y(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Directional-HSDT critical uniaxial Ny resultant.
    """

    return find_directional_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=0.0,
        Ny_ref=1.0,
        max_m=max_m,
        max_n=max_n,
    )


def directional_critical_equal_biaxial_load(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    max_m: int = 10,
    max_n: int = 10,
) -> BucklingModeResult:
    """
    Directional-HSDT equal biaxial compression:

        Nx = Ny = N_cr.
    """

    return find_directional_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=1.0,
        Ny_ref=1.0,
        max_m=max_m,
        max_n=max_n,
    )