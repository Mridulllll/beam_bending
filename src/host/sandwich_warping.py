"""
Material-informed piecewise warping functions for sandwich plates.

The baseline construction assumes a symmetric three-layer sandwich

    face / core / face

and derives the warping derivative from an assumed continuous
transverse shear-stress distribution

    tau(z) proportional to p(z)

with

    p(z) = 1 - 4*z^2/h^2.

Since

    tau(z) = G(z) * gamma(z)
           = G(z) * f'(z) * beta,

the shear-strain warping derivative becomes

    f'(z) proportional to p(z) / G(z).

Normalizing with respect to the core shear modulus gives

    f'(z) = G_core/G(z) * p(z).

This automatically concentrates shear deformation in a compliant
core while maintaining shear-stress continuity across material
interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from host.generalized_stiffness import (
    WarpingFunctions,
)


@dataclass(frozen=True)
class SymmetricSandwichWarping:
    h: float
    hc: float
    hf: float

    G_face: float
    G_core: float

    f: Callable[[float], float]
    df: Callable[[float], float]


def material_informed_symmetric_warping(
    h: float,
    hc: float,
    G_face: float,
    G_core: float,
) -> SymmetricSandwichWarping:
    """
    Construct a material-informed piecewise sandwich warping function.

    Assumptions
    -----------
    - symmetric sandwich,
    - equal upper/lower faces,
    - midplane at z = 0,
    - scalar transverse shear modulus,
    - continuous target shear-stress profile.
    """

    h = float(h)
    hc = float(hc)

    G_face = float(G_face)
    G_core = float(G_core)

    if h <= 0.0:
        raise ValueError(
            "Total thickness must be positive."
        )

    if hc <= 0.0 or hc >= h:
        raise ValueError(
            "Core thickness must satisfy 0 < hc < h."
        )

    if G_face <= 0.0 or G_core <= 0.0:
        raise ValueError(
            "Shear moduli must be positive."
        )

    hf = 0.5 * (
        h - hc
    )

    z_interface = 0.5 * hc

    def stress_shape(z):
        z = float(z)

        return (
            1.0
            - 4.0 * z**2 / h**2
        )

    def local_G(z):
        z = float(z)

        if abs(z) <= z_interface:
            return G_core

        return G_face

    def df(z):
        z = float(z)

        return (
            G_core
            / local_G(z)
            * stress_shape(z)
        )

    # --------------------------------------------------------------
    # Integrate df analytically piecewise.
    #
    # Define primitive of
    #
    #     1 - 4 z^2 / h^2
    #
    # as
    #
    #     R(z) = z - 4 z^3/(3 h^2).
    #
    # f is chosen odd with f(0)=0.
    # --------------------------------------------------------------

    def R(z):
        z = float(z)

        return (
            z
            - 4.0 * z**3
            / (3.0 * h**2)
        )

    R_interface = R(
        z_interface
    )

    face_ratio = (
        G_core
        / G_face
    )

    def f_positive(z):
        z = float(z)

        if z <= z_interface:
            return R(z)

        return (
            R_interface
            + face_ratio
            * (
                R(z)
                - R_interface
            )
        )

    def f(z):
        z = float(z)

        if z >= 0.0:
            return f_positive(z)

        return -f_positive(-z)

    return SymmetricSandwichWarping(
        h=h,
        hc=hc,
        hf=hf,
        G_face=G_face,
        G_core=G_core,
        f=f,
        df=df,
    )


def material_informed_generalized_warping(
    h: float,
    hc: float,
    G_face: float,
    G_core: float,
) -> WarpingFunctions:
    """
    Adapter to the existing five-field generalized stiffness framework.
    """

    warping = (
        material_informed_symmetric_warping(
            h=h,
            hc=hc,
            G_face=G_face,
            G_core=G_core,
        )
    )

    return WarpingFunctions(
        f=warping.f,
        df=warping.df,
        g=lambda z: 0.0,
        dg=lambda z: 0.0,
    )

# ==================================================================
# Direction-dependent material-informed sandwich warping
# ==================================================================


@dataclass(frozen=True)
class DirectionalSymmetricSandwichWarping:
    """
    Direction-dependent material-informed warping for a symmetric
    three-layer sandwich.

    Separate warping functions are retained for the xz and yz
    transverse-shear planes.
    """

    h: float
    hc: float
    hf: float

    G_face_xz: float
    G_face_yz: float

    G_core_xz: float
    G_core_yz: float

    fx: Callable[[float], float]
    dfx: Callable[[float], float]

    fy: Callable[[float], float]
    dfy: Callable[[float], float]


def material_informed_directional_symmetric_warping(
    h: float,
    hc: float,
    G_face_xz: float,
    G_face_yz: float,
    G_core_xz: float,
    G_core_yz: float,
) -> DirectionalSymmetricSandwichWarping:
    """
    Construct direction-dependent material-informed warping functions.

    Assumptions
    -----------
    - symmetric three-layer sandwich,
    - equal upper/lower faces,
    - midplane at z = 0,
    - piecewise constant transverse shear moduli,
    - continuous target transverse shear-stress profile.

    The directional derivatives satisfy

        dfx(z) =
            G_core_xz / G_xz(z) * p(z)

        dfy(z) =
            G_core_yz / G_yz(z) * p(z)

    with

        p(z) = 1 - 4*z^2/h^2.
    """

    h = float(h)
    hc = float(hc)

    G_face_xz = float(G_face_xz)
    G_face_yz = float(G_face_yz)

    G_core_xz = float(G_core_xz)
    G_core_yz = float(G_core_yz)

    if h <= 0.0:
        raise ValueError(
            "Total thickness must be positive."
        )

    if hc <= 0.0 or hc >= h:
        raise ValueError(
            "Core thickness must satisfy 0 < hc < h."
        )

    for name, value in (
        ("G_face_xz", G_face_xz),
        ("G_face_yz", G_face_yz),
        ("G_core_xz", G_core_xz),
        ("G_core_yz", G_core_yz),
    ):
        if value <= 0.0:
            raise ValueError(
                "{} must be positive.".format(
                    name
                )
            )

    hf = 0.5 * (
        h - hc
    )

    z_interface = 0.5 * hc

    def stress_shape(z):
        z = float(z)

        return (
            1.0
            - 4.0 * z**2 / h**2
        )

    def R(z):
        z = float(z)

        return (
            z
            - 4.0 * z**3
            / (3.0 * h**2)
        )

    R_interface = R(
        z_interface
    )

    face_ratio_x = (
        G_core_xz
        / G_face_xz
    )

    face_ratio_y = (
        G_core_yz
        / G_face_yz
    )

    def dfx(z):
        z = float(z)

        if abs(z) <= z_interface:
            return stress_shape(z)

        return (
            face_ratio_x
            * stress_shape(z)
        )

    def dfy(z):
        z = float(z)

        if abs(z) <= z_interface:
            return stress_shape(z)

        return (
            face_ratio_y
            * stress_shape(z)
        )

    def fx_positive(z):
        z = float(z)

        if z <= z_interface:
            return R(z)

        return (
            R_interface
            + face_ratio_x
            * (
                R(z)
                - R_interface
            )
        )

    def fy_positive(z):
        z = float(z)

        if z <= z_interface:
            return R(z)

        return (
            R_interface
            + face_ratio_y
            * (
                R(z)
                - R_interface
            )
        )

    def fx(z):
        z = float(z)

        if z >= 0.0:
            return fx_positive(z)

        return -fx_positive(-z)

    def fy(z):
        z = float(z)

        if z >= 0.0:
            return fy_positive(z)

        return -fy_positive(-z)

    return DirectionalSymmetricSandwichWarping(
        h=h,
        hc=hc,
        hf=hf,

        G_face_xz=G_face_xz,
        G_face_yz=G_face_yz,

        G_core_xz=G_core_xz,
        G_core_yz=G_core_yz,

        fx=fx,
        dfx=dfx,

        fy=fy,
        dfy=dfy,
    )


def material_informed_directional_generalized_warping(
    h: float,
    hc: float,
    G_face_xz: float,
    G_face_yz: float,
    G_core_xz: float,
    G_core_yz: float,
):
    """
    Adapter to the directional 15-component HOST stiffness framework.
    """

    from host.generalized_stiffness import (
        DirectionalWarpingFunctions,
    )

    warping = (
        material_informed_directional_symmetric_warping(
            h=h,
            hc=hc,

            G_face_xz=G_face_xz,
            G_face_yz=G_face_yz,

            G_core_xz=G_core_xz,
            G_core_yz=G_core_yz,
        )
    )

    return DirectionalWarpingFunctions(
        fx=warping.fx,
        dfx=warping.dfx,

        fy=warping.fy,
        dfy=warping.dfy,

        g=lambda z: 0.0,
        dg=lambda z: 0.0,
    )
