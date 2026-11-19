"""
One-dimensional HOST beam specialization.

The beam model is derived from the same 3-D constitutive and
through-thickness HOST framework used by the plate formulation.

Beam kinematics for bending along the longitudinal coordinate s:

    u(s,z) = u0(s) - z*w0'(s) + f_s(z)*beta(s)
    w(s,z) = w0(s)

with

    eps_ss   = eps0 + z*kappa + f_s*chi
    gamma_sz = f_s'*beta

and generalized beam strain vector

    e_b = [eps0, kappa, chi, beta]^T.

Here

    kappa = -w0''
    chi   = beta'

The beam constitutive matrix is obtained directly from the complete
3-D material tensor by imposing zero stress in all inactive
directions. This is important for a beam with free lateral surfaces:
the beam model must not accidentally impose plane strain in the
transverse direction.

HOST permanent Voigt ordering:

    [xx, yy, zz, xy, xz, yz]

For a beam aligned with homogenized x:

    active strains/stresses = [xx, xz]

For a beam aligned with homogenized y:

    active strains/stresses = [yy, yz]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from host.generalized_stiffness import (
    Layer,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


BeamAxis = Literal["x", "y"]


@dataclass(frozen=True)
class SandwichBeamGeometry:
    """
    Geometrical definition of a sandwich beam specimen.

    All dimensions use the same consistent length unit.

    Parameters
    ----------
    length
        Total physical specimen length.

    width
        Beam width.

    core_height
        Sandwich-core height.

    face_bottom
        Bottom face-sheet thickness.

    face_top
        Top face-sheet thickness.

    Notes
    -----
    The specimen length is deliberately distinct from the eventual
    support span and load span used by 3- and 4-point bending tests.
    """

    length: float
    width: float
    core_height: float
    face_bottom: float
    face_top: float

    def __post_init__(self):
        values = {
            "length": self.length,
            "width": self.width,
            "core_height": self.core_height,
            "face_bottom": self.face_bottom,
            "face_top": self.face_top,
        }

        for name, value in values.items():
            if float(value) <= 0.0:
                raise ValueError(
                    "{} must be positive.".format(
                        name
                    )
                )

    @property
    def total_height(self) -> float:
        return (
            float(self.face_bottom)
            + float(self.core_height)
            + float(self.face_top)
        )


@dataclass(frozen=True)
class ThreeLayerHOSTBeam:
    """
    Three-layer HOST beam section.

    K4 ordering:

        [eps0, kappa, chi, beta]

    and corresponding generalized resultants:

        [N, M, H, Q].
    """

    geometry: SandwichBeamGeometry
    axis: BeamAxis
    layers: tuple
    warping: object
    K4: np.ndarray


def _validate_axis(
    axis: str,
) -> BeamAxis:
    axis = str(axis).lower()

    if axis not in (
        "x",
        "y",
    ):
        raise ValueError(
            "Beam axis must be 'x' or 'y'."
        )

    return axis


def beam_active_voigt_indices(
    axis: BeamAxis,
) -> tuple[int, int]:
    """
    Return active normal/shear Voigt indices.

    x beam:
        eps_xx, gamma_xz -> (0, 4)

    y beam:
        eps_yy, gamma_yz -> (1, 5)
    """

    axis = _validate_axis(
        axis
    )

    if axis == "x":
        return (
            0,
            4,
        )

    return (
        1,
        5,
    )


def free_beam_constitutive(
    C_3d: np.ndarray,
    *,
    axis: BeamAxis = "x",
) -> np.ndarray:
    """
    Reduce a complete 6x6 3-D constitutive matrix to the 2x2
    free-surface beam constitutive matrix.

    Active stresses are

        [sigma_ss, tau_sz].

    All remaining stress components are set to zero.

    Partition

        [sigma_a]   [Caa Cai] [eps_a]
        [sigma_i] = [Cia Cii] [eps_i]

    and impose

        sigma_i = 0.

    Then

        eps_i = -Cii^-1 Cia eps_a

    and

        sigma_a =
        (Caa - Cai Cii^-1 Cia) eps_a.

    This is a Schur-complement reduction and is preferable to simply
    extracting [xx,xz] or [yy,yz] from the 3-D tensor.
    """

    C = np.asarray(
        C_3d,
        dtype=float,
    )

    if C.shape != (
        6,
        6,
    ):
        raise ValueError(
            "C_3d must have shape (6, 6)."
        )

    if not np.allclose(
        C,
        C.T,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "C_3d must be symmetric."
        )

    active = beam_active_voigt_indices(
        axis
    )

    inactive = tuple(
        i
        for i in range(6)
        if i not in active
    )

    Caa = C[
        np.ix_(
            active,
            active,
        )
    ]

    Cai = C[
        np.ix_(
            active,
            inactive,
        )
    ]

    Cia = C[
        np.ix_(
            inactive,
            active,
        )
    ]

    Cii = C[
        np.ix_(
            inactive,
            inactive,
        )
    ]

    try:
        correction = (
            Cai
            @ np.linalg.solve(
                Cii,
                Cia,
            )
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Inactive constitutive block is singular; "
            "beam reduction requires the complete 3-D tensor."
        ) from exc

    Cb = (
        Caa
        - correction
    )

    Cb = 0.5 * (
        Cb
        + Cb.T
    )

    eigenvalues = np.linalg.eigvalsh(
        Cb
    )

    if np.min(
        eigenvalues
    ) <= 0.0:
        raise ValueError(
            "Reduced beam constitutive matrix is not "
            "positive definite."
        )

    return Cb


def _beam_warping_values(
    warping,
    z: float,
    *,
    axis: BeamAxis,
) -> tuple[float, float]:
    """
    Evaluate the appropriate longitudinal HOST warping function.

    Directional objects:
        x -> fx, dfx
        y -> fy, dfy

    Historical scalar objects:
        f, df
    """

    axis = _validate_axis(
        axis
    )

    z = float(
        z
    )

    if axis == "x":
        if hasattr(
            warping,
            "fx",
        ) and hasattr(
            warping,
            "dfx",
        ):
            return (
                float(
                    warping.fx(z)
                ),
                float(
                    warping.dfx(z)
                ),
            )

    if axis == "y":
        if hasattr(
            warping,
            "fy",
        ) and hasattr(
            warping,
            "dfy",
        ):
            return (
                float(
                    warping.fy(z)
                ),
                float(
                    warping.dfy(z)
                ),
            )

    if hasattr(
        warping,
        "f",
    ) and hasattr(
        warping,
        "df",
    ):
        return (
            float(
                warping.f(z)
            ),
            float(
                warping.df(z)
            ),
        )

    raise TypeError(
        "Warping object does not provide the required "
        "beam warping functions."
    )


def beam_thickness_operator(
    z: float,
    warping,
    *,
    axis: BeamAxis = "x",
) -> np.ndarray:
    """
    Return the 2x4 HOST beam thickness operator.

    Beam generalized strain ordering:

        0 eps0
        1 kappa
        2 chi
        3 beta

    Physical beam strains:

        eps_ss   = eps0 + z*kappa + f*chi
        gamma_sz = f'*beta
    """

    z = float(
        z
    )

    f, df = _beam_warping_values(
        warping,
        z,
        axis=axis,
    )

    B = np.zeros(
        (
            2,
            4,
        ),
        dtype=float,
    )

    B[0, 0] = 1.0
    B[0, 1] = z
    B[0, 2] = f

    B[1, 3] = df

    return B


def _evaluate_layer_stiffness(
    stiffness,
    z: float,
) -> np.ndarray:
    if callable(
        stiffness
    ):
        C = stiffness(
            float(z)
        )
    else:
        C = stiffness

    C = np.asarray(
        C,
        dtype=float,
    )

    if C.shape != (
        6,
        6,
    ):
        raise ValueError(
            "Beam layers must provide complete 6x6 "
            "3-D constitutive matrices."
        )

    return C


def integrate_beam_layer_stiffness(
    layer: Layer,
    warping,
    *,
    width: float,
    axis: BeamAxis = "x",
    quadrature_order: int = 10,
) -> np.ndarray:
    """
    Integrate one physical layer contribution:

        K_layer =
        b integral B(z)^T C_beam(z) B(z) dz.
    """

    width = float(
        width
    )

    if width <= 0.0:
        raise ValueError(
            "Beam width must be positive."
        )

    if quadrature_order < 1:
        raise ValueError(
            "quadrature_order must be at least 1."
        )

    axis = _validate_axis(
        axis
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
        (
            4,
            4,
        ),
        dtype=float,
    )

    for xi_i, wi_i in zip(
        xi,
        wi,
    ):
        z = (
            center
            + jac
            * float(xi_i)
        )

        C3 = _evaluate_layer_stiffness(
            layer.stiffness,
            z,
        )

        Cb = free_beam_constitutive(
            C3,
            axis=axis,
        )

        B = beam_thickness_operator(
            z,
            warping,
            axis=axis,
        )

        K += (
            width
            * float(wi_i)
            * jac
            * (
                B.T
                @ Cb
                @ B
            )
        )

    return 0.5 * (
        K
        + K.T
    )


def assemble_beam_stiffness(
    layers,
    warping,
    *,
    width: float,
    axis: BeamAxis = "x",
    quadrature_order: int = 10,
) -> np.ndarray:
    """
    Assemble the total 4x4 HOST beam-section stiffness.
    """

    layers = tuple(
        layers
    )

    if len(
        layers
    ) == 0:
        raise ValueError(
            "At least one beam layer is required."
        )

    K = np.zeros(
        (
            4,
            4,
        ),
        dtype=float,
    )

    for layer in layers:
        K += integrate_beam_layer_stiffness(
            layer,
            warping,
            width=width,
            axis=axis,
            quadrature_order=quadrature_order,
        )

    return 0.5 * (
        K
        + K.T
    )


def build_three_layer_host_beam(
    C_bottom,
    C_core,
    C_top,
    geometry: SandwichBeamGeometry,
    *,
    axis: BeamAxis = "x",
    warping=None,
    quadrature_order: int = 10,
) -> ThreeLayerHOSTBeam:
    """
    Construct a three-layer HOST beam.

    Important
    ---------
    C_bottom, C_core and C_top must be complete 6x6 3-D stiffness
    matrices. They are reduced to the appropriate free-surface beam
    constitutive state internally.

    No specimen dimension is hard-coded.
    """

    axis = _validate_axis(
        axis
    )

    h = float(
        geometry.total_height
    )

    z_bottom = (
        -0.5
        * h
    )

    z_interface_1 = (
        z_bottom
        + float(
            geometry.face_bottom
        )
    )

    z_interface_2 = (
        z_interface_1
        + float(
            geometry.core_height
        )
    )

    z_top = (
        +0.5
        * h
    )

    layers = (
        Layer(
            name="bottom_face",
            z_bottom=z_bottom,
            z_top=z_interface_1,
            stiffness=C_bottom,
        ),
        Layer(
            name="core",
            z_bottom=z_interface_1,
            z_top=z_interface_2,
            stiffness=C_core,
        ),
        Layer(
            name="top_face",
            z_bottom=z_interface_2,
            z_top=z_top,
            stiffness=C_top,
        ),
    )

    if warping is None:
        warping = (
            reddy_baseline_generalized_warping(
                h
            )
        )

    K4 = assemble_beam_stiffness(
        layers,
        warping,
        width=geometry.width,
        axis=axis,
        quadrature_order=quadrature_order,
    )

    return ThreeLayerHOSTBeam(
        geometry=geometry,
        axis=axis,
        layers=layers,
        warping=warping,
        K4=K4,
    )