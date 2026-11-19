"""
Three-layer sandwich construction for the five-field HOST HSDT.

The baseline sandwich consists of

    bottom face / homogenized core / top face.

For the five-field formulation, sigma_zz = 0 is imposed while the
transverse shear response is retained.

A later quasi-3D extension can retain transverse-normal deformation
and the complete 3-D constitutive response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from host.constitutive import (
    sigma_zz_zero_stiffness,
)

from host.generalized_stiffness import (
    Layer,
    assemble_generalized_stiffness,
)

from host.hsdt import (
    extract_hsdt_stiffness,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


@dataclass(frozen=True)
class ThreeLayerSandwich:
    hf_bottom: float
    hc: float
    hf_top: float
    h_total: float

    layers: tuple

    K14: np.ndarray
    K11: np.ndarray


def build_three_layer_sandwich(
    C_bottom: np.ndarray,
    C_core: np.ndarray,
    C_top: np.ndarray,
    hf_bottom: float,
    hc: float,
    hf_top: float,
    *,
    quadrature_order: int = 10,
    warping=None,
) -> ThreeLayerSandwich:
    """
    Construct a three-layer five-field HSDT sandwich.

    Parameters
    ----------
    C_bottom, C_core, C_top
        Full 6x6 constitutive matrices in permanent HOST ordering:

            [xx, yy, zz, xy, xz, yz].

        The matrices may later come directly from RVE homogenization.

    hf_bottom
        Bottom face-sheet thickness.

    hc
        Core thickness.

    hf_top
        Top face-sheet thickness.
    """

    hf_bottom = float(hf_bottom)
    hc = float(hc)
    hf_top = float(hf_top)

    if hf_bottom <= 0.0:
        raise ValueError(
            "Bottom face thickness must be positive."
        )

    if hc <= 0.0:
        raise ValueError(
            "Core thickness must be positive."
        )

    if hf_top <= 0.0:
        raise ValueError(
            "Top face thickness must be positive."
        )

    h_total = (
        hf_bottom
        + hc
        + hf_top
    )

    z_bottom = -0.5 * h_total

    z_interface_1 = (
        z_bottom
        + hf_bottom
    )

    z_interface_2 = (
        z_interface_1
        + hc
    )

    z_top = +0.5 * h_total

    Cb = sigma_zz_zero_stiffness(
        C_bottom
    )

    Cc = sigma_zz_zero_stiffness(
        C_core
    )

    Ct = sigma_zz_zero_stiffness(
        C_top
    )

    layers = (
        Layer(
            name="bottom_face",
            z_bottom=z_bottom,
            z_top=z_interface_1,
            stiffness=Cb,
        ),
        Layer(
            name="core",
            z_bottom=z_interface_1,
            z_top=z_interface_2,
            stiffness=Cc,
        ),
        Layer(
            name="top_face",
            z_bottom=z_interface_2,
            z_top=z_top,
            stiffness=Ct,
        ),
    )

    if warping is None:
        warping = (
            reddy_baseline_generalized_warping(
                h_total
            )
        )

    K14 = assemble_generalized_stiffness(
        layers,
        warping,
        quadrature_order=quadrature_order,
    )

    K11 = extract_hsdt_stiffness(
        K14
    )

    return ThreeLayerSandwich(
        hf_bottom=hf_bottom,
        hc=hc,
        hf_top=hf_top,
        h_total=h_total,
        layers=layers,
        K14=K14,
        K11=K11,
    )