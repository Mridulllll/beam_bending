"""
Verification tests for the one-dimensional HOST beam specialization.
"""

import numpy as np
import pytest

from host.beam import (
    SandwichBeamGeometry,
    beam_active_voigt_indices,
    beam_thickness_operator,
    build_three_layer_host_beam,
    free_beam_constitutive,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.generalized_stiffness import (
    DirectionalWarpingFunctions,
)


def isotropic_material(
    E=70000.0,
    nu=0.33,
):
    return isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )


def geometry(
    *,
    length=240.0,
    width=40.0,
    core_height=60.0,
    face_bottom=1.0,
    face_top=1.0,
):
    return SandwichBeamGeometry(
        length=length,
        width=width,
        core_height=core_height,
        face_bottom=face_bottom,
        face_top=face_top,
    )


def test_geometry_keeps_length_width_and_height_independent():
    g = geometry(
        length=275.0,
        width=55.0,
        core_height=63.0,
        face_bottom=0.8,
        face_top=1.2,
    )

    assert g.length == 275.0
    assert g.width == 55.0
    assert g.core_height == 63.0

    np.testing.assert_allclose(
        g.total_height,
        65.0,
    )


def test_geometry_rejects_nonpositive_dimensions():
    with pytest.raises(
        ValueError
    ):
        SandwichBeamGeometry(
            length=240.0,
            width=0.0,
            core_height=60.0,
            face_bottom=1.0,
            face_top=1.0,
        )


def test_active_voigt_indices():
    assert beam_active_voigt_indices(
        "x"
    ) == (
        0,
        4,
    )

    assert beam_active_voigt_indices(
        "y"
    ) == (
        1,
        5,
    )


def test_isotropic_free_beam_reduction_recovers_E_and_G():
    E = 70000.0
    nu = 0.33

    C3 = isotropic_material(
        E=E,
        nu=nu,
    )

    G = (
        E
        / (
            2.0
            * (1.0 + nu)
        )
    )

    expected = np.array(
        [
            [
                E,
                0.0,
            ],
            [
                0.0,
                G,
            ],
        ],
        dtype=float,
    )

    for axis in (
        "x",
        "y",
    ):
        Cb = free_beam_constitutive(
            C3,
            axis=axis,
        )

        np.testing.assert_allclose(
            Cb,
            expected,
            rtol=1.0e-12,
            atol=1.0e-9,
        )


def test_homogeneous_layered_beam_recovers_EA_and_EI():
    E = 70000.0
    nu = 0.33

    C3 = isotropic_material(
        E=E,
        nu=nu,
    )

    g = geometry(
        length=240.0,
        width=40.0,
        core_height=60.0,
        face_bottom=1.0,
        face_top=1.0,
    )

    beam = build_three_layer_host_beam(
        C_bottom=C3,
        C_core=C3,
        C_top=C3,
        geometry=g,
        axis="x",
        quadrature_order=12,
    )

    b = g.width
    h = g.total_height

    EA = (
        E
        * b
        * h
    )

    EI = (
        E
        * b
        * h**3
        / 12.0
    )

    np.testing.assert_allclose(
        beam.K4[0, 0],
        EA,
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        beam.K4[1, 1],
        EI,
        rtol=1.0e-12,
    )


def test_beam_section_stiffness_scales_linearly_with_width():
    """
    Beam section stiffness must scale linearly with width.

    Compare stiffness per unit width rather than the raw matrices.
    Some theoretically zero extension/bending coupling terms contain
    quadrature round-off of order 1e-7 in the full-width matrices.
    Relative-error tests are not meaningful for those near-zero terms.
    """

    C3 = isotropic_material()

    beam40 = build_three_layer_host_beam(
        C3,
        C3,
        C3,
        geometry(
            width=40.0
        ),
    )

    beam60 = build_three_layer_host_beam(
        C3,
        C3,
        C3,
        geometry(
            width=60.0
        ),
    )

    K40_per_width = (
        beam40.K4
        / beam40.geometry.width
    )

    K60_per_width = (
        beam60.K4
        / beam60.geometry.width
    )

    np.testing.assert_allclose(
        K60_per_width,
        K40_per_width,
        rtol=1.0e-12,
        atol=1.0e-8,
    )

def test_section_stiffness_is_independent_of_specimen_length():
    """
    Length belongs to the beam boundary-value problem, not to the
    cross-sectional constitutive integration.
    """

    C3 = isotropic_material()

    beam240 = build_three_layer_host_beam(
        C3,
        C3,
        C3,
        geometry(
            length=240.0
        ),
    )

    beam400 = build_three_layer_host_beam(
        C3,
        C3,
        C3,
        geometry(
            length=400.0
        ),
    )

    np.testing.assert_allclose(
        beam240.K4,
        beam400.K4,
        rtol=1.0e-13,
        atol=1.0e-9,
    )


def test_core_height_is_a_free_section_parameter():
    E = 70000.0

    C3 = isotropic_material(
        E=E
    )

    for hc in (
        20.0,
        40.0,
        60.0,
        80.0,
    ):
        g = geometry(
            core_height=hc,
        )

        beam = build_three_layer_host_beam(
            C3,
            C3,
            C3,
            g,
            quadrature_order=12,
        )

        expected_EI = (
            E
            * g.width
            * g.total_height**3
            / 12.0
        )

        np.testing.assert_allclose(
            beam.K4[1, 1],
            expected_EI,
            rtol=1.0e-12,
        )


def test_directional_beam_uses_fx_for_x_and_fy_for_y():
    """
    Artificial directional warping is used here so the axis-selection
    behavior can be checked independently of material symmetry.
    """

    warping = DirectionalWarpingFunctions(
        fx=lambda z: z,
        dfx=lambda z: 1.0,

        fy=lambda z: 2.0 * z,
        dfy=lambda z: 2.0,

        g=lambda z: 0.0,
        dg=lambda z: 0.0,
    )

    Bx = beam_thickness_operator(
        0.25,
        warping,
        axis="x",
    )

    By = beam_thickness_operator(
        0.25,
        warping,
        axis="y",
    )

    np.testing.assert_allclose(
        Bx[0, 2],
        0.25,
    )

    np.testing.assert_allclose(
        By[0, 2],
        0.50,
    )

    np.testing.assert_allclose(
        Bx[1, 3],
        1.0,
    )

    np.testing.assert_allclose(
        By[1, 3],
        2.0,
    )


def test_directional_axis_changes_only_directional_warping_for_isotropic_C():
    C3 = isotropic_material()

    warping = DirectionalWarpingFunctions(
        fx=lambda z: z,
        dfx=lambda z: 1.0,

        fy=lambda z: 2.0 * z,
        dfy=lambda z: 2.0,

        g=lambda z: 0.0,
        dg=lambda z: 0.0,
    )

    g = geometry(
        core_height=20.0,
    )

    beam_x = build_three_layer_host_beam(
        C3,
        C3,
        C3,
        g,
        axis="x",
        warping=warping,
    )

    beam_y = build_three_layer_host_beam(
        C3,
        C3,
        C3,
        g,
        axis="y",
        warping=warping,
    )

    # Classical extension and bending are unchanged.
    np.testing.assert_allclose(
        beam_y.K4[0, 0],
        beam_x.K4[0, 0],
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        beam_y.K4[1, 1],
        beam_x.K4[1, 1],
        rtol=1.0e-12,
    )

    # Warping and shear terms scale with the imposed factor two.
    np.testing.assert_allclose(
        beam_y.K4[2, 2],
        4.0 * beam_x.K4[2, 2],
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        beam_y.K4[3, 3],
        4.0 * beam_x.K4[3, 3],
        rtol=1.0e-12,
    )