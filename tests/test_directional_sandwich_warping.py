"""
Tests for direction-dependent material-informed sandwich warping.
"""

import numpy as np

from host.sandwich_warping import (
    material_informed_directional_symmetric_warping,
    material_informed_symmetric_warping,
)


def build_directional():
    return material_informed_directional_symmetric_warping(
        h=11.0,
        hc=10.0,

        G_face_xz=26315.78947368421,
        G_face_yz=26315.78947368421,

        G_core_xz=700.0,
        G_core_yz=1000.0,
    )


def test_directional_warping_is_odd():
    warping = build_directional()

    for z in np.linspace(
        0.0,
        5.5,
        21,
    ):
        np.testing.assert_allclose(
            warping.fx(-z),
            -warping.fx(z),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

        np.testing.assert_allclose(
            warping.fy(-z),
            -warping.fy(z),
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_directional_surface_shear_condition():
    warping = build_directional()

    z_surface = 0.5 * warping.h

    np.testing.assert_allclose(
        warping.dfx(z_surface),
        0.0,
        atol=1.0e-13,
    )

    np.testing.assert_allclose(
        warping.dfy(z_surface),
        0.0,
        atol=1.0e-13,
    )


def test_directional_warping_is_continuous_at_interface():
    warping = build_directional()

    zi = 0.5 * warping.hc

    eps = 1.0e-9

    np.testing.assert_allclose(
        warping.fx(zi - eps),
        warping.fx(zi + eps),
        rtol=1.0e-8,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        warping.fy(zi - eps),
        warping.fy(zi + eps),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_directional_interface_shear_stress_continuity():
    warping = build_directional()

    zi = 0.5 * warping.hc

    eps = 1.0e-9

    tau_x_core = (
        warping.G_core_xz
        * warping.dfx(
            zi - eps
        )
    )

    tau_x_face = (
        warping.G_face_xz
        * warping.dfx(
            zi + eps
        )
    )

    tau_y_core = (
        warping.G_core_yz
        * warping.dfy(
            zi - eps
        )
    )

    tau_y_face = (
        warping.G_face_yz
        * warping.dfy(
            zi + eps
        )
    )

    np.testing.assert_allclose(
        tau_x_core,
        tau_x_face,
        rtol=1.0e-8,
    )

    np.testing.assert_allclose(
        tau_y_core,
        tau_y_face,
        rtol=1.0e-8,
    )


def test_x_and_y_face_shear_strains_follow_directional_modulus_ratios():
    warping = build_directional()

    z_core = 0.0
    z_face = 5.25

    p_core = (
        1.0
        - 4.0
        * z_core**2
        / warping.h**2
    )

    p_face = (
        1.0
        - 4.0
        * z_face**2
        / warping.h**2
    )

    normalized_x = (
        warping.dfx(z_face)
        / p_face
    ) / (
        warping.dfx(z_core)
        / p_core
    )

    normalized_y = (
        warping.dfy(z_face)
        / p_face
    ) / (
        warping.dfy(z_core)
        / p_core
    )

    np.testing.assert_allclose(
        normalized_x,
        warping.G_core_xz
        / warping.G_face_xz,
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        normalized_y,
        warping.G_core_yz
        / warping.G_face_yz,
        rtol=1.0e-12,
    )


def test_isotropic_limit_matches_scalar_material_informed_warping():
    h = 11.0
    hc = 10.0

    G_face = 26315.78947368421
    G_core = 900.0

    scalar = material_informed_symmetric_warping(
        h=h,
        hc=hc,
        G_face=G_face,
        G_core=G_core,
    )

    directional = (
        material_informed_directional_symmetric_warping(
            h=h,
            hc=hc,

            G_face_xz=G_face,
            G_face_yz=G_face,

            G_core_xz=G_core,
            G_core_yz=G_core,
        )
    )

    for z in np.linspace(
        -0.5 * h,
        0.5 * h,
        51,
    ):
        np.testing.assert_allclose(
            directional.fx(z),
            scalar.f(z),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

        np.testing.assert_allclose(
            directional.fy(z),
            scalar.f(z),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

        np.testing.assert_allclose(
            directional.dfx(z),
            scalar.df(z),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

        np.testing.assert_allclose(
            directional.dfy(z),
            scalar.df(z),
            rtol=1.0e-13,
            atol=1.0e-13,
        )