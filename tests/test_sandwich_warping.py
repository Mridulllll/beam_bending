import numpy as np

from host.sandwich_warping import (
    material_informed_symmetric_warping,
)


def reference_warping():
    h = 11.0e-3
    hc = 10.0e-3

    G_face = 70.0e9 / (
        2.0 * (1.0 + 0.33)
    )

    G_core = 0.7e9 / (
        2.0 * (1.0 + 0.25)
    )

    return material_informed_symmetric_warping(
        h=h,
        hc=hc,
        G_face=G_face,
        G_core=G_core,
    )


def test_material_informed_warping_is_odd():
    warping = reference_warping()

    z_values = np.linspace(
        -0.5 * warping.h,
        0.5 * warping.h,
        21,
    )

    for z in z_values:
        np.testing.assert_allclose(
            warping.f(-z),
            -warping.f(z),
            rtol=1.0e-12,
            atol=1.0e-15,
        )


def test_surface_shear_condition():
    warping = reference_warping()

    np.testing.assert_allclose(
        warping.df(
            -0.5 * warping.h
        ),
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        warping.df(
            +0.5 * warping.h
        ),
        0.0,
        atol=1.0e-14,
    )


def test_warping_is_continuous_at_interfaces():
    warping = reference_warping()

    zi = 0.5 * warping.hc

    eps = 1.0e-10

    for sign in (-1.0, +1.0):
        z_interface = sign * zi

        f_left = warping.f(
            z_interface - eps
        )

        f_right = warping.f(
            z_interface + eps
        )

        scale = max(
            1.0e-12,
            abs(
                warping.f(
                    z_interface
                )
            ),
        )

        assert (
            abs(
                f_left - f_right
            )
            / scale
            < 1.0e-5
        )


def test_interface_shear_stress_continuity():
    """
    Because

        tau_xz = G(z) * f'(z) * beta_x,

    the quantity

        G(z) * f'(z)

    must be continuous at the face/core interface.
    """

    warping = reference_warping()

    zi = 0.5 * warping.hc

    h = warping.h

    stress_shape = (
        1.0
        - 4.0 * zi**2 / h**2
    )

    df_core = stress_shape

    df_face = (
        warping.G_core
        / warping.G_face
        * stress_shape
    )

    tau_core_factor = (
        warping.G_core
        * df_core
    )

    tau_face_factor = (
        warping.G_face
        * df_face
    )

    np.testing.assert_allclose(
        tau_core_factor,
        tau_face_factor,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_face_shear_strain_is_smaller_than_core_shear_strain():
    warping = reference_warping()

    zi = 0.5 * warping.hc

    eps = 1.0e-9

    df_core = warping.df(
        zi - eps
    )

    df_face = warping.df(
        zi + eps
    )

    assert abs(df_face) < abs(df_core)


def test_face_to_core_shear_strain_ratio_matches_modulus_ratio():
    """
    At the same target shear stress,

        gamma_face/gamma_core = G_core/G_face.
    """

    warping = reference_warping()

    zi = 0.5 * warping.hc

    eps = 1.0e-9

    ratio_numeric = (
        warping.df(
            zi + eps
        )
        / warping.df(
            zi - eps
        )
    )

    ratio_expected = (
        warping.G_core
        / warping.G_face
    )

    np.testing.assert_allclose(
        ratio_numeric,
        ratio_expected,
        rtol=1.0e-5,
        atol=1.0e-8,
    )