import numpy as np

from host.warping import (
    reddy_cubic_warping,
    reddy_baseline_generalized_warping,
)


def test_reddy_warping_is_odd():
    h = 0.011

    warping = reddy_cubic_warping(h)

    z_values = np.linspace(
        -0.5 * h,
        0.5 * h,
        11,
    )

    for z in z_values:
        np.testing.assert_allclose(
            warping.f(-z),
            -warping.f(z),
            rtol=1.0e-13,
            atol=1.0e-15,
        )


def test_reddy_derivative_is_even():
    h = 0.011

    warping = reddy_cubic_warping(h)

    z_values = np.linspace(
        -0.5 * h,
        0.5 * h,
        11,
    )

    for z in z_values:
        np.testing.assert_allclose(
            warping.df(-z),
            warping.df(z),
            rtol=1.0e-13,
            atol=1.0e-15,
        )


def test_reddy_surface_shear_condition():
    """
    df/dz must vanish exactly at the external surfaces.
    """

    h = 0.011

    warping = reddy_cubic_warping(h)

    np.testing.assert_allclose(
        warping.df(-0.5 * h),
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        warping.df(+0.5 * h),
        0.0,
        atol=1.0e-14,
    )


def test_reddy_midplane_derivative():
    h = 0.011

    warping = reddy_cubic_warping(h)

    np.testing.assert_allclose(
        warping.df(0.0),
        1.0,
        atol=1.0e-14,
    )


def test_baseline_disables_quasi3d_terms():
    h = 0.011

    warping = reddy_baseline_generalized_warping(h)

    z_values = np.linspace(
        -0.5 * h,
        0.5 * h,
        7,
    )

    for z in z_values:
        assert warping.g(z) == 0.0
        assert warping.dg(z) == 0.0