"""
Verification of directional-HSDT single-mode buckling.

The strongest checks enforce exact recovery of the historical scalar
buckling formulation when

    fx = fy = f.
"""

import numpy as np

from host.buckling import (
    critical_directional_load_factor_mode,
    critical_equal_biaxial_load,
    critical_load_factor_mode,
    critical_uniaxial_load_x,
    directional_critical_equal_biaxial_load,
    directional_critical_uniaxial_load_x,
    directional_critical_uniaxial_load_y,
    find_critical_buckling_mode,
)

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
)

from host.generalized_stiffness import (
    Layer,
    assemble_directional_generalized_stiffness,
    assemble_generalized_stiffness,
    scalar_to_directional_warping,
)

from host.hsdt import (
    extract_directional_hsdt_stiffness,
    extract_hsdt_stiffness,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def homogeneous_scalar_and_directional_plate():
    """
    Construct physically identical scalar and directional HSDT
    stiffness matrices.
    """

    E = 70000.0
    nu = 0.33
    h = 1.0

    C = sigma_zz_zero_stiffness(
        isotropic_stiffness_3d(
            E=E,
            nu=nu,
        )
    )

    layers = [
        Layer(
            "plate",
            -0.5 * h,
            +0.5 * h,
            C,
        )
    ]

    scalar_warping = (
        reddy_baseline_generalized_warping(
            h
        )
    )

    directional_warping = (
        scalar_to_directional_warping(
            scalar_warping
        )
    )

    K14 = assemble_generalized_stiffness(
        layers,
        scalar_warping,
        quadrature_order=10,
    )

    K15 = assemble_directional_generalized_stiffness(
        layers,
        directional_warping,
        quadrature_order=10,
    )

    K11 = extract_hsdt_stiffness(
        K14
    )

    K12 = extract_directional_hsdt_stiffness(
        K15
    )

    return K11, K12


def test_directional_buckling_mode_is_positive():
    _, K12 = (
        homogeneous_scalar_and_directional_plate()
    )

    result = critical_directional_load_factor_mode(
        K12,
        a=100.0,
        b=80.0,
        m=1,
        n=1,
        Nx_ref=1.0,
        Ny_ref=0.25,
    )

    assert np.isfinite(
        result.load_factor
    )

    assert result.load_factor > 0.0


def test_scalar_limit_reproduces_each_buckling_mode():
    """
    Strong modal regression:

        lambda_dir(m,n) = lambda_scalar(m,n)

    when fx = fy.
    """

    K11, K12 = (
        homogeneous_scalar_and_directional_plate()
    )

    for m, n in [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    ]:
        scalar = critical_load_factor_mode(
            K11,
            a=100.0,
            b=80.0,
            m=m,
            n=n,
            Nx_ref=1.0,
            Ny_ref=0.4,
        )

        directional = (
            critical_directional_load_factor_mode(
                K12,
                a=100.0,
                b=80.0,
                m=m,
                n=n,
                Nx_ref=1.0,
                Ny_ref=0.4,
            )
        )

        np.testing.assert_allclose(
            directional.load_factor,
            scalar.load_factor,
            rtol=1.0e-11,
            atol=1.0e-10,
        )

        np.testing.assert_allclose(
            directional.elastic_transverse_stiffness,
            scalar.elastic_transverse_stiffness,
            rtol=1.0e-11,
            atol=1.0e-10,
        )

        np.testing.assert_allclose(
            directional.geometric_coefficient,
            scalar.geometric_coefficient,
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_scalar_limit_reproduces_uniaxial_x_buckling():
    K11, K12 = (
        homogeneous_scalar_and_directional_plate()
    )

    scalar = critical_uniaxial_load_x(
        K11,
        a=100.0,
        b=80.0,
        max_m=6,
        max_n=6,
    )

    directional = directional_critical_uniaxial_load_x(
        K12,
        a=100.0,
        b=80.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        directional.load_factor,
        scalar.load_factor,
        rtol=1.0e-11,
    )

    assert directional.m == scalar.m
    assert directional.n == scalar.n


def test_scalar_limit_reproduces_uniaxial_y_buckling():
    K11, K12 = (
        homogeneous_scalar_and_directional_plate()
    )

    scalar = find_critical_buckling_mode(
        K11,
        a=100.0,
        b=80.0,
        Nx_ref=0.0,
        Ny_ref=1.0,
        max_m=6,
        max_n=6,
    )

    directional = directional_critical_uniaxial_load_y(
        K12,
        a=100.0,
        b=80.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        directional.load_factor,
        scalar.load_factor,
        rtol=1.0e-11,
    )

    assert directional.m == scalar.m
    assert directional.n == scalar.n


def test_scalar_limit_reproduces_equal_biaxial_buckling():
    K11, K12 = (
        homogeneous_scalar_and_directional_plate()
    )

    scalar = critical_equal_biaxial_load(
        K11,
        a=100.0,
        b=80.0,
        max_m=6,
        max_n=6,
    )

    directional = directional_critical_equal_biaxial_load(
        K12,
        a=100.0,
        b=80.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        directional.load_factor,
        scalar.load_factor,
        rtol=1.0e-11,
    )

    assert directional.m == scalar.m
    assert directional.n == scalar.n


def test_isotropic_square_plate_has_equal_directional_nx_ny_buckling():
    """
    In the isotropic scalar limit, a square plate must recover the
    expected x/y symmetry.
    """

    _, K12 = (
        homogeneous_scalar_and_directional_plate()
    )

    Nx = directional_critical_uniaxial_load_x(
        K12,
        a=100.0,
        b=100.0,
        max_m=6,
        max_n=6,
    )

    Ny = directional_critical_uniaxial_load_y(
        K12,
        a=100.0,
        b=100.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        Nx.load_factor,
        Ny.load_factor,
        rtol=1.0e-11,
    )