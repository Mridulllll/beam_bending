"""
Tests for proportional multimode buckling interaction curves.
"""

import numpy as np

from host.buckling import (
    critical_equal_biaxial_load,
    critical_uniaxial_load_x,
    find_critical_buckling_mode,
)

from host.buckling_interaction import (
    nx_nxy_interaction,
    nx_ny_interaction,
)

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
)

from host.generalized_stiffness import (
    Layer,
    assemble_generalized_stiffness,
)

from host.hsdt import (
    extract_hsdt_stiffness,
)

from host.multimode_buckling import (
    solve_multimode_buckling,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def homogeneous_plate():
    """
    Verified homogeneous five-field HSDT plate used as an
    interaction-test fixture.
    """

    E = 70000.0
    nu = 0.33
    h = 1.0

    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    C5 = sigma_zz_zero_stiffness(
        C3
    )

    K14 = assemble_generalized_stiffness(
        [
            Layer(
                "plate",
                -0.5 * h,
                +0.5 * h,
                C5,
            )
        ],
        reddy_baseline_generalized_warping(
            h
        ),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(
        K14
    )


def get_point(
    points,
    parameter,
):
    for point in points:
        if np.isclose(
            point.parameter,
            parameter,
        ):
            return point

    raise AssertionError(
        "No interaction point found for parameter={}".format(
            parameter
        )
    )


def critical_uniaxial_load_y(
    K_hsdt,
    a,
    b,
    *,
    max_m=10,
    max_n=10,
):
    return find_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=0.0,
        Ny_ref=1.0,
        max_m=max_m,
        max_n=max_n,
    )


def test_nx_nxy_reference_directions():
    """
    Nx-Nxy parameterization must recover pure negative shear,
    pure Nx compression, and pure positive shear.
    """

    K = homogeneous_plate()

    points = nx_nxy_interaction(
        K,
        a=100.0,
        b=100.0,
        angles_deg=[
            -90.0,
            0.0,
            90.0,
        ],
        max_m=6,
        max_n=6,
    )

    p_minus = get_point(
        points,
        -90.0,
    )

    p_zero = get_point(
        points,
        0.0,
    )

    p_plus = get_point(
        points,
        90.0,
    )

    # Pure negative shear.
    np.testing.assert_allclose(
        p_minus.Nx_ref,
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p_minus.Ny_ref,
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p_minus.Nxy_ref,
        -1.0,
        atol=1.0e-14,
    )

    # Pure Nx compression.
    np.testing.assert_allclose(
        p_zero.Nx_ref,
        1.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p_zero.Ny_ref,
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p_zero.Nxy_ref,
        0.0,
        atol=1.0e-14,
    )

    # Pure positive shear.
    np.testing.assert_allclose(
        p_plus.Nx_ref,
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p_plus.Ny_ref,
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p_plus.Nxy_ref,
        1.0,
        atol=1.0e-14,
    )


def test_nx_ny_reference_directions():
    """
    Nx-Ny parameterization must recover pure Nx, equal biaxial
    direction, and pure Ny.
    """

    K = homogeneous_plate()

    points = nx_ny_interaction(
        K,
        a=100.0,
        b=100.0,
        angles_deg=[
            0.0,
            45.0,
            90.0,
        ],
        max_m=6,
        max_n=6,
    )

    p0 = get_point(
        points,
        0.0,
    )

    p45 = get_point(
        points,
        45.0,
    )

    p90 = get_point(
        points,
        90.0,
    )

    # Pure Nx.
    np.testing.assert_allclose(
        p0.Nx_ref,
        1.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p0.Ny_ref,
        0.0,
        atol=1.0e-14,
    )

    # Equal biaxial loading direction.
    np.testing.assert_allclose(
        p45.Nx_ref,
        1.0 / np.sqrt(2.0),
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        p45.Ny_ref,
        1.0 / np.sqrt(2.0),
        rtol=1.0e-12,
    )

    # Pure Ny.
    np.testing.assert_allclose(
        p90.Nx_ref,
        0.0,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        p90.Ny_ref,
        1.0,
        atol=1.0e-14,
    )


def test_critical_loads_follow_reference_direction():
    """
    Critical resultants must be lambda_cr times the prescribed
    reference load vector.
    """

    K = homogeneous_plate()

    point = nx_nxy_interaction(
        K,
        a=100.0,
        b=100.0,
        angles_deg=[
            30.0,
        ],
        max_m=6,
        max_n=6,
    )[0]

    assert point.load_factor > 0.0

    np.testing.assert_allclose(
        point.Nx_cr,
        point.load_factor
        * point.Nx_ref,
        rtol=1.0e-13,
    )

    np.testing.assert_allclose(
        point.Ny_cr,
        point.load_factor
        * point.Ny_ref,
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        point.Nxy_cr,
        point.load_factor
        * point.Nxy_ref,
        rtol=1.0e-13,
    )


def test_nx_endpoint_matches_existing_uniaxial_solution():
    """
    theta=0 in the Nx-Nxy interaction curve must reproduce the
    already verified pure Nx buckling result.
    """

    K = homogeneous_plate()

    point = nx_nxy_interaction(
        K,
        a=100.0,
        b=100.0,
        angles_deg=[
            0.0,
        ],
        max_m=6,
        max_n=6,
    )[0]

    reference = critical_uniaxial_load_x(
        K,
        a=100.0,
        b=100.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        point.Nx_cr,
        reference.load_factor,
        rtol=1.0e-10,
    )


def test_pure_shear_endpoint_matches_multimode_solution():
    """
    theta=90 degrees in the Nx-Nxy interaction curve must reproduce
    the direct multimode pure-shear solution.
    """

    K = homogeneous_plate()

    point = nx_nxy_interaction(
        K,
        a=100.0,
        b=100.0,
        angles_deg=[
            90.0,
        ],
        max_m=6,
        max_n=6,
    )[0]

    reference = solve_multimode_buckling(
        K,
        a=100.0,
        b=100.0,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        point.Nxy_cr,
        reference.load_factor,
        rtol=1.0e-10,
    )


def test_equal_biaxial_direction_matches_existing_solution():
    """
    theta=45 degrees uses the normalized reference vector

        [1/sqrt(2), 1/sqrt(2), 0].

    The physical equal-biaxial critical resultant must therefore
    reproduce the existing Nx_ref=Ny_ref=1 solution after accounting
    for the reference-vector normalization.
    """

    K = homogeneous_plate()

    point = nx_ny_interaction(
        K,
        a=100.0,
        b=100.0,
        angles_deg=[
            45.0,
        ],
        max_m=6,
        max_n=6,
    )[0]

    reference = critical_equal_biaxial_load(
        K,
        a=100.0,
        b=100.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        point.Nx_cr,
        reference.load_factor,
        rtol=1.0e-10,
    )

    np.testing.assert_allclose(
        point.Ny_cr,
        reference.load_factor,
        rtol=1.0e-10,
    )