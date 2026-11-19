import numpy as np

from host.ritz_plate import (
    RitzPlateGeometry,
    assemble_directional_five_field_ritz_stiffness,
    orthonormal_legendre_values,
    solve_directional_five_field_ritz_4pb,
)


def test_orthonormal_legendre_constant_mode():
    geometry = RitzPlateGeometry(
        length=10.0,
        width=4.0,
    )

    x = np.linspace(
        0.0,
        geometry.length,
        5,
    )

    values = orthonormal_legendre_values(
        0.0,
        geometry.length,
        3,
        x,
    )

    np.testing.assert_allclose(
        values[0],
        np.full(
            x.shape,
            1.0 / np.sqrt(
                geometry.length
            ),
        ),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_directional_ritz_stiffness_is_symmetric():
    K12 = np.diag(
        np.linspace(
            1.0,
            12.0,
            12,
        )
    )

    geometry = RitzPlateGeometry(
        length=10.0,
        width=4.0,
    )

    stiffness = (
        assemble_directional_five_field_ritz_stiffness(
            K12,
            geometry,
            nx_degree=3,
            ny_degree=2,
        )
    )

    np.testing.assert_allclose(
        stiffness,
        stiffness.T,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_4pb_ritz_equilibrium_and_support_constraints():
    # Synthetic positive directional plate stiffness.  Magnitudes are chosen
    # only to exercise all five fields and all generalized strain sectors.
    K12 = np.diag(
        [
            1.0e5,
            1.0e5,
            5.0e4,
            1.0e7,
            1.0e7,
            5.0e6,
            2.0e6,
            2.0e6,
            1.0e6,
            1.0e6,
            1.0e4,
            1.0e4,
        ]
    )

    solution = (
        solve_directional_five_field_ritz_4pb(
            K12,
            length=20.0,
            width=4.0,
            support_left=2.0,
            support_right=18.0,
            load_left=7.0,
            load_right=13.0,
            total_load=100.0,
            nx_degree=4,
            ny_degree=2,
        )
    )

    np.testing.assert_allclose(
        solution.width_average(
            "w0",
            2.0,
        ),
        0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        solution.width_average(
            "w0",
            18.0,
        ),
        0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        solution.support_reaction_left
        + solution.support_reaction_right,
        100.0,
        rtol=1.0e-8,
        atol=1.0e-8,
    )

    assert (
        solution.width_average(
            "w0",
            10.0,
        )
        > 0.0
    )

    assert (
        solution.projected_residual_relative
        < 1.0e-8
    )

    assert (
        solution.constraint_residual_max
        < 1.0e-10
    )

    np.testing.assert_allclose(
        solution.strain_energy,
        solution.linear_ramp_external_work,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
