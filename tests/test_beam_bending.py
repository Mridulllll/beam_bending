"""
Verification of axially condensed HOST beam bending equations.
"""

import numpy as np

from host.beam import (
    SandwichBeamGeometry,
    build_three_layer_host_beam,
)

from host.beam_bending import (
    beam_state_space,
    condense_zero_axial_force,
    propagate_uniform_segment,
    recover_axial_strain,
    recover_full_generalized_strain,
    reduced_resultants,
    state_derivative,
    state_generalized_quantities,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)


def make_beam(
    *,
    width=40.0,
    length=240.0,
):
    C = isotropic_stiffness_3d(
        E=70000.0,
        nu=0.33,
    )

    geometry = SandwichBeamGeometry(
        length=length,
        width=width,
        core_height=60.0,
        face_bottom=1.0,
        face_top=1.0,
    )

    return build_three_layer_host_beam(
        C,
        C,
        C,
        geometry,
        quadrature_order=12,
    )


def test_zero_axial_force_condensation_is_symmetric_positive_definite():
    beam = make_beam()

    reduced = condense_zero_axial_force(
        beam.K4
    )

    np.testing.assert_allclose(
        reduced.K3,
        reduced.K3.T,
        rtol=1.0e-13,
        atol=1.0e-8,
    )

    assert np.min(
        np.linalg.eigvalsh(
            reduced.K3
        )
    ) > 0.0


def test_zero_axial_force_recovery_reproduces_N_zero():
    beam = make_beam()

    reduced = condense_zero_axial_force(
        beam.K4
    )

    e3 = np.asarray(
        [
            2.0e-4,
            -3.0e-4,
            1.0e-3,
        ]
    )

    e4 = recover_full_generalized_strain(
        reduced,
        e3,
        axial_force=0.0,
    )

    resultants = (
        beam.K4
        @ e4
    )

    scale = max(
        1.0,
        np.max(
            np.abs(
                resultants[1:]
            )
        ),
    )

    assert abs(
        resultants[0]
    ) <= (
        1.0e-12
        * scale
    )


def test_reduced_resultants_match_full_K4_resultants():
    beam = make_beam()

    reduced = condense_zero_axial_force(
        beam.K4
    )

    e3 = np.asarray(
        [
            1.2e-4,
            -0.7e-4,
            2.5e-3,
        ]
    )

    e4 = recover_full_generalized_strain(
        reduced,
        e3,
    )

    full_resultants = (
        beam.K4
        @ e4
    )

    r3 = reduced_resultants(
        reduced,
        e3,
    )

    np.testing.assert_allclose(
        r3,
        full_resultants[1:4],
        rtol=1.0e-11,
        atol=1.0e-7,
    )


def test_symmetric_homogeneous_beam_has_negligible_axial_coupling():
    beam = make_beam()

    reduced = condense_zero_axial_force(
        beam.K4
    )

    reference = np.max(
        np.abs(
            beam.K4
        )
    )

    assert np.max(
        np.abs(
            reduced.axial_coupling
        )
    ) <= (
        1.0e-14
        * reference
    )


def test_reduced_stiffness_scales_linearly_with_width():
    beam40 = make_beam(
        width=40.0
    )

    beam60 = make_beam(
        width=60.0
    )

    K40 = condense_zero_axial_force(
        beam40.K4
    ).K3

    K60 = condense_zero_axial_force(
        beam60.K4
    ).K3

    np.testing.assert_allclose(
        K60 / 60.0,
        K40 / 40.0,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_state_generalized_quantities_satisfy_constitutive_relation():
    beam = make_beam()

    K3 = condense_zero_axial_force(
        beam.K4
    ).K3

    state = np.asarray(
        [
            0.2,
            -0.01,
            3.0e-4,
            1200.0,
            -20.0,
            450.0,
        ]
    )

    values = state_generalized_quantities(
        K3,
        state,
    )

    strain = np.asarray(
        [
            values["kappa"],
            values["chi"],
            values["beta"],
        ]
    )

    resultants = (
        K3
        @ strain
    )

    np.testing.assert_allclose(
        resultants,
        [
            state[3],
            state[5],
            values["Q"],
        ],
        rtol=1.0e-11,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        values["beta"],
        state[2],
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_state_derivative_satisfies_kinematics_and_equilibrium():
    beam = make_beam()

    K3 = condense_zero_axial_force(
        beam.K4
    ).K3

    state = np.asarray(
        [
            0.5,
            0.02,
            4.0e-4,
            800.0,
            35.0,
            250.0,
        ]
    )

    q = 12.5

    values = state_generalized_quantities(
        K3,
        state,
    )

    derivative = state_derivative(
        K3,
        state,
        q=q,
    )

    np.testing.assert_allclose(
        derivative[0],
        state[1],
    )

    np.testing.assert_allclose(
        derivative[1],
        -values["kappa"],
    )

    np.testing.assert_allclose(
        derivative[2],
        values["chi"],
    )

    np.testing.assert_allclose(
        derivative[3],
        state[4],
    )

    np.testing.assert_allclose(
        derivative[4],
        -q,
    )

    np.testing.assert_allclose(
        derivative[5],
        values["Q"],
    )


def test_state_space_matrix_matches_direct_derivative():
    beam = make_beam()

    K3 = condense_zero_axial_force(
        beam.K4
    ).K3

    state = np.asarray(
        [
            0.1,
            -0.03,
            2.0e-4,
            900.0,
            15.0,
            300.0,
        ]
    )

    q = -8.0

    A, b = beam_state_space(
        K3
    )

    matrix_result = (
        A @ state
        + b * q
    )

    direct_result = state_derivative(
        K3,
        state,
        q=q,
    )

    np.testing.assert_allclose(
        matrix_result,
        direct_result,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_zero_length_propagation_is_identity():
    beam = make_beam()

    K3 = condense_zero_axial_force(
        beam.K4
    ).K3

    state = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    result = propagate_uniform_segment(
        K3,
        state,
        0.0,
        q=10.0,
    )

    np.testing.assert_array_equal(
        result,
        state,
    )


def test_uniform_segment_propagation_has_composition_property():
    beam = make_beam()

    K3 = condense_zero_axial_force(
        beam.K4
    ).K3

    state = np.asarray(
        [
            0.1,
            0.01,
            2.0e-4,
            700.0,
            -10.0,
            180.0,
        ]
    )

    q = 3.5

    direct = propagate_uniform_segment(
        K3,
        state,
        75.0,
        q=q,
    )

    first = propagate_uniform_segment(
        K3,
        state,
        30.0,
        q=q,
    )

    sequential = propagate_uniform_segment(
        K3,
        first,
        45.0,
        q=q,
    )

    np.testing.assert_allclose(
        sequential,
        direct,
        rtol=1.0e-10,
        atol=1.0e-8,
    )


def test_pure_bending_limit_of_diagonal_reduced_stiffness():
    """
    For a synthetic uncoupled section

        M = D*kappa

    with constant M and zero shear quantities,

        w'' = -M/D.

    Therefore

        theta(L) = -M L / D

    and for w(0)=theta(0)=0,

        w(L) = -M L^2 / (2D).
    """

    D = 2.5e9

    K3 = np.diag(
        [
            D,
            8.0e8,
            2.0e7,
        ]
    )

    M = 1200.0
    L = 80.0

    state0 = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            M,
            0.0,
            0.0,
        ]
    )

    stateL = propagate_uniform_segment(
        K3,
        state0,
        L,
        q=0.0,
    )

    expected_theta = (
        -M
        * L
        / D
    )

    expected_w = (
        -M
        * L**2
        / (
            2.0
            * D
        )
    )

    np.testing.assert_allclose(
        stateL[1],
        expected_theta,
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        stateL[0],
        expected_w,
        rtol=1.0e-12,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        stateL[3],
        M,
        rtol=1.0e-13,
    )

    np.testing.assert_allclose(
        stateL[4],
        0.0,
        atol=1.0e-13,
    )