"""
Verification of the exact HOST point-load beam solver.
"""

import numpy as np

from host.beam_point_loads import (
    PointForce,
    apply_point_force_jump,
    simply_supported_reactions,
    solve_simply_supported_point_loads,
    symmetric_four_point_bending_loads,
    symmetric_support_positions,
    three_point_bending_loads,
)


def synthetic_eb_K3(
    EI=2.5e9,
):
    """
    Synthetic uncoupled section.

    The beta/H/Q subsystem is independent of bending, so the w/M/V
    response reduces exactly to Euler-Bernoulli beam bending.
    """

    return np.diag(
        [
            EI,
            8.0e8,
            2.0e7,
        ]
    )


def test_point_force_jump_changes_only_V():
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

    result = apply_point_force_jump(
        state,
        7.5,
    )

    expected = state.copy()
    expected[4] -= 7.5

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_symmetric_three_point_reactions_are_equal():
    L = 240.0

    support_left = 20.0
    support_right = 220.0

    P = 1000.0

    loads = three_point_bending_loads(
        load=P,
        load_position=0.5 * L,
    )

    reactions = simply_supported_reactions(
        loads,
        support_left=support_left,
        support_right=support_right,
    )

    np.testing.assert_allclose(
        reactions.left,
        -0.5 * P,
    )

    np.testing.assert_allclose(
        reactions.right,
        -0.5 * P,
    )


def test_symmetric_four_point_reactions_are_equal():
    L = 240.0

    support_left, support_right = (
        symmetric_support_positions(
            specimen_length=L,
            support_span=200.0,
        )
    )

    P = 1200.0

    loads = symmetric_four_point_bending_loads(
        total_load=P,
        specimen_length=L,
        load_span=80.0,
    )

    reactions = simply_supported_reactions(
        loads,
        support_left=support_left,
        support_right=support_right,
    )

    np.testing.assert_allclose(
        reactions.left,
        -0.5 * P,
    )

    np.testing.assert_allclose(
        reactions.right,
        -0.5 * P,
    )


def test_reactions_satisfy_force_and_moment_equilibrium():
    loads = (
        PointForce(
            75.0,
            300.0,
        ),
        PointForce(
            140.0,
            500.0,
        ),
        PointForce(
            195.0,
            200.0,
        ),
    )

    xA = 20.0
    xB = 220.0

    reactions = simply_supported_reactions(
        loads,
        support_left=xA,
        support_right=xB,
    )

    force_balance = (
        reactions.left
        + reactions.right
        + sum(
            load.force
            for load in loads
        )
    )

    moment_balance = (
        reactions.right
        * (
            xB - xA
        )
        + sum(
            load.force
            * (
                load.x - xA
            )
            for load in loads
        )
    )

    np.testing.assert_allclose(
        force_balance,
        0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        moment_balance,
        0.0,
        atol=1.0e-10,
    )


def test_three_point_solver_satisfies_support_and_free_end_conditions():
    K3 = synthetic_eb_K3()

    L = 240.0
    xA = 20.0
    xB = 220.0

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=three_point_bending_loads(
            load=1000.0,
            load_position=120.0,
        ),
    )

    np.testing.assert_allclose(
        solution.state(
            xA
        )[0],
        0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        solution.state(
            xB
        )[0],
        0.0,
        atol=1.0e-10,
    )

    left = solution.state(
        0.0,
        side="left",
    )

    right = solution.state(
        L,
        side="right",
    )

    np.testing.assert_allclose(
        left[
            3:6
        ],
        0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        right[
            3:6
        ],
        0.0,
        atol=1.0e-8,
    )


def test_euler_bernoulli_three_point_midspan_deflection():
    """
    Simply supported beam with supports at the physical ends and a
    central concentrated load.

    Classical magnitude:

        |w_mid| = P S^3 / (48 EI)

    where S is the support span.
    """

    EI = 3.0e9
    K3 = synthetic_eb_K3(
        EI=EI
    )

    L = 200.0
    P = 800.0

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=0.0,
        support_right=L,
        applied_forces=three_point_bending_loads(
            load=P,
            load_position=0.5 * L,
        ),
    )

    w_mid = solution.state(
        0.5 * L,
        side="right",
    )[0]

    expected_magnitude = (
        P
        * L**3
        / (
            48.0
            * EI
        )
    )

    np.testing.assert_allclose(
        abs(
            w_mid
        ),
        expected_magnitude,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_three_point_solution_is_symmetric():
    K3 = synthetic_eb_K3()

    L = 240.0

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=20.0,
        support_right=220.0,
        applied_forces=three_point_bending_loads(
            load=1000.0,
            load_position=120.0,
        ),
    )

    for x in (
        30.0,
        60.0,
        90.0,
        110.0,
    ):
        left = solution.state(
            x
        )

        right = solution.state(
            L - x
        )

        np.testing.assert_allclose(
            left[0],
            right[0],
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_four_point_constant_moment_region_has_constant_M():
    K3 = synthetic_eb_K3()

    L = 240.0

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=200.0,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=1200.0,
        specimen_length=L,
        load_span=80.0,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    x1 = loads[0].x
    x2 = loads[1].x

    samples = np.linspace(
        x1 + 1.0,
        x2 - 1.0,
        5,
    )

    moments = np.asarray(
        [
            solution.state(
                float(x)
            )[3]
            for x in samples
        ]
    )

    np.testing.assert_allclose(
        moments,
        moments[0],
        rtol=1.0e-11,
        atol=1.0e-8,
    )


def test_four_point_constant_moment_region_has_zero_V():
    K3 = synthetic_eb_K3()

    L = 240.0

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=200.0,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=1200.0,
        specimen_length=L,
        load_span=80.0,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    x_mid = 0.5 * (
        loads[0].x
        + loads[1].x
    )

    np.testing.assert_allclose(
        solution.state(
            x_mid
        )[4],
        0.0,
        atol=1.0e-10,
    )


def test_V_jump_matches_each_applied_and_reaction_force():
    K3 = synthetic_eb_K3()

    L = 240.0

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=200.0,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=1200.0,
        specimen_length=L,
        load_span=80.0,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    for load in solution.all_point_forces():
        left = solution.state(
            load.x,
            side="left",
        )

        right = solution.state(
            load.x,
            side="right",
        )

        np.testing.assert_allclose(
            right[4] - left[4],
            -load.force,
            rtol=1.0e-12,
            atol=1.0e-10,
        )


def test_deep_sandwich_multiple_shooting_remains_stable():
    """
    Regression for the numerical instability that occurs when a deep
    sandwich beam is solved by propagating one initial state over the
    complete specimen.

    This K3 is representative of the 40 mm wide, 62 mm deep Gyroid
    sandwich benchmark. Its higher-order state equations contain a
    strongly growing/decaying boundary-layer pair.

    Multiple shooting must satisfy the support/free-end conditions
    without relying on catastrophic long-distance cancellation.
    """

    K3 = np.asarray(
        [
            [
                6.398193758705e9,
                4.493807442472e9,
                9.110959940136e-13,
            ],
            [
                4.493807442472e9,
                3.182153909040e9,
                3.419234044758e-12,
            ],
            [
                9.110959940136e-13,
                3.419234044758e-12,
                9.648618871764e5,
            ],
        ],
        dtype=float,
    )

    L = 240.0

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=200.0,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=1000.0,
        specimen_length=L,
        load_span=80.0,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    left_support = solution.state(
        xA
    )

    right_support = solution.state(
        xB
    )

    right_end = solution.state(
        L,
        side="right",
    )

    midspan = solution.state(
        0.5 * L
    )

    np.testing.assert_allclose(
        left_support[0],
        0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        right_support[0],
        0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        right_end[3],
        0.0,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        right_end[4],
        0.0,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        right_end[5],
        0.0,
        atol=1.0e-7,
    )

    # Constant-moment four-point region.
    np.testing.assert_allclose(
        midspan[3],
        30000.0,
        rtol=1.0e-11,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        midspan[4],
        0.0,
        atol=1.0e-8,
    )

    assert np.isfinite(
        midspan[0]
    )

    assert midspan[0] > 0.0



def _high_growth_HOST_section():
    """
    Synthetic positive-definite HOST section with a short
    higher-order boundary-layer length.

    The strong M-H coupling produces a real state eigenvalue of
    approximately 0.8 / mm, which is deliberately much more severe
    than the earlier deep-sandwich regression.
    """

    D = 1.0e9
    H = 1.0e8
    rho = 0.992

    coupling = (
        rho
        * np.sqrt(
            D * H
        )
    )

    return np.asarray(
        [
            [
                D,
                coupling,
                0.0,
            ],
            [
                coupling,
                H,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0e6,
            ],
        ],
        dtype=float,
    )


def test_adaptive_multiple_shooting_inserts_auxiliary_knots():
    """
    A high-growth HOST section must be subdivided more finely than the
    six physical knots of the standard four-point fixture.
    """

    K3 = _high_growth_HOST_section()

    L = 240.0

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=200.0,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=1000.0,
        specimen_length=L,
        load_span=80.0,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    # Physical fixture knots:
    #
    # 0, 20, 80, 160, 220, 240
    #
    # Adaptive multiple shooting must insert additional ones.

    assert len(
        solution.knot_positions
    ) > 6

    np.testing.assert_allclose(
        solution.state(
            xA
        )[0],
        0.0,
        atol=1.0e-9,
    )

    np.testing.assert_allclose(
        solution.state(
            xB
        )[0],
        0.0,
        atol=1.0e-9,
    )

    np.testing.assert_allclose(
        solution.state(
            L
        )[3:6],
        0.0,
        atol=1.0e-6,
    )


def test_high_growth_solution_preserves_exact_width_scaling():
    """
    Scaling the complete section stiffness by a factor s is equivalent
    to changing beam width by the same factor.

    For fixed applied forces,

        w(s K3) = w(K3) / s.

    This is an important regression because loss of width scaling was
    the diagnostic that exposed insufficient shooting refinement in
    the TPMS design sweep.
    """

    K3 = _high_growth_HOST_section()

    L = 240.0

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=200.0,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=1000.0,
        specimen_length=L,
        load_span=80.0,
    )

    solution_40 = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    scale = 1.5

    solution_60 = solve_simply_supported_point_loads(
        scale * K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    x_mid = 0.5 * L

    w40 = solution_40.state(
        x_mid
    )[0]

    w60 = solution_60.state(
        x_mid
    )[0]

    np.testing.assert_allclose(
        scale * w60,
        w40,
        rtol=1.0e-7,
        atol=1.0e-9,
    )

    # The generalized-resultant field is governed by static
    # equilibrium and therefore remains unchanged by width.

    state40 = solution_40.state(
        x_mid
    )

    state60 = solution_60.state(
        x_mid
    )

    np.testing.assert_allclose(
        state60[3:6],
        state40[3:6],
        rtol=1.0e-7,
        atol=1.0e-6,
    )
