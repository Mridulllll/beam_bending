import numpy as np

from host.beam_point_loads import (
    PointForce,
)
from host.constitutive import isotropic_stiffness_3d
from host.sandwich_warping import material_informed_symmetric_warping

from host.beam_two_mode import (
    WarpingMode,
    build_two_mode_beam_section,
)
from host.beam_two_mode_point_loads import (
    solve_simply_supported_two_mode_point_loads,
)
from host.variational_shear_modes import (
    BeamLayer2D,
    derive_variational_second_shear_mode,
    free_stress_beam_stiffness,
)


def K5(width=30.0):
    h = 12.0
    hc = 10.0

    Cface = free_stress_beam_stiffness(
        isotropic_stiffness_3d(
            E=70000.0,
            nu=0.33,
        )
    )

    Ccore = free_stress_beam_stiffness(
        isotropic_stiffness_3d(
            E=3500.0,
            nu=0.30,
        )
    )

    f1 = material_informed_symmetric_warping(
        h=h,
        hc=hc,
        G_face=Cface[1, 1],
        G_core=Ccore[1, 1],
    )

    half = (
        BeamLayer2D("core+", 0.0, 5.0, Ccore),
        BeamLayer2D("face+", 5.0, 6.0, Cface),
    )

    f2 = derive_variational_second_shear_mode(
        half,
        f1.f,
        elements_per_layer=(100, 10),
    )

    full = (
        BeamLayer2D("face-", -6.0, -5.0, Cface),
        BeamLayer2D("core", -5.0, 5.0, Ccore),
        BeamLayer2D("face+", 5.0, 6.0, Cface),
    )

    return build_two_mode_beam_section(
        full,
        width=width,
        mode1=WarpingMode(f1.f, f1.df),
        mode2=WarpingMode(f2.f, f2.df),
        quadrature_order=8,
        integration_breakpoints=np.unique(
            np.concatenate(
                (
                    -f2.nodes_half[::-1],
                    f2.nodes_half,
                )
            )
        ),
    ).K5


def solve(width=30.0):
    return solve_simply_supported_two_mode_point_loads(
        K5(width),
        length=120.0,
        support_left=10.0,
        support_right=110.0,
        applied_forces=(
            PointForce(40.0, 500.0),
            PointForce(80.0, 500.0),
        ),
        max_segment_length=8.0,
    )


def test_four_point_solution_satisfies_supports_and_free_end_conditions():
    solution = solve()

    left_support = solution.state(10.0)
    right_support = solution.state(110.0)
    right_end = solution.state(120.0)

    np.testing.assert_allclose(
        left_support[0],
        0.0,
        atol=1.0e-11,
    )

    np.testing.assert_allclose(
        right_support[0],
        0.0,
        atol=1.0e-11,
    )

    np.testing.assert_allclose(
        right_end[4:8],
        0.0,
        atol=2.0e-7,
    )


def test_four_point_constant_moment_region_has_zero_classical_shear():
    solution = solve()

    state_a = solution.state(50.0)
    state_b = solution.state(60.0)
    state_c = solution.state(70.0)

    np.testing.assert_allclose(
        [state_a[5], state_b[5], state_c[5]],
        0.0,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        [state_a[4], state_b[4], state_c[4]],
        state_b[4],
        rtol=1.0e-11,
        atol=1.0e-7,
    )


def test_point_force_jump_changes_only_V():
    solution = solve()

    left = solution.state(
        40.0,
        side="left",
    )

    right = solution.state(
        40.0,
        side="right",
    )

    jump = right - left

    expected = np.zeros(8)
    expected[5] = -500.0

    np.testing.assert_allclose(
        jump,
        expected,
        atol=1.0e-9,
    )


def test_fixed_load_deflection_scales_inverse_with_width():
    s30 = solve(width=30.0)
    s60 = solve(width=60.0)

    w30 = s30.state(60.0)[0]
    w60 = s60.state(60.0)[0]

    np.testing.assert_allclose(
        w60,
        0.5 * w30,
        rtol=2.0e-10,
        atol=1.0e-11,
    )
