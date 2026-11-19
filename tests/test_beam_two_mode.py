import numpy as np

from host.constitutive import isotropic_stiffness_3d
from host.sandwich_warping import material_informed_symmetric_warping

from host.beam_two_mode import (
    WarpingMode,
    boundary_layer_spectrum,
    build_two_mode_beam_section,
    local_shear_condensation_ratio,
    two_mode_state_derivative,
    two_mode_state_matrix,
)
from host.variational_shear_modes import (
    BeamLayer2D,
    derive_variational_second_shear_mode,
    free_stress_beam_stiffness,
)


def build_section(width=30.0):
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

    Gface = Cface[1, 1]
    Gcore = Ccore[1, 1]

    f1 = material_informed_symmetric_warping(
        h=h,
        hc=hc,
        G_face=Gface,
        G_core=Gcore,
    )

    half_layers = (
        BeamLayer2D("core+", 0.0, 5.0, Ccore),
        BeamLayer2D("face+", 5.0, 6.0, Cface),
    )

    f2 = derive_variational_second_shear_mode(
        half_layers,
        f1.f,
        elements_per_layer=(120, 12),
    )

    full_layers = (
        BeamLayer2D("face-", -6.0, -5.0, Cface),
        BeamLayer2D("core", -5.0, 5.0, Ccore),
        BeamLayer2D("face+", 5.0, 6.0, Cface),
    )

    return build_two_mode_beam_section(
        full_layers,
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
    )


def test_two_mode_section_is_symmetric_positive_definite():
    section = build_section()

    np.testing.assert_allclose(
        section.K6,
        section.K6.T,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        section.K5,
        section.K5.T,
        atol=1.0e-8,
    )

    assert np.min(np.linalg.eigvalsh(section.K6)) > 0.0
    assert np.min(np.linalg.eigvalsh(section.K5)) > 0.0


def test_two_mode_section_scales_linearly_with_width():
    section30 = build_section(width=30.0)
    section60 = build_section(width=60.0)

    np.testing.assert_allclose(
        section60.K6,
        2.0 * section30.K6,
        rtol=2.0e-12,
        atol=2.0e-5,
    )

    np.testing.assert_allclose(
        section60.K5,
        2.0 * section30.K5,
        rtol=2.0e-12,
        atol=2.0e-5,
    )


def test_two_mode_state_matrix_matches_direct_derivative():
    K5 = build_section().K5

    A = two_mode_state_matrix(K5)

    y = np.asarray(
        [
            0.02,
            -0.001,
            0.003,
            -0.002,
            1200.0,
            -75.0,
            80.0,
            -25.0,
        ]
    )

    np.testing.assert_allclose(
        A @ y,
        two_mode_state_derivative(
            K5,
            y,
        ),
        rtol=1.0e-12,
        atol=1.0e-11,
    )


def test_two_mode_has_four_rigid_polynomial_and_four_higher_order_modes():
    spectrum = boundary_layer_spectrum(
        build_section().K5
    )

    assert spectrum["zero_eigenvalue_count"] == 4
    assert len(spectrum["modes"]) == 4
    assert len(spectrum["stable_modes"]) == 2

    for mode in spectrum["stable_modes"]:
        assert np.isfinite(mode["decay_length"])
        assert mode["decay_length"] > 0.0


def test_local_secondary_mode_condensation_does_not_increase_stiffness():
    alpha = local_shear_condensation_ratio(
        build_section().K5
    )

    assert 0.0 < alpha <= 1.0
