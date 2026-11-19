import numpy as np

from host.constitutive import isotropic_stiffness_3d
from host.sandwich_warping import material_informed_symmetric_warping

from host.variational_shear_modes import (
    BeamLayer2D,
    derive_variational_second_shear_mode,
    free_stress_beam_stiffness,
)


def section():
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

    layers_half = (
        BeamLayer2D(
            "core",
            0.0,
            0.5 * hc,
            Ccore,
        ),
        BeamLayer2D(
            "face",
            0.5 * hc,
            0.5 * h,
            Cface,
        ),
    )

    return h, f1, layers_half


def test_free_stress_isotropic_reduction_recovers_E_and_G():
    E = 70000.0
    nu = 0.33

    Cb = free_stress_beam_stiffness(
        isotropic_stiffness_3d(
            E=E,
            nu=nu,
        )
    )

    np.testing.assert_allclose(
        Cb,
        np.diag(
            [
                E,
                E / (2.0 * (1.0 + nu)),
            ]
        ),
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_variational_mode_is_odd_and_derivative_even():
    h, f1, layers = section()

    mode = derive_variational_second_shear_mode(
        layers,
        f1.f,
        elements_per_layer=(80, 8),
    )

    for z in np.linspace(0.0, 0.5 * h, 19):
        np.testing.assert_allclose(
            mode.f(-z),
            -mode.f(z),
            atol=1.0e-12,
        )

        np.testing.assert_allclose(
            mode.df(-z),
            mode.df(z),
            atol=1.0e-12,
        )


def test_variational_mode_is_E_orthogonal_to_existing_mode():
    _, f1, layers = section()

    mode = derive_variational_second_shear_mode(
        layers,
        f1.f,
        elements_per_layer=(120, 12),
    )

    scale = max(
        1.0,
        abs(mode.modal_E_norm),
    )

    assert (
        abs(mode.E_orthogonality_residual) / scale
        < 1.0e-10
    )


def test_variational_mode_eigenvalue_converges_under_mesh_refinement():
    _, f1, layers = section()

    coarse = derive_variational_second_shear_mode(
        layers,
        f1.f,
        elements_per_layer=(60, 6),
    )

    fine = derive_variational_second_shear_mode(
        layers,
        f1.f,
        elements_per_layer=(120, 12),
    )

    relative = abs(
        coarse.eigenvalue - fine.eigenvalue
    ) / fine.eigenvalue

    assert relative < 5.0e-4
