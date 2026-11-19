import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
)

from host.fsdt_reference import (
    build_fsdt_section,
    fsdt_modal_stiffness,
    center_deflection_uniform_pressure_fsdt,
)

from host.generalized_stiffness import (
    Layer,
)


def reference_layers(
    E_core=0.7e9,
):
    E_face = 70.0e9

    C_face = (
        sigma_zz_zero_stiffness(
            isotropic_stiffness_3d(
                E=E_face,
                nu=0.33,
            )
        )
    )

    C_core = (
        sigma_zz_zero_stiffness(
            isotropic_stiffness_3d(
                E=E_core,
                nu=0.25,
            )
        )
    )

    return [
        Layer(
            "bottom_face",
            -5.5e-3,
            -5.0e-3,
            C_face,
        ),
        Layer(
            "core",
            -5.0e-3,
            +5.0e-3,
            C_core,
        ),
        Layer(
            "top_face",
            +5.0e-3,
            +5.5e-3,
            C_face,
        ),
    ]


def test_fsdt_section_shapes():
    section = build_fsdt_section(
        reference_layers()
    )

    assert section.D.shape == (3, 3)
    assert section.S.shape == (2, 2)


def test_fsdt_section_is_symmetric():
    section = build_fsdt_section(
        reference_layers()
    )

    np.testing.assert_allclose(
        section.D,
        section.D.T,
        rtol=1.0e-12,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        section.S,
        section.S.T,
        rtol=1.0e-12,
        atol=1.0e-10,
    )


def test_fsdt_modal_stiffness_positive_definite():
    section = build_fsdt_section(
        reference_layers()
    )

    K = fsdt_modal_stiffness(
        section,
        a=0.1,
        b=0.1,
        m=1,
        n=1,
    )

    eigvals = np.linalg.eigvalsh(
        K
    )

    assert np.all(
        eigvals > 0.0
    )


def test_fsdt_solution_is_finite():
    section = build_fsdt_section(
        reference_layers()
    )

    w = (
        center_deflection_uniform_pressure_fsdt(
            section,
            q0=10.0e3,
            a=0.1,
            b=0.1,
            max_mode=15,
        )
    )

    assert np.isfinite(w)
    assert abs(w) > 0.0


def test_softer_core_increases_fsdt_deflection():
    stiff = build_fsdt_section(
        reference_layers(
            E_core=7.0e9,
        )
    )

    soft = build_fsdt_section(
        reference_layers(
            E_core=0.07e9,
        )
    )

    kwargs = dict(
        q0=10.0e3,
        a=0.1,
        b=0.1,
        max_mode=15,
    )

    w_stiff = (
        center_deflection_uniform_pressure_fsdt(
            stiff,
            **kwargs,
        )
    )

    w_soft = (
        center_deflection_uniform_pressure_fsdt(
            soft,
            **kwargs,
        )
    )

    assert abs(w_soft) > abs(w_stiff)

def test_energy_equivalent_shear_stiffness_softens_with_core():
    from host.fsdt_reference import (
        energy_equivalent_shear_stiffness,
    )

    stiff_layers = reference_layers(
        E_core=7.0e9,
    )

    soft_layers = reference_layers(
        E_core=0.07e9,
    )

    S_stiff = energy_equivalent_shear_stiffness(
        stiff_layers
    )

    S_soft = energy_equivalent_shear_stiffness(
        soft_layers
    )

    assert S_soft[0, 0] < S_stiff[0, 0]
    assert S_soft[1, 1] < S_stiff[1, 1]


def test_energy_equivalent_shear_matches_harmonic_layer_formula():
    from host.fsdt_reference import (
        energy_equivalent_shear_stiffness,
    )

    E_face = 70.0e9
    nu_face = 0.33

    E_core = 0.7e9
    nu_core = 0.25

    G_face = (
        E_face
        / (2.0 * (1.0 + nu_face))
    )

    G_core = (
        E_core
        / (2.0 * (1.0 + nu_core))
    )

    hf = 0.5e-3
    hc = 10.0e-3
    h = 2.0 * hf + hc

    layers = reference_layers(
        E_core=E_core
    )

    S = energy_equivalent_shear_stiffness(
        layers
    )

    expected = (
        h**2
        / (
            2.0 * hf / G_face
            + hc / G_core
        )
    )

    np.testing.assert_allclose(
        S[0, 0],
        expected,
        rtol=1.0e-10,
    )

    np.testing.assert_allclose(
        S[1, 1],
        expected,
        rtol=1.0e-10,
    )
