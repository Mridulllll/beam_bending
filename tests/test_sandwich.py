import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
)

from host.sandwich import (
    build_three_layer_sandwich,
)


def build_reference_sandwich(
    core_E,
):
    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=core_E,
        nu=0.25,
    )

    return build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=C_core,
        C_top=C_face,
        hf_bottom=0.5e-3,
        hc=10.0e-3,
        hf_top=0.5e-3,
    )


def test_three_layer_geometry():
    sandwich = build_reference_sandwich(
        1.0e9
    )

    assert np.isclose(
        sandwich.h_total,
        11.0e-3,
    )

    assert len(
        sandwich.layers
    ) == 3

    assert np.isclose(
        sandwich.layers[0].z_bottom,
        -5.5e-3,
    )

    assert np.isclose(
        sandwich.layers[-1].z_top,
        +5.5e-3,
    )


def test_symmetric_sandwich_stiffness_is_symmetric():
    sandwich = build_reference_sandwich(
        1.0e9
    )

    np.testing.assert_allclose(
        sandwich.K11,
        sandwich.K11.T,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_softer_core_increases_deflection():
    stiff = build_reference_sandwich(
        5.0e9
    )

    soft = build_reference_sandwich(
        0.2e9
    )

    kwargs = dict(
        q0=10.0e3,
        a=0.1,
        b=0.1,
        max_mode=15,
    )

    w_stiff = (
        center_deflection_uniform_pressure_hsdt(
            stiff.K11,
            **kwargs,
        )
    )

    w_soft = (
        center_deflection_uniform_pressure_hsdt(
            soft.K11,
            **kwargs,
        )
    )

    assert abs(w_soft) > abs(w_stiff)

def test_custom_warping_is_supported():
    from host.sandwich_warping import (
        material_informed_generalized_warping,
    )

    E_face = 70.0e9
    nu_face = 0.33

    E_core = 0.7e9
    nu_core = 0.25

    hf = 0.5e-3
    hc = 10.0e-3
    h = 2.0 * hf + hc

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

    C_core = isotropic_stiffness_3d(
        E=E_core,
        nu=nu_core,
    )

    G_face = (
        E_face
        / (2.0 * (1.0 + nu_face))
    )

    G_core = (
        E_core
        / (2.0 * (1.0 + nu_core))
    )

    warping = (
        material_informed_generalized_warping(
            h=h,
            hc=hc,
            G_face=G_face,
            G_core=G_core,
        )
    )

    sandwich = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=C_core,
        C_top=C_face,
        hf_bottom=hf,
        hc=hc,
        hf_top=hf,
        warping=warping,
    )

    assert sandwich.K11.shape == (11, 11)

    np.testing.assert_allclose(
        sandwich.K11,
        sandwich.K11.T,
        rtol=1.0e-12,
        atol=1.0e-8,
    )
