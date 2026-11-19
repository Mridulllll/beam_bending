import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.generalized_stiffness import (
    Layer,
    assemble_generalized_stiffness,
)

from host.hsdt import (
    extract_hsdt_stiffness,
    hsdt_strain_energy_density,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def test_hsdt_reduction_shape():
    C = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    h = 0.01

    layers = [
        Layer(
            "homogeneous_plate",
            -0.5 * h,
            +0.5 * h,
            C,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=8,
    )

    K11 = extract_hsdt_stiffness(K14)

    assert K11.shape == (11, 11)


def test_hsdt_stiffness_is_symmetric():
    C = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    h = 0.01

    layers = [
        Layer(
            "homogeneous_plate",
            -0.5 * h,
            +0.5 * h,
            C,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=8,
    )

    K11 = extract_hsdt_stiffness(K14)

    np.testing.assert_allclose(
        K11,
        K11.T,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_hsdt_generalized_energy_positive_for_nonzero_state():
    C = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    h = 0.01

    layers = [
        Layer(
            "homogeneous_plate",
            -0.5 * h,
            +0.5 * h,
            C,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=8,
    )

    K11 = extract_hsdt_stiffness(K14)

    E = np.zeros(11)

    # Activate a mixture of classical and higher-order strains.
    E[0] = 1.0e-4
    E[3] = 1.0e-2
    E[6] = 2.0e-3
    E[9] = 1.0e-3

    energy = hsdt_strain_energy_density(
        E,
        K11,
    )

    assert energy > 0.0


def test_reddy_baseline_has_zero_quasi3d_rows_and_columns():
    C = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    h = 0.01

    layers = [
        Layer(
            "homogeneous_plate",
            -0.5 * h,
            +0.5 * h,
            C,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=8,
    )

    # eta_x, eta_y, eta correspond to 11:14.
    np.testing.assert_allclose(
        K14[11:14, :],
        0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        K14[:, 11:14],
        0.0,
        atol=1.0e-12,
    )