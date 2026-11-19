import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)
from host.generalized_stiffness import (
    Layer,
    WarpingFunctions,
    assemble_generalized_stiffness,
    classical_abd,
    extract_abd_from_generalized_stiffness,
    generalized_strain_energy_density,
)


def polynomial_test_warping():
    """
    Algebraic verification functions only.

    These are NOT yet the physical HOST warping functions.

        f(z) = z + z^3
        g(z) = 1 + z^2
    """

    return WarpingFunctions(
        f=lambda z: z + z**3,
        df=lambda z: 1.0 + 3.0 * z**2,
        g=lambda z: 1.0 + z**2,
        dg=lambda z: 2.0 * z,
    )


def symmetric_sandwich():
    """
    Symmetric three-layer verification sandwich.
    """

    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=1.0e9,
        nu=0.25,
    )

    hf = 0.5e-3
    hc = 10.0e-3

    h = 2.0 * hf + hc

    z0 = -0.5 * h
    z1 = -0.5 * hc
    z2 = +0.5 * hc
    z3 = +0.5 * h

    return [
        Layer(
            name="bottom_face",
            z_bottom=z0,
            z_top=z1,
            stiffness=C_face,
        ),
        Layer(
            name="core",
            z_bottom=z1,
            z_top=z2,
            stiffness=C_core,
        ),
        Layer(
            name="top_face",
            z_bottom=z2,
            z_top=z3,
            stiffness=C_face,
        ),
    ]


def test_generalized_stiffness_shape_and_symmetry():
    layers = symmetric_sandwich()

    K = assemble_generalized_stiffness(
        layers,
        polynomial_test_warping(),
        quadrature_order=8,
    )

    assert K.shape == (14, 14)

    np.testing.assert_allclose(
        K,
        K.T,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_abd_blocks_match_independent_integration():
    layers = symmetric_sandwich()

    K = assemble_generalized_stiffness(
        layers,
        polynomial_test_warping(),
        quadrature_order=8,
    )

    A_from_K, B_from_K, D_from_K = (
        extract_abd_from_generalized_stiffness(K)
    )

    A_ref, B_ref, D_ref = classical_abd(
        layers,
        quadrature_order=8,
    )

    np.testing.assert_allclose(
        A_from_K,
        A_ref,
        rtol=1.0e-12,
        atol=1.0e-5,
    )

    np.testing.assert_allclose(
        B_from_K,
        B_ref,
        rtol=1.0e-12,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        D_from_K,
        D_ref,
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_symmetric_sandwich_has_zero_classical_B():
    layers = symmetric_sandwich()

    A, B, D = classical_abd(
        layers,
        quadrature_order=8,
    )

    scale = max(
        1.0,
        np.linalg.norm(A, ord=np.inf),
    )

    assert (
        np.linalg.norm(B, ord=np.inf)
        / scale
        < 1.0e-12
    )


def test_asymmetric_sandwich_produces_nonzero_B():
    C_bottom = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=1.0e9,
        nu=0.25,
    )

    C_top = isotropic_stiffness_3d(
        E=35.0e9,
        nu=0.30,
    )

    layers = [
        Layer(
            "bottom_face",
            -5.5e-3,
            -5.0e-3,
            C_bottom,
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
            C_top,
        ),
    ]

    _, B, _ = classical_abd(
        layers,
        quadrature_order=8,
    )

    assert np.linalg.norm(B) > 0.0


def test_classical_strain_energy_is_positive():
    layers = symmetric_sandwich()

    K = assemble_generalized_stiffness(
        layers,
        polynomial_test_warping(),
        quadrature_order=8,
    )

    # Activate only classical membrane/bending generalized strains.
    Egen = np.zeros(14, dtype=float)

    Egen[0] = 1.0e-4
    Egen[1] = -0.5e-4
    Egen[2] = 0.25e-4

    Egen[3] = 0.01
    Egen[4] = -0.005
    Egen[5] = 0.0025

    energy = generalized_strain_energy_density(
        Egen,
        K,
    )

    assert energy > 0.0


def test_graded_core_stiffness_function_is_supported():
    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    def graded_core(z):
        # Verification grading only:
        # E varies linearly while remaining positive.
        E0 = 1.0e9
        factor = 1.0 + 20.0 * z

        return isotropic_stiffness_3d(
            E=E0 * factor,
            nu=0.25,
        )

    layers = [
        Layer(
            "bottom_face",
            -5.5e-3,
            -5.0e-3,
            C_face,
        ),
        Layer(
            "graded_core",
            -5.0e-3,
            +5.0e-3,
            graded_core,
        ),
        Layer(
            "top_face",
            +5.0e-3,
            +5.5e-3,
            C_face,
        ),
    ]

    K = assemble_generalized_stiffness(
        layers,
        polynomial_test_warping(),
        quadrature_order=10,
    )

    assert K.shape == (14, 14)
    assert np.all(np.isfinite(K))