"""
Tests for directional HOST generalized stiffness.
"""

import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.generalized_stiffness import (
    DirectionalWarpingFunctions,
    Layer,
    WarpingFunctions,
    assemble_directional_generalized_stiffness,
    assemble_generalized_stiffness,
    directional_generalized_strain_energy_density,
    directional_thickness_operator,
    scalar_to_directional_warping,
    thickness_operator,
)


def scalar_warping():
    return WarpingFunctions(
        f=lambda z: z + 0.2 * z**3,
        df=lambda z: 1.0 + 0.6 * z**2,
        g=lambda z: 0.1 * z,
        dg=lambda z: 0.1,
    )


def directional_warping():
    return DirectionalWarpingFunctions(
        fx=lambda z: z + 0.2 * z**3,
        dfx=lambda z: 1.0 + 0.6 * z**2,

        fy=lambda z: 0.8 * z + 0.1 * z**3,
        dfy=lambda z: 0.8 + 0.3 * z**2,

        g=lambda z: 0.1 * z,
        dg=lambda z: 0.1,
    )


def homogeneous_layer():
    C = isotropic_stiffness_3d(
        E=70000.0,
        nu=0.33,
    )

    return Layer(
        "plate",
        -0.5,
        +0.5,
        C,
    )


def test_directional_thickness_operator_shape():
    F = directional_thickness_operator(
        0.1,
        directional_warping(),
    )

    assert F.shape == (
        6,
        15,
    )


def test_directional_thickness_operator_uses_fx_and_fy():
    z = 0.2

    warping = directional_warping()

    F = directional_thickness_operator(
        z,
        warping,
    )

    np.testing.assert_allclose(
        F[0, 6],
        warping.fx(z),
    )

    np.testing.assert_allclose(
        F[1, 7],
        warping.fy(z),
    )

    np.testing.assert_allclose(
        F[3, 8],
        warping.fx(z),
    )

    np.testing.assert_allclose(
        F[3, 9],
        warping.fy(z),
    )

    np.testing.assert_allclose(
        F[4, 10],
        warping.dfx(z),
    )

    np.testing.assert_allclose(
        F[5, 11],
        warping.dfy(z),
    )


def test_directional_stiffness_shape_and_symmetry():
    K = assemble_directional_generalized_stiffness(
        [
            homogeneous_layer()
        ],
        directional_warping(),
        quadrature_order=10,
    )

    assert K.shape == (
        15,
        15,
    )

    np.testing.assert_allclose(
        K,
        K.T,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_directional_stiffness_has_positive_energy():
    K = assemble_directional_generalized_stiffness(
        [
            homogeneous_layer()
        ],
        directional_warping(),
        quadrature_order=10,
    )

    E = np.zeros(
        15,
        dtype=float,
    )

    E[0] = 1.0e-3
    E[3] = 2.0e-4
    E[10] = 3.0e-3

    U = directional_generalized_strain_energy_density(
        E,
        K,
    )

    assert U > 0.0


def test_scalar_to_directional_operator_reproduces_physical_strains():
    """
    In the scalar limit, the 15-component directional operator must
    reproduce the same physical 3-D strain field as the historical
    14-component operator.

    The directional generalized coordinates use

        chi_xy_x
        chi_xy_y

    separately, whereas the scalar formulation uses

        chi_xy = chi_xy_x + chi_xy_y.
    """

    scalar = scalar_warping()

    directional = scalar_to_directional_warping(
        scalar
    )

    z = 0.17

    F14 = thickness_operator(
        z,
        scalar,
    )

    F15 = directional_thickness_operator(
        z,
        directional,
    )

    E15 = np.array(
        [
            1.0e-3,
            -2.0e-3,
            3.0e-4,

            4.0e-4,
            -5.0e-4,
            6.0e-4,

            7.0e-4,
            -8.0e-4,

            9.0e-4,
            -1.1e-3,

            1.2e-3,
            -1.3e-3,

            1.4e-3,
            -1.5e-3,

            1.6e-3,
        ],
        dtype=float,
    )

    E14 = np.array(
        [
            E15[0],
            E15[1],
            E15[2],

            E15[3],
            E15[4],
            E15[5],

            E15[6],
            E15[7],
            E15[8] + E15[9],

            E15[10],
            E15[11],

            E15[12],
            E15[13],

            E15[14],
        ],
        dtype=float,
    )

    strain14 = F14 @ E14
    strain15 = F15 @ E15

    np.testing.assert_allclose(
        strain15,
        strain14,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_scalar_limit_reproduces_generalized_energy():
    """
    Scalar and directional formulations must give identical physical
    energy when the directional warping functions are identical and
    the generalized coordinates represent the same strain field.
    """

    layer = homogeneous_layer()

    scalar = scalar_warping()

    directional = scalar_to_directional_warping(
        scalar
    )

    K14 = assemble_generalized_stiffness(
        [
            layer
        ],
        scalar,
        quadrature_order=10,
    )

    K15 = assemble_directional_generalized_stiffness(
        [
            layer
        ],
        directional,
        quadrature_order=10,
    )

    E15 = np.array(
        [
            1.0e-3,
            -2.0e-3,
            3.0e-4,

            4.0e-4,
            -5.0e-4,
            6.0e-4,

            7.0e-4,
            -8.0e-4,

            9.0e-4,
            -1.1e-3,

            1.2e-3,
            -1.3e-3,

            1.4e-3,
            -1.5e-3,

            1.6e-3,
        ],
        dtype=float,
    )

    E14 = np.array(
        [
            E15[0],
            E15[1],
            E15[2],

            E15[3],
            E15[4],
            E15[5],

            E15[6],
            E15[7],
            E15[8] + E15[9],

            E15[10],
            E15[11],

            E15[12],
            E15[13],

            E15[14],
        ],
        dtype=float,
    )

    U14 = 0.5 * float(
        E14
        @ K14
        @ E14
    )

    U15 = (
        directional_generalized_strain_energy_density(
            E15,
            K15,
        )
    )

    np.testing.assert_allclose(
        U15,
        U14,
        rtol=1.0e-11,
        atol=1.0e-13,
    )