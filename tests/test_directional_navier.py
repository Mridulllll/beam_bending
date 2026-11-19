"""
Tests for the directional-warping Navier formulation.
"""

import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.generalized_stiffness import (
    Layer,
    assemble_directional_generalized_stiffness,
    assemble_generalized_stiffness,
    scalar_to_directional_warping,
)

from host.hsdt import (
    extract_directional_hsdt_stiffness,
    extract_hsdt_stiffness,
)

from host.navier import (
    center_deflection_uniform_pressure_directional_hsdt,
    directional_generalized_strain_basis_operators,
    directional_navier_modal_stiffness,
    generalized_strain_basis_operators,
    navier_modal_stiffness_basis_aware,
    solve_directional_navier_mode,
    solve_navier_mode_basis_aware,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def homogeneous_scalar_and_directional_hsdt_plate():
    """
    Construct physically identical scalar- and directional-warping
    HSDT stiffness matrices.

    This provides the exact scalar limit of the directional theory.
    """

    E = 70.0e9
    nu = 0.33
    h = 0.01

    C = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    layers = [
        Layer(
            "plate",
            -0.5 * h,
            +0.5 * h,
            C,
        )
    ]

    scalar_warping = (
        reddy_baseline_generalized_warping(h)
    )

    directional_warping = (
        scalar_to_directional_warping(
            scalar_warping
        )
    )

    K14 = assemble_generalized_stiffness(
        layers,
        scalar_warping,
        quadrature_order=10,
    )

    K15 = assemble_directional_generalized_stiffness(
        layers,
        directional_warping,
        quadrature_order=10,
    )

    K11 = extract_hsdt_stiffness(
        K14
    )

    K12 = extract_directional_hsdt_stiffness(
        K15
    )

    return K11, K12


def test_directional_basis_operator_shapes():
    operators = (
        directional_generalized_strain_basis_operators(
            alpha=np.pi,
            beta=2.0 * np.pi,
        )
    )

    assert set(
        operators.keys()
    ) == {
        "SS",
        "CC",
        "CS",
        "SC",
    }

    for G in operators.values():
        assert G.shape == (
            12,
            5,
        )


def test_directional_twisting_warping_support():
    alpha = np.pi
    beta = 2.0 * np.pi

    operators = (
        directional_generalized_strain_basis_operators(
            alpha,
            beta,
        )
    )

    Gcc = operators["CC"]

    # chi_xy_x = beta_x,y
    np.testing.assert_allclose(
        Gcc[8, :],
        np.array(
            [
                0.0,
                0.0,
                0.0,
                beta,
                0.0,
            ]
        ),
    )

    # chi_xy_y = beta_y,x
    np.testing.assert_allclose(
        Gcc[9, :],
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                alpha,
            ]
        ),
    )


def test_directional_shear_support():
    operators = (
        directional_generalized_strain_basis_operators(
            alpha=np.pi,
            beta=2.0 * np.pi,
        )
    )

    # beta_x belongs to CS.
    assert (
        np.linalg.norm(
            operators["CS"][10, :]
        )
        > 0.0
    )

    assert (
        np.linalg.norm(
            operators["SS"][10, :]
        )
        == 0.0
    )

    # beta_y belongs to SC.
    assert (
        np.linalg.norm(
            operators["SC"][11, :]
        )
        > 0.0
    )

    assert (
        np.linalg.norm(
            operators["SS"][11, :]
        )
        == 0.0
    )


def test_scalar_and_directional_modal_operators_are_compatible():
    """
    In the scalar limit the old chi_xy row equals the sum of the
    two directional twisting-warping rows.
    """

    alpha = 1.7
    beta = 2.3

    scalar = generalized_strain_basis_operators(
        alpha,
        beta,
    )

    directional = (
        directional_generalized_strain_basis_operators(
            alpha,
            beta,
        )
    )

    np.testing.assert_allclose(
        scalar["SS"][0:8, :],
        directional["SS"][0:8, :],
    )

    np.testing.assert_allclose(
        scalar["CC"][0:8, :],
        directional["CC"][0:8, :],
    )

    np.testing.assert_allclose(
        scalar["CC"][8, :],
        (
            directional["CC"][8, :]
            + directional["CC"][9, :]
        ),
    )

    np.testing.assert_allclose(
        scalar["CS"][9, :],
        directional["CS"][10, :],
    )

    np.testing.assert_allclose(
        scalar["SC"][10, :],
        directional["SC"][11, :],
    )


def test_directional_modal_stiffness_is_symmetric():
    _, K12 = (
        homogeneous_scalar_and_directional_hsdt_plate()
    )

    Kmn = directional_navier_modal_stiffness(
        K12,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    np.testing.assert_allclose(
        Kmn,
        Kmn.T,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_directional_modal_stiffness_positive_definite():
    _, K12 = (
        homogeneous_scalar_and_directional_hsdt_plate()
    )

    Kmn = directional_navier_modal_stiffness(
        K12,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    eigvals = np.linalg.eigvalsh(
        Kmn
    )

    assert np.all(
        eigvals > 0.0
    )


def test_scalar_limit_reproduces_modal_stiffness():
    """
    Strong regression test:

    when fx = fy, the directional Navier formulation must reproduce
    the historical scalar basis-aware modal stiffness.
    """

    K11, K12 = (
        homogeneous_scalar_and_directional_hsdt_plate()
    )

    for m, n in [
        (1, 1),
        (1, 3),
        (3, 1),
        (3, 3),
    ]:
        Kmn_scalar = (
            navier_modal_stiffness_basis_aware(
                K11,
                a=1.0,
                b=0.8,
                m=m,
                n=n,
            )
        )

        Kmn_directional = (
            directional_navier_modal_stiffness(
                K12,
                a=1.0,
                b=0.8,
                m=m,
                n=n,
            )
        )

        np.testing.assert_allclose(
            Kmn_directional,
            Kmn_scalar,
            rtol=1.0e-11,
            atol=1.0e-7,
        )


def test_scalar_limit_reproduces_modal_solution():
    K11, K12 = (
        homogeneous_scalar_and_directional_hsdt_plate()
    )

    q_scalar = solve_navier_mode_basis_aware(
        K11,
        q0=1000.0,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    q_directional = solve_directional_navier_mode(
        K12,
        q0=1000.0,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    np.testing.assert_allclose(
        q_directional,
        q_scalar,
        rtol=1.0e-11,
        atol=1.0e-15,
    )


def test_directional_center_deflection_is_finite():
    _, K12 = (
        homogeneous_scalar_and_directional_hsdt_plate()
    )

    w = (
        center_deflection_uniform_pressure_directional_hsdt(
            K12,
            q0=1000.0,
            a=1.0,
            b=0.8,
            max_mode=5,
        )
    )

    assert np.isfinite(
        w
    )

    assert abs(
        w
    ) > 0.0