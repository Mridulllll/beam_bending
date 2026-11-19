"""
Verification of the directional HOST generalized strain operator.
"""

import sympy as sp

from host.generalized_strains import (
    build_directional_generalized_strain_system,
    build_generalized_strain_system,
)


def test_directional_generalized_operator_dimensions():
    system = build_directional_generalized_strain_system()

    assert system.E.shape == (
        15,
        1,
    )

    assert system.F.shape == (
        6,
        15,
    )

    assert system.reconstructed_strain.shape == (
        6,
        1,
    )


def test_directional_operator_reconstructs_exact_strain():
    system = build_directional_generalized_strain_system()

    residual = (
        system.reconstructed_strain
        - system.kin.strain
    )

    for expression in residual:
        assert sp.simplify(
            expression
        ) == 0


def test_directional_gamma_xy_has_separate_warping_terms():
    system = build_directional_generalized_strain_system()

    kin = system.kin

    expected = (
        sp.diff(
            kin.u0,
            kin.y,
        )
        + sp.diff(
            kin.v0,
            kin.x,
        )
        - 2
        * kin.z
        * sp.diff(
            kin.w0,
            kin.x,
            kin.y,
        )
        + kin.fx
        * sp.diff(
            kin.beta_x,
            kin.y,
        )
        + kin.fy
        * sp.diff(
            kin.beta_y,
            kin.x,
        )
    )

    assert sp.simplify(
        system.reconstructed_strain[3]
        - expected
    ) == 0


def test_directional_transverse_shear_terms():
    system = build_directional_generalized_strain_system()

    kin = system.kin

    expected_xz = (
        sp.diff(
            kin.fx,
            kin.z,
        )
        * kin.beta_x
        + kin.g
        * sp.diff(
            kin.eta,
            kin.x,
        )
    )

    expected_yz = (
        sp.diff(
            kin.fy,
            kin.z,
        )
        * kin.beta_y
        + kin.g
        * sp.diff(
            kin.eta,
            kin.y,
        )
    )

    assert sp.simplify(
        system.reconstructed_strain[4]
        - expected_xz
    ) == 0

    assert sp.simplify(
        system.reconstructed_strain[5]
        - expected_yz
    ) == 0


def test_scalar_system_remains_14_component():
    system = build_generalized_strain_system()

    assert system.E.shape == (
        14,
        1,
    )

    assert system.F.shape == (
        6,
        14,
    )


def test_scalar_limit_of_directional_strain_field():
    """
    Directional HOST must reduce to scalar HOST when

        fy(z) = fx(z).

    We compare the reconstructed physical 3-D strain fields rather
    than the generalized coordinates because the directional theory
    uses two independent chi_xy coordinates.
    """

    directional = (
        build_directional_generalized_strain_system()
    )

    scalar = (
        build_generalized_strain_system()
    )

    reduced_directional = sp.Matrix(
        [
            sp.simplify(
                expression.subs(
                    directional.kin.fy,
                    directional.kin.fx,
                )
            )
            for expression
            in directional.reconstructed_strain
        ]
    )

    # The symbolic field names are otherwise identical.
    residual = (
        reduced_directional
        - scalar.reconstructed_strain
    )

    for expression in residual:
        assert sp.simplify(
            expression
        ) == 0