import sympy as sp

from host.generalized_strains import (
    build_generalized_strain_system,
)


def test_generalized_operator_dimensions():
    system = build_generalized_strain_system()

    assert system.F.shape == (6, 14)
    assert system.E.shape == (14, 1)


def test_generalized_operator_reconstructs_exact_strain():
    system = build_generalized_strain_system()

    residual = (
        system.reconstructed_strain
        - system.kin.strain
    )

    for expression in residual:
        assert sp.simplify(expression) == 0


def test_membrane_bending_warping_structure():
    system = build_generalized_strain_system()

    F = system.F
    z = system.kin.z
    f = system.kin.f

    # eps_xx = eps0_xx + z*kappa_xx + f*chi_xx
    assert F[0, 0] == 1
    assert F[0, 3] == z
    assert F[0, 6] == f

    # eps_yy = eps0_yy + z*kappa_yy + f*chi_yy
    assert F[1, 1] == 1
    assert F[1, 4] == z
    assert F[1, 7] == f

    # gamma_xy = gamma0_xy + z*kappa_xy + f*chi_xy
    assert F[3, 2] == 1
    assert F[3, 5] == z
    assert F[3, 8] == f


def test_transverse_terms():
    system = build_generalized_strain_system()

    kin = system.kin
    F = system.F

    fp = sp.diff(kin.f, kin.z)
    gp = sp.diff(kin.g, kin.z)

    # eps_zz
    assert F[2, 13] == gp

    # gamma_xz
    assert F[4, 9] == fp
    assert F[4, 11] == kin.g

    # gamma_yz
    assert F[5, 10] == fp
    assert F[5, 12] == kin.g