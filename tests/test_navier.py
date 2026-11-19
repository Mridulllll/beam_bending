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
)

from host.navier import (
    navier_mode,
    generalized_strain_amplitudes,
    navier_modal_stiffness,
    uniform_pressure_coefficient,
    navier_modal_load,
    solve_navier_mode,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def homogeneous_hsdt_plate():
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

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(K14)


def test_navier_wavenumbers():
    mode = navier_mode(
        a=2.0,
        b=1.0,
        m=2,
        n=3,
    )

    np.testing.assert_allclose(
        mode.alpha,
        np.pi,
    )

    np.testing.assert_allclose(
        mode.beta,
        3.0 * np.pi,
    )


def test_generalized_strain_amplitude_shape():
    G = generalized_strain_amplitudes(
        alpha=np.pi,
        beta=2.0 * np.pi,
    )

    assert G.shape == (11, 5)


def test_modal_stiffness_is_symmetric():
    K = homogeneous_hsdt_plate()

    Kmn = navier_modal_stiffness(
        K,
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


def test_modal_stiffness_positive_definite():
    K = homogeneous_hsdt_plate()

    Kmn = navier_modal_stiffness(
        K,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    eigvals = np.linalg.eigvalsh(Kmn)

    assert np.all(eigvals > 0.0)


def test_uniform_pressure_only_has_odd_modes():
    q0 = 1000.0

    assert (
        uniform_pressure_coefficient(
            q0,
            1.0,
            1.0,
            1,
            1,
        )
        != 0.0
    )

    assert (
        uniform_pressure_coefficient(
            q0,
            1.0,
            1.0,
            2,
            1,
        )
        == 0.0
    )

    assert (
        uniform_pressure_coefficient(
            q0,
            1.0,
            1.0,
            1,
            2,
        )
        == 0.0
    )


def test_uniform_pressure_coefficient_11():
    q0 = 1000.0

    q11 = uniform_pressure_coefficient(
        q0,
        1.0,
        1.0,
        1,
        1,
    )

    expected = (
        16.0
        * q0
        / np.pi**2
    )

    np.testing.assert_allclose(
        q11,
        expected,
    )


def test_modal_load_only_acts_on_w():
    f = navier_modal_load(
        q0=1000.0,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    assert f.shape == (5,)

    assert f[0] == 0.0
    assert f[1] == 0.0
    assert f[3] == 0.0
    assert f[4] == 0.0

    assert f[2] != 0.0


def test_single_mode_solution_is_finite():
    K = homogeneous_hsdt_plate()

    q = solve_navier_mode(
        K,
        q0=1000.0,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    assert q.shape == (5,)
    assert np.all(np.isfinite(q))


def test_positive_downward_load_gives_nonzero_w():
    K = homogeneous_hsdt_plate()

    q = solve_navier_mode(
        K,
        q0=1000.0,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    assert abs(q[2]) > 0.0

def test_basis_aware_modal_stiffness_is_symmetric():
    from host.navier import (
        navier_modal_stiffness_basis_aware,
    )

    K = homogeneous_hsdt_plate()

    Kmn = navier_modal_stiffness_basis_aware(
        K,
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


def test_basis_operators_have_expected_support():
    from host.navier import (
        generalized_strain_basis_operators,
    )

    operators = generalized_strain_basis_operators(
        np.pi,
        2.0 * np.pi,
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
        assert G.shape == (11, 5)

    # beta_x belongs only to CS.
    assert (
        np.linalg.norm(
            operators["CS"][9, :]
        )
        > 0.0
    )

    assert (
        np.linalg.norm(
            operators["SS"][9, :]
        )
        == 0.0
    )

    # beta_y belongs only to SC.
    assert (
        np.linalg.norm(
            operators["SC"][10, :]
        )
        > 0.0
    )


def test_basis_aware_solution_is_finite():
    from host.navier import (
        solve_navier_mode_basis_aware,
    )

    K = homogeneous_hsdt_plate()

    q = solve_navier_mode_basis_aware(
        K,
        q0=1000.0,
        a=1.0,
        b=0.8,
        m=1,
        n=1,
    )

    assert q.shape == (5,)
    assert np.all(np.isfinite(q))
