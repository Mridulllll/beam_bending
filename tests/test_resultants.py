import numpy as np

from host.resultants import (
    generalized_resultants,
    internal_virtual_work_density,
)


def test_generalized_resultant_partition():
    K = np.eye(11)

    E = np.arange(
        1.0,
        12.0,
    )

    R = generalized_resultants(
        K,
        E,
    )

    np.testing.assert_allclose(
        R.N,
        E[0:3],
    )

    np.testing.assert_allclose(
        R.M,
        E[3:6],
    )

    np.testing.assert_allclose(
        R.P,
        E[6:9],
    )

    np.testing.assert_allclose(
        R.Q,
        E[9:11],
    )

    np.testing.assert_allclose(
        R.full,
        E,
    )


def test_resultants_follow_constitutive_mapping():
    rng = np.random.default_rng(1234)

    A = rng.standard_normal(
        (11, 11)
    )

    K = (
        A.T @ A
        + np.eye(11)
    )

    E = rng.standard_normal(11)

    R = generalized_resultants(
        K,
        E,
    )

    np.testing.assert_allclose(
        R.full,
        K @ E,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_virtual_work_density():
    K = np.eye(11)

    E = np.arange(
        1.0,
        12.0,
    )

    delta_E = np.linspace(
        0.1,
        1.1,
        11,
    )

    R = generalized_resultants(
        K,
        E,
    )

    work = internal_virtual_work_density(
        delta_E,
        R,
    )

    expected = float(
        delta_E @ E
    )

    np.testing.assert_allclose(
        work,
        expected,
    )