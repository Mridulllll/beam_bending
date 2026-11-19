import numpy as np
import pytest

from host.constitutive import (
    VOIGT_ORDER,
    INPLANE_INDICES,
    isotropic_stiffness_3d,
    validate_stiffness_matrix,
)


def test_permanent_voigt_order():
    assert VOIGT_ORDER == (
        "xx",
        "yy",
        "zz",
        "xy",
        "xz",
        "yz",
    )

    assert INPLANE_INDICES == (0, 1, 3)


def test_isotropic_stiffness_is_symmetric_positive_definite():
    C = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    np.testing.assert_allclose(
        C,
        C.T,
        rtol=0.0,
        atol=0.0,
    )

    eigvals = np.linalg.eigvalsh(C)

    assert np.all(eigvals > 0.0)


def test_invalid_stiffness_shape_rejected():
    with pytest.raises(ValueError):
        validate_stiffness_matrix(
            np.eye(3)
        )


def test_nonsymmetric_stiffness_rejected():
    C = np.eye(6)
    C[0, 1] = 0.5

    with pytest.raises(ValueError):
        validate_stiffness_matrix(C)

def test_isotropic_plane_stress_condensation():
    import numpy as np

    from host.constitutive import (
        isotropic_stiffness_3d,
        plane_stress_condensation,
    )

    E = 70.0e9
    nu = 0.33

    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    Q = plane_stress_condensation(C3)

    expected = (
        E
        / (1.0 - nu**2)
        * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [
                    0.0,
                    0.0,
                    (1.0 - nu) / 2.0,
                ],
            ]
        )
    )

    np.testing.assert_allclose(
        Q,
        expected,
        rtol=1.0e-12,
        atol=1.0e-3,
    )


def test_plane_stress_condensation_is_symmetric():
    import numpy as np

    from host.constitutive import (
        isotropic_stiffness_3d,
        plane_stress_condensation,
    )

    C3 = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    Q = plane_stress_condensation(C3)

    np.testing.assert_allclose(
        Q,
        Q.T,
        rtol=1.0e-14,
        atol=1.0e-8,
    )


def test_plane_stress_bending_rigidity():
    import numpy as np

    from host.constitutive import (
        isotropic_stiffness_3d,
        plane_stress_condensation,
    )

    E = 70.0e9
    nu = 0.33
    h = 0.01

    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    Q = plane_stress_condensation(C3)

    D_from_Q = (
        Q[0, 0]
        * h**3
        / 12.0
    )

    D_classical = (
        E
        * h**3
        / (
            12.0
            * (1.0 - nu**2)
        )
    )

    np.testing.assert_allclose(
        D_from_Q,
        D_classical,
        rtol=1.0e-12,
    )


def test_sigma_zz_zero_reduction_retains_transverse_shear():
    import numpy as np

    from host.constitutive import (
        isotropic_stiffness_3d,
        sigma_zz_zero_stiffness,
    )

    E = 70.0e9
    nu = 0.33

    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    C5 = sigma_zz_zero_stiffness(C3)

    G = E / (2.0 * (1.0 + nu))

    # Plane-stress in-plane terms.
    expected_C11 = E / (1.0 - nu**2)
    expected_C12 = nu * E / (1.0 - nu**2)

    np.testing.assert_allclose(
        C5[0, 0],
        expected_C11,
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        C5[0, 1],
        expected_C12,
        rtol=1.0e-12,
    )

    # In-plane shear remains.
    np.testing.assert_allclose(
        C5[3, 3],
        G,
        rtol=1.0e-12,
    )

    # Crucially, transverse shear remains.
    np.testing.assert_allclose(
        C5[4, 4],
        G,
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(
        C5[5, 5],
        G,
        rtol=1.0e-12,
    )


def test_sigma_zz_zero_reduction_eliminates_zz_row_column():
    import numpy as np

    from host.constitutive import (
        isotropic_stiffness_3d,
        sigma_zz_zero_stiffness,
    )

    C3 = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C5 = sigma_zz_zero_stiffness(C3)

    np.testing.assert_allclose(
        C5[2, :],
        0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        C5[:, 2],
        0.0,
        atol=0.0,
    )
