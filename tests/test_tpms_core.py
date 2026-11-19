import numpy as np
import pytest

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.tpms_core import (
    build_homogenized_tpms_core,
    representative_transverse_shear_modulus,
)


def test_isotropic_tensor_produces_equal_transverse_shear():
    C = isotropic_stiffness_3d(
        E=2.0e9,
        nu=0.25,
    )

    core = build_homogenized_tpms_core(
        C
    )

    np.testing.assert_allclose(
        core.G_xz,
        core.G_yz,
        rtol=1.0e-12,
    )

    assert np.isclose(
        core.shear_anisotropy_xz_yz,
        1.0,
    )


def test_scalar_shear_modulus_for_nearly_equal_transverse_shear():
    C = isotropic_stiffness_3d(
        E=2.0e9,
        nu=0.25,
    )

    core = build_homogenized_tpms_core(
        C
    )

    G = (
        representative_transverse_shear_modulus(
            core
        )
    )

    np.testing.assert_allclose(
        G,
        core.G_xz,
    )


def test_strong_transverse_shear_anisotropy_is_not_silently_averaged():
    C = isotropic_stiffness_3d(
        E=2.0e9,
        nu=0.25,
    )

    C[5, 5] *= 0.5

    core = build_homogenized_tpms_core(
        C
    )

    with pytest.raises(
        ValueError
    ):
        representative_transverse_shear_modulus(
            core,
            relative_tolerance=0.05,
        )