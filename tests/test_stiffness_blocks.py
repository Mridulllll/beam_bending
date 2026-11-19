import numpy as np

from host.constitutive import isotropic_stiffness_3d

from host.generalized_stiffness import (
    Layer,
    WarpingFunctions,
    assemble_generalized_stiffness,
)

from host.stiffness_blocks import (
    BLOCK_RANGES,
    decompose_generalized_stiffness,
)


def symmetric_layers():
    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=1.0e9,
        nu=0.25,
    )

    return [
        Layer(
            "bottom_face",
            -5.5e-3,
            -5.0e-3,
            C_face,
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
            C_face,
        ),
    ]


def parity_warping():
    """
    Verification-only parity choice.

    f(z) = z^3
        odd

    f'(z) = 3 z^2
        even

    g(z) = z
        odd

    g'(z) = 1
        even
    """

    return WarpingFunctions(
        f=lambda z: z**3,
        df=lambda z: 3.0 * z**2,
        g=lambda z: z,
        dg=lambda z: 1.0,
    )


def test_block_ranges_cover_all_14_components():
    sizes = []

    for start, stop in BLOCK_RANGES.values():
        sizes.extend(range(start, stop))

    assert sorted(sizes) == list(range(14))


def test_decomposition_recovers_classical_blocks():
    K = assemble_generalized_stiffness(
        symmetric_layers(),
        parity_warping(),
        quadrature_order=10,
    )

    blocks = decompose_generalized_stiffness(K)

    np.testing.assert_allclose(
        blocks.A,
        K[0:3, 0:3],
    )

    np.testing.assert_allclose(
        blocks.B,
        K[0:3, 3:6],
    )

    np.testing.assert_allclose(
        blocks.D,
        K[3:6, 3:6],
    )


def test_symmetric_sandwich_has_zero_B():
    K = assemble_generalized_stiffness(
        symmetric_layers(),
        parity_warping(),
        quadrature_order=10,
    )

    blocks = decompose_generalized_stiffness(K)

    scale = max(
        1.0,
        np.linalg.norm(blocks.A),
    )

    assert np.linalg.norm(blocks.B) / scale < 1.0e-12


def test_odd_f_eliminates_membrane_warping_coupling():
    """
    For

        C(-z) = C(z)
        f(-z) = -f(z)

    the membrane-warping integrand is odd:

        C(z) f(z)

    and must integrate to zero over the symmetric thickness.
    """

    K = assemble_generalized_stiffness(
        symmetric_layers(),
        parity_warping(),
        quadrature_order=12,
    )

    blocks = decompose_generalized_stiffness(K)

    scale = max(
        1.0,
        np.linalg.norm(blocks.A),
    )

    assert (
        np.linalg.norm(blocks.membrane_warping)
        / scale
        < 1.0e-12
    )


def test_odd_f_does_not_eliminate_bending_warping_coupling():
    """
    z*f(z) is even when f is odd.

    Therefore bending-warping coupling generally remains nonzero.
    """

    K = assemble_generalized_stiffness(
        symmetric_layers(),
        parity_warping(),
        quadrature_order=12,
    )

    blocks = decompose_generalized_stiffness(K)

    assert np.linalg.norm(
        blocks.bending_warping
    ) > 0.0


def test_shear_stiffness_is_nonzero():
    K = assemble_generalized_stiffness(
        symmetric_layers(),
        parity_warping(),
        quadrature_order=12,
    )

    blocks = decompose_generalized_stiffness(K)

    assert np.linalg.norm(
        blocks.shear_shear
    ) > 0.0


def test_normal_stiffness_is_nonzero():
    K = assemble_generalized_stiffness(
        symmetric_layers(),
        parity_warping(),
        quadrature_order=12,
    )

    blocks = decompose_generalized_stiffness(K)

    assert np.linalg.norm(
        blocks.normal_normal
    ) > 0.0