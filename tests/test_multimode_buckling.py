import numpy as np

from host.buckling import (
    critical_uniaxial_load_x,
)

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
)

from host.generalized_stiffness import (
    Layer,
    assemble_generalized_stiffness,
)

from host.hsdt import (
    extract_hsdt_stiffness,
)

from host.multimode_buckling import (
    NavierBucklingMode,
    assemble_multimode_buckling_matrices,
    geometric_coupling_entry,
    solve_multimode_buckling,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def homogeneous_plate():
    E = 70000.0
    nu = 0.33
    h = 1.0

    C = sigma_zz_zero_stiffness(
        isotropic_stiffness_3d(
            E=E,
            nu=nu,
        )
    )

    K14 = assemble_generalized_stiffness(
        [
            Layer(
                "plate",
                -0.5 * h,
                +0.5 * h,
                C,
            )
        ],
        reddy_baseline_generalized_warping(
            h
        ),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(
        K14
    )


def test_geometric_matrix_is_symmetric():
    K = homogeneous_plate()

    _, KG, _ = (
        assemble_multimode_buckling_matrices(
            K,
            a=100.0,
            b=80.0,
            Nx_ref=1.0,
            Ny_ref=0.4,
            Nxy_ref=0.25,
            max_m=4,
            max_n=4,
        )
    )

    np.testing.assert_allclose(
        KG,
        KG.T,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_normal_prestress_does_not_couple_different_modes():
    mode_11 = NavierBucklingMode(
        m=1,
        n=1,
    )

    mode_21 = NavierBucklingMode(
        m=2,
        n=1,
    )

    value = geometric_coupling_entry(
        mode_11,
        mode_21,
        a=100.0,
        b=100.0,
        Nx_ref=1.0,
        Ny_ref=0.5,
        Nxy_ref=0.0,
    )

    np.testing.assert_allclose(
        value,
        0.0,
        atol=1.0e-14,
    )


def test_shear_prestress_couples_navier_modes():
    mode_11 = NavierBucklingMode(
        m=1,
        n=1,
    )

    mode_22 = NavierBucklingMode(
        m=2,
        n=2,
    )

    value = geometric_coupling_entry(
        mode_11,
        mode_22,
        a=100.0,
        b=100.0,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
    )

    assert abs(value) > 0.0


def test_multimode_uniaxial_matches_existing_single_mode_search():
    K = homogeneous_plate()

    reference = critical_uniaxial_load_x(
        K,
        a=100.0,
        b=100.0,
        max_m=5,
        max_n=5,
    )

    multimode = solve_multimode_buckling(
        K,
        a=100.0,
        b=100.0,
        Nx_ref=1.0,
        Ny_ref=0.0,
        Nxy_ref=0.0,
        max_m=5,
        max_n=5,
    )

    np.testing.assert_allclose(
        multimode.load_factor,
        reference.load_factor,
        rtol=1.0e-10,
    )


def test_pure_shear_buckling_has_positive_solution():
    K = homogeneous_plate()

    result = solve_multimode_buckling(
        K,
        a=100.0,
        b=100.0,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
        max_m=5,
        max_n=5,
    )

    assert np.isfinite(
        result.load_factor
    )

    assert result.load_factor > 0.0


def test_pure_shear_requires_multimode_coupling():
    K = homogeneous_plate()

    _, KG, _ = (
        assemble_multimode_buckling_matrices(
            K,
            a=100.0,
            b=100.0,
            Nx_ref=0.0,
            Ny_ref=0.0,
            Nxy_ref=1.0,
            max_m=4,
            max_n=4,
        )
    )

    off_diagonal = (
        KG
        - np.diag(
            np.diag(KG)
        )
    )

    assert np.linalg.norm(
        off_diagonal
    ) > 0.0