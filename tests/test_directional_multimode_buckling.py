"""
Verification of directional multimode HOST buckling.
"""

import numpy as np

from host.buckling import (
    directional_critical_uniaxial_load_x,
)

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
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

from host.multimode_buckling import (
    assemble_directional_multimode_buckling_matrices,
    assemble_multimode_buckling_matrices,
    solve_directional_multimode_buckling,
    solve_multimode_buckling,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def scalar_and_directional_plate():
    E = 70000.0
    nu = 0.33
    h = 1.0

    C = sigma_zz_zero_stiffness(
        isotropic_stiffness_3d(
            E=E,
            nu=nu,
        )
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
        reddy_baseline_generalized_warping(
            h
        )
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

    return (
        extract_hsdt_stiffness(
            K14
        ),
        extract_directional_hsdt_stiffness(
            K15
        ),
    )


def test_scalar_limit_reproduces_multimode_matrices():
    """
    Strong scalar-limit regression including shear-induced
    geometric mode coupling.
    """

    K11, K12 = (
        scalar_and_directional_plate()
    )

    KE_scalar, KG_scalar, modes_scalar = (
        assemble_multimode_buckling_matrices(
            K11,
            a=100.0,
            b=80.0,
            Nx_ref=0.7,
            Ny_ref=0.2,
            Nxy_ref=0.35,
            max_m=4,
            max_n=4,
        )
    )

    KE_dir, KG_dir, modes_dir = (
        assemble_directional_multimode_buckling_matrices(
            K12,
            a=100.0,
            b=80.0,
            Nx_ref=0.7,
            Ny_ref=0.2,
            Nxy_ref=0.35,
            max_m=4,
            max_n=4,
        )
    )

    assert modes_dir == modes_scalar

    np.testing.assert_allclose(
        KE_dir,
        KE_scalar,
        rtol=1.0e-11,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        KG_dir,
        KG_scalar,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_scalar_limit_reproduces_pure_shear_buckling():
    K11, K12 = (
        scalar_and_directional_plate()
    )

    scalar = solve_multimode_buckling(
        K11,
        a=100.0,
        b=80.0,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
        max_m=5,
        max_n=5,
    )

    directional = (
        solve_directional_multimode_buckling(
            K12,
            a=100.0,
            b=80.0,
            Nx_ref=0.0,
            Ny_ref=0.0,
            Nxy_ref=1.0,
            max_m=5,
            max_n=5,
        )
    )

    np.testing.assert_allclose(
        directional.load_factor,
        scalar.load_factor,
        rtol=1.0e-10,
    )


def test_directional_pure_shear_solution_is_positive():
    _, K12 = (
        scalar_and_directional_plate()
    )

    result = (
        solve_directional_multimode_buckling(
            K12,
            a=100.0,
            b=80.0,
            Nx_ref=0.0,
            Ny_ref=0.0,
            Nxy_ref=1.0,
            max_m=5,
            max_n=5,
        )
    )

    assert np.isfinite(
        result.load_factor
    )

    assert result.load_factor > 0.0


def test_directional_uniaxial_multimode_matches_single_mode_search():
    _, K12 = (
        scalar_and_directional_plate()
    )

    single = (
        directional_critical_uniaxial_load_x(
            K12,
            a=100.0,
            b=80.0,
            max_m=5,
            max_n=5,
        )
    )

    multimode = (
        solve_directional_multimode_buckling(
            K12,
            a=100.0,
            b=80.0,
            Nx_ref=1.0,
            Ny_ref=0.0,
            Nxy_ref=0.0,
            max_m=5,
            max_n=5,
        )
    )

    np.testing.assert_allclose(
        multimode.load_factor,
        single.load_factor,
        rtol=1.0e-10,
    )


def test_directional_geometric_matrix_is_symmetric():
    _, K12 = (
        scalar_and_directional_plate()
    )

    _, KG, _ = (
        assemble_directional_multimode_buckling_matrices(
            K12,
            a=100.0,
            b=80.0,
            Nx_ref=0.3,
            Ny_ref=0.4,
            Nxy_ref=0.5,
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