"""
Verification of directional HOST buckling interaction curves.
"""

import numpy as np

from host.buckling import (
    directional_critical_uniaxial_load_x,
)

from host.buckling_interaction import (
    directional_nx_nxy_interaction,
    directional_nx_ny_interaction,
    directional_proportional_buckling_point,
    nx_nxy_interaction,
    nx_ny_interaction,
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
    solve_directional_multimode_buckling,
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


def test_scalar_limit_reproduces_nx_nxy_interaction():
    K11, K12 = scalar_and_directional_plate()

    angles = [
        -90.0,
        -45.0,
        0.0,
        30.0,
        60.0,
        90.0,
    ]

    scalar = nx_nxy_interaction(
        K11,
        a=100.0,
        b=80.0,
        angles_deg=angles,
        max_m=6,
        max_n=6,
    )

    directional = directional_nx_nxy_interaction(
        K12,
        a=100.0,
        b=80.0,
        angles_deg=angles,
        max_m=6,
        max_n=6,
    )

    for s, d in zip(
        scalar,
        directional,
    ):
        np.testing.assert_allclose(
            d.load_factor,
            s.load_factor,
            rtol=1.0e-10,
        )

        np.testing.assert_allclose(
            [
                d.Nx_cr,
                d.Ny_cr,
                d.Nxy_cr,
            ],
            [
                s.Nx_cr,
                s.Ny_cr,
                s.Nxy_cr,
            ],
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_scalar_limit_reproduces_nx_ny_interaction():
    K11, K12 = scalar_and_directional_plate()

    angles = [
        0.0,
        30.0,
        45.0,
        60.0,
        90.0,
    ]

    scalar = nx_ny_interaction(
        K11,
        a=100.0,
        b=80.0,
        angles_deg=angles,
        max_m=6,
        max_n=6,
    )

    directional = directional_nx_ny_interaction(
        K12,
        a=100.0,
        b=80.0,
        angles_deg=angles,
        max_m=6,
        max_n=6,
    )

    for s, d in zip(
        scalar,
        directional,
    ):
        np.testing.assert_allclose(
            d.load_factor,
            s.load_factor,
            rtol=1.0e-10,
        )

        np.testing.assert_allclose(
            [
                d.Nx_cr,
                d.Ny_cr,
                d.Nxy_cr,
            ],
            [
                s.Nx_cr,
                s.Ny_cr,
                s.Nxy_cr,
            ],
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_directional_nx_endpoint_matches_single_mode_solution():
    _, K12 = scalar_and_directional_plate()

    point = directional_nx_nxy_interaction(
        K12,
        a=100.0,
        b=80.0,
        angles_deg=[
            0.0,
        ],
        max_m=6,
        max_n=6,
    )[0]

    reference = directional_critical_uniaxial_load_x(
        K12,
        a=100.0,
        b=80.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        point.Nx_cr,
        reference.load_factor,
        rtol=1.0e-10,
    )


def test_directional_pure_shear_endpoint_matches_direct_solution():
    _, K12 = scalar_and_directional_plate()

    point = directional_nx_nxy_interaction(
        K12,
        a=100.0,
        b=80.0,
        angles_deg=[
            90.0,
        ],
        max_m=6,
        max_n=6,
    )[0]

    reference = solve_directional_multimode_buckling(
        K12,
        a=100.0,
        b=80.0,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        point.Nxy_cr,
        reference.load_factor,
        rtol=1.0e-10,
    )


def test_directional_resultants_follow_reference_direction():
    _, K12 = scalar_and_directional_plate()

    point = directional_proportional_buckling_point(
        K12,
        a=100.0,
        b=80.0,
        Nx_ref=0.6,
        Ny_ref=0.2,
        Nxy_ref=0.4,
        parameter=1.0,
        max_m=6,
        max_n=6,
    )

    np.testing.assert_allclose(
        point.Nx_cr,
        point.load_factor
        * point.Nx_ref,
        rtol=1.0e-13,
    )

    np.testing.assert_allclose(
        point.Ny_cr,
        point.load_factor
        * point.Ny_ref,
        rtol=1.0e-13,
    )

    np.testing.assert_allclose(
        point.Nxy_cr,
        point.load_factor
        * point.Nxy_ref,
        rtol=1.0e-13,
    )