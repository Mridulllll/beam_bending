import numpy as np

from host.buckling import (
    critical_load_factor_mode,
    critical_uniaxial_load_x,
    geometric_modal_coefficient,
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

from host.kirchhoff import (
    isotropic_bending_rigidity,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def homogeneous_hsdt_plate(
    E,
    nu,
    h,
):
    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    C5 = sigma_zz_zero_stiffness(
        C3
    )

    layers = [
        Layer(
            "plate",
            -0.5 * h,
            +0.5 * h,
            C5,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(
            h
        ),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(
        K14
    )


def test_geometric_modal_coefficient_positive():
    value = geometric_modal_coefficient(
        a=100.0,
        b=100.0,
        m=1,
        n=1,
        Nx_ref=1.0,
    )

    assert value > 0.0


def test_buckling_mode_result_is_positive():
    K = homogeneous_hsdt_plate(
        E=70000.0,
        nu=0.33,
        h=1.0,
    )

    result = critical_load_factor_mode(
        K,
        a=100.0,
        b=100.0,
        m=1,
        n=1,
        Nx_ref=1.0,
    )

    assert result.load_factor > 0.0


def test_thin_hsdt_buckling_converges_to_kirchhoff():
    E = 70000.0
    nu = 0.33

    a = 100.0
    b = 100.0

    h = 0.1

    K = homogeneous_hsdt_plate(
        E=E,
        nu=nu,
        h=h,
    )

    result = critical_load_factor_mode(
        K,
        a=a,
        b=b,
        m=1,
        n=1,
        Nx_ref=1.0,
    )

    D = isotropic_bending_rigidity(
        E,
        nu,
        h,
    )

    alpha = np.pi / a
    beta = np.pi / b

    N_kl = (
        D
        * (
            alpha**2
            + beta**2
        )**2
        / alpha**2
    )

    relative_error = abs(
        result.load_factor - N_kl
    ) / N_kl

    assert relative_error < 5.0e-3


def test_square_plate_search_finds_11_mode():
    K = homogeneous_hsdt_plate(
        E=70000.0,
        nu=0.33,
        h=0.1,
    )

    result = critical_uniaxial_load_x(
        K,
        a=100.0,
        b=100.0,
        max_m=6,
        max_n=6,
    )

    assert result.m == 1
    assert result.n == 1


def test_critical_load_scales_with_thickness_cubed_in_thin_limit():
    E = 70000.0
    nu = 0.33

    a = 100.0
    b = 100.0

    h1 = 0.05
    h2 = 0.10

    K1 = homogeneous_hsdt_plate(
        E,
        nu,
        h1,
    )

    K2 = homogeneous_hsdt_plate(
        E,
        nu,
        h2,
    )

    N1 = critical_uniaxial_load_x(
        K1,
        a,
        b,
        max_m=4,
        max_n=4,
    ).load_factor

    N2 = critical_uniaxial_load_x(
        K2,
        a,
        b,
        max_m=4,
        max_n=4,
    ).load_factor

    ratio = N2 / N1

    np.testing.assert_allclose(
        ratio,
        (h2 / h1)**3,
        rtol=5.0e-3,
    )