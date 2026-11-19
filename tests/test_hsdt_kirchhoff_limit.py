import numpy as np

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
    center_deflection_uniform_pressure,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def build_hsdt_plate_stiffness(
    E: float,
    nu: float,
    h: float,
) -> np.ndarray:
    """
    Build the five-field HSDT stiffness for a homogeneous isotropic plate.

    The five-field baseline imposes sigma_zz = 0 while retaining
    transverse shear stiffness.
    """

    C3 = isotropic_stiffness_3d(
        E=E,
        nu=nu,
    )

    C_hsdt = sigma_zz_zero_stiffness(
        C3
    )

    layers = [
        Layer(
            name="homogeneous_plate",
            z_bottom=-0.5 * h,
            z_top=+0.5 * h,
            stiffness=C_hsdt,
        )
    ]

    K14 = assemble_generalized_stiffness(
        layers,
        reddy_baseline_generalized_warping(h),
        quadrature_order=10,
    )

    return extract_hsdt_stiffness(
        K14
    )


def relative_error(
    value: float,
    reference: float,
) -> float:
    return (
        float(value) - float(reference)
    ) / float(reference)


def test_hsdt_center_deflection_converges_to_kirchhoff():
    """
    The five-field HSDT must approach the Kirchhoff-Love solution as
    h/a -> 0.

    This is an asymptotic verification test, not merely a numerical
    consistency test.
    """

    E = 70.0e9
    nu = 0.33

    a = 1.0
    b = 1.0

    q0 = 1000.0

    thickness_ratios = np.array(
        [
            1.0e-1,
            5.0e-2,
            1.0e-2,
            5.0e-3,
            1.0e-3,
        ],
        dtype=float,
    )

    errors = []

    for h_over_a in thickness_ratios:
        h = (
            h_over_a
            * a
        )

        K_hsdt = build_hsdt_plate_stiffness(
            E=E,
            nu=nu,
            h=h,
        )

        w_hsdt = (
            center_deflection_uniform_pressure_hsdt(
                K_hsdt=K_hsdt,
                q0=q0,
                a=a,
                b=b,
                max_mode=31,
            )
        )

        w_kl = (
            center_deflection_uniform_pressure(
                q0=q0,
                E=E,
                nu=nu,
                h=h,
                a=a,
                b=b,
                max_mode=31,
            )
        )

        err = abs(
            relative_error(
                w_hsdt,
                w_kl,
            )
        )

        errors.append(err)

    errors = np.asarray(
        errors,
        dtype=float,
    )

    # The thin-plate limit must improve strongly.
    assert errors[-1] < errors[0]

    # The final thin configuration should be very close to KL.
    assert errors[-1] < 5.0e-3


def test_hsdt_kirchhoff_error_decreases_with_slenderness():
    """
    Check the general convergence trend for the thinner configurations.
    """

    E = 70.0e9
    nu = 0.33

    a = 1.0
    b = 1.0

    q0 = 1000.0

    thickness_ratios = [
        1.0e-2,
        5.0e-3,
        1.0e-3,
    ]

    errors = []

    for h_over_a in thickness_ratios:
        h = (
            h_over_a
            * a
        )

        K_hsdt = build_hsdt_plate_stiffness(
            E,
            nu,
            h,
        )

        w_hsdt = (
            center_deflection_uniform_pressure_hsdt(
                K_hsdt,
                q0,
                a,
                b,
                max_mode=31,
            )
        )

        w_kl = (
            center_deflection_uniform_pressure(
                q0,
                E,
                nu,
                h,
                a,
                b,
                max_mode=31,
            )
        )

        errors.append(
            abs(
                relative_error(
                    w_hsdt,
                    w_kl,
                )
            )
        )

    # Allow tiny floating-point wiggles while requiring the expected trend.
    assert errors[1] <= errors[0] * 1.05
    assert errors[2] <= errors[1] * 1.05


def test_very_thin_plate_center_deflection_agrees_with_kirchhoff():
    """
    Direct numerical comparison at h/a = 1e-3.
    """

    E = 70.0e9
    nu = 0.33

    a = 1.0
    b = 1.0
    h = 1.0e-3

    q0 = 1000.0

    K_hsdt = build_hsdt_plate_stiffness(
        E,
        nu,
        h,
    )

    w_hsdt = (
        center_deflection_uniform_pressure_hsdt(
            K_hsdt,
            q0,
            a,
            b,
            max_mode=31,
        )
    )

    w_kl = (
        center_deflection_uniform_pressure(
            q0,
            E,
            nu,
            h,
            a,
            b,
            max_mode=31,
        )
    )

    rel = abs(
        relative_error(
            w_hsdt,
            w_kl,
        )
    )

    assert rel < 5.0e-3