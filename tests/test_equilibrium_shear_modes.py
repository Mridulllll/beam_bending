import numpy as np

from host.constitutive import isotropic_stiffness_3d
from host.equilibrium_shear_modes import (
    derive_equilibrium_shear_mode,
)
from host.variational_shear_modes import (
    BeamLayer2D,
    free_stress_beam_stiffness,
)


def isotropic_beam_C(E, nu):
    return free_stress_beam_stiffness(
        isotropic_stiffness_3d(
            E=E,
            nu=nu,
        )
    )


def test_homogeneous_rectangle_recovers_five_sixths_shear_factor():
    E = 70000.0
    nu = 0.33
    G = E / (2.0 * (1.0 + nu))

    width = 30.0
    h = 12.0

    Cb = isotropic_beam_C(
        E,
        nu,
    )

    mode = derive_equilibrium_shear_mode(
        (
            BeamLayer2D(
                "solid",
                -0.5 * h,
                +0.5 * h,
                Cb,
            ),
        ),
        width=width,
    )

    np.testing.assert_allclose(
        mode.shear_correction_factor,
        5.0 / 6.0,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    np.testing.assert_allclose(
        mode.equivalent_shear_stiffness,
        (5.0 / 6.0)
        * G
        * width
        * h,
        rtol=2.0e-13,
        atol=1.0e-8,
    )


def test_homogeneous_rectangle_has_parabolic_shear_derivative():
    E = 70000.0
    nu = 0.33
    width = 20.0
    h = 10.0

    Cb = isotropic_beam_C(
        E,
        nu,
    )

    mode = derive_equilibrium_shear_mode(
        (
            BeamLayer2D(
                "solid",
                -0.5 * h,
                +0.5 * h,
                Cb,
            ),
        ),
        width=width,
    )

    for z in np.linspace(
        -0.5 * h,
        +0.5 * h,
        21,
    ):
        expected = (
            1.0
            - 4.0 * z * z / (h * h)
        )

        np.testing.assert_allclose(
            mode.df(z),
            expected,
            rtol=2.0e-12,
            atol=2.0e-12,
        )


def test_equilibrium_mode_satisfies_unit_resultant_and_free_surfaces():
    h = 12.0
    hc = 10.0
    width = 30.0

    Cface = isotropic_beam_C(
        70000.0,
        0.33,
    )

    Ccore = isotropic_beam_C(
        3500.0,
        0.30,
    )

    layers = (
        BeamLayer2D(
            "face_bottom",
            -0.5 * h,
            -0.5 * hc,
            Cface,
        ),
        BeamLayer2D(
            "core",
            -0.5 * hc,
            +0.5 * hc,
            Ccore,
        ),
        BeamLayer2D(
            "face_top",
            +0.5 * hc,
            +0.5 * h,
            Cface,
        ),
    )

    mode = derive_equilibrium_shear_mode(
        layers,
        width=width,
    )

    np.testing.assert_allclose(
        mode.unit_shear_resultant_check,
        1.0,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    np.testing.assert_allclose(
        mode.tau_per_unit_shear(
            -0.5 * h
        ),
        0.0,
        atol=1.0e-15,
    )

    np.testing.assert_allclose(
        mode.tau_per_unit_shear(
            +0.5 * h
        ),
        0.0,
        atol=1.0e-15,
    )


def test_symmetric_sandwich_mode_is_odd_with_even_derivative():
    h = 12.0
    hc = 10.0
    width = 30.0

    Cface = isotropic_beam_C(
        70000.0,
        0.33,
    )

    Ccore = isotropic_beam_C(
        3500.0,
        0.30,
    )

    mode = derive_equilibrium_shear_mode(
        (
            BeamLayer2D(
                "face_bottom",
                -0.5 * h,
                -0.5 * hc,
                Cface,
            ),
            BeamLayer2D(
                "core",
                -0.5 * hc,
                +0.5 * hc,
                Ccore,
            ),
            BeamLayer2D(
                "face_top",
                +0.5 * hc,
                +0.5 * h,
                Cface,
            ),
        ),
        width=width,
    )

    np.testing.assert_allclose(
        mode.neutral_axis,
        0.0,
        atol=1.0e-14,
    )

    interface = 0.5 * hc

    for z in np.linspace(
        0.0,
        0.5 * h,
        19,
    ):
        np.testing.assert_allclose(
            mode.f(-z),
            -mode.f(z),
            atol=2.0e-12,
        )

        # f' is even wherever it is single-valued.  At a material interface
        # gamma_sz=f' has two legitimate one-sided traces because G jumps
        # while tau_sz remains continuous, so comparing the exact interface
        # value from opposite layer-ownership conventions is not meaningful.
        if abs(z - interface) > 1.0e-12:
            np.testing.assert_allclose(
                mode.df(-z),
                mode.df(z),
                atol=2.0e-12,
            )


def test_symmetric_sandwich_one_sided_derivative_traces_are_even():
    h = 12.0
    hc = 10.0
    width = 30.0

    Cface = isotropic_beam_C(
        70000.0,
        0.33,
    )

    Ccore = isotropic_beam_C(
        3500.0,
        0.30,
    )

    mode = derive_equilibrium_shear_mode(
        (
            BeamLayer2D(
                "face_bottom",
                -0.5 * h,
                -0.5 * hc,
                Cface,
            ),
            BeamLayer2D(
                "core",
                -0.5 * hc,
                +0.5 * hc,
                Ccore,
            ),
            BeamLayer2D(
                "face_top",
                +0.5 * hc,
                +0.5 * h,
                Cface,
            ),
        ),
        width=width,
    )

    interface = 0.5 * hc
    eps = 1.0e-8

    for z in (
        interface - eps,
        interface + eps,
    ):
        np.testing.assert_allclose(
            mode.df(-z),
            mode.df(z),
            rtol=2.0e-9,
            atol=2.0e-12,
        )


def test_symmetric_sandwich_preserves_interface_shear_traction():
    h = 12.0
    hc = 10.0
    width = 30.0

    Cface = isotropic_beam_C(
        70000.0,
        0.33,
    )

    Ccore = isotropic_beam_C(
        3500.0,
        0.30,
    )

    mode = derive_equilibrium_shear_mode(
        (
            BeamLayer2D(
                "face_bottom",
                -0.5 * h,
                -0.5 * hc,
                Cface,
            ),
            BeamLayer2D(
                "core",
                -0.5 * hc,
                +0.5 * hc,
                Ccore,
            ),
            BeamLayer2D(
                "face_top",
                +0.5 * hc,
                +0.5 * h,
                Cface,
            ),
        ),
        width=width,
    )

    zi = 0.5 * hc
    eps = 1.0e-9

    tau_core = mode.tau_per_unit_shear(
        zi - eps
    )

    tau_face = mode.tau_per_unit_shear(
        zi + eps
    )

    np.testing.assert_allclose(
        tau_core,
        tau_face,
        rtol=2.0e-9,
        atol=1.0e-14,
    )

    # The strain jump follows the inverse shear modulus ratio.
    gamma_core = mode.gamma_per_unit_shear(
        zi - eps
    )

    gamma_face = mode.gamma_per_unit_shear(
        zi + eps
    )

    np.testing.assert_allclose(
        gamma_core / gamma_face,
        Cface[1, 1] / Ccore[1, 1],
        rtol=2.0e-9,
        atol=1.0e-12,
    )
