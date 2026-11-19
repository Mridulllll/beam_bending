import numpy as np

from host.kirchhoff import (
    isotropic_bending_rigidity,
    uniform_pressure_coefficient,
    modal_amplitude,
    center_deflection_uniform_pressure,
)


def test_isotropic_bending_rigidity():
    E = 70.0e9
    nu = 0.33
    h = 0.01

    D = isotropic_bending_rigidity(
        E,
        nu,
        h,
    )

    expected = (
        E
        * h**3
        / (
            12.0
            * (1.0 - nu**2)
        )
    )

    np.testing.assert_allclose(
        D,
        expected,
        rtol=1.0e-14,
    )


def test_uniform_load_fourier_parity():
    assert (
        uniform_pressure_coefficient(
            1000.0,
            1,
            1,
        )
        != 0.0
    )

    assert (
        uniform_pressure_coefficient(
            1000.0,
            2,
            1,
        )
        == 0.0
    )


def test_kirchhoff_modal_amplitude_positive():
    D = isotropic_bending_rigidity(
        70.0e9,
        0.33,
        0.01,
    )

    W11 = modal_amplitude(
        q0=1000.0,
        D=D,
        a=1.0,
        b=1.0,
        m=1,
        n=1,
    )

    assert W11 > 0.0


def test_center_deflection_series_converges():
    kwargs = dict(
        q0=1000.0,
        E=70.0e9,
        nu=0.33,
        h=0.01,
        a=1.0,
        b=1.0,
    )

    w15 = center_deflection_uniform_pressure(
        **kwargs,
        max_mode=15,
    )

    w31 = center_deflection_uniform_pressure(
        **kwargs,
        max_mode=31,
    )

    relative_change = abs(
        w31 - w15
    ) / abs(w31)

    assert relative_change < 1.0e-4