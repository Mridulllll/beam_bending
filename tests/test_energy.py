import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.energy import (
    modal_energy_decomposition,
    reconstructed_total_energy,
)

from host.navier import (
    navier_modal_load,
    solve_navier_mode_basis_aware,
)

from host.sandwich import (
    build_three_layer_sandwich,
)


def reference_problem():
    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=1.0e9,
        nu=0.25,
    )

    sandwich = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=C_core,
        C_top=C_face,
        hf_bottom=0.5e-3,
        hc=10.0e-3,
        hf_top=0.5e-3,
    )

    return sandwich


def test_energy_decomposition_reconstructs_total():
    sandwich = reference_problem()

    a = 0.1
    b = 0.1

    q = solve_navier_mode_basis_aware(
        sandwich.K11,
        q0=10.0e3,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    energy = modal_energy_decomposition(
        sandwich.K11,
        q,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    reconstructed = (
        reconstructed_total_energy(
            energy
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        energy.total,
        rtol=1.0e-11,
        atol=1.0e-14,
    )


def test_equilibrium_energy_equals_half_external_work():
    sandwich = reference_problem()

    a = 0.1
    b = 0.1

    q_modal = solve_navier_mode_basis_aware(
        sandwich.K11,
        q0=10.0e3,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    energy = modal_energy_decomposition(
        sandwich.K11,
        q_modal,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    f_modal = navier_modal_load(
        q0=10.0e3,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    expected = (
        0.5
        * float(
            q_modal @ f_modal
        )
    )

    np.testing.assert_allclose(
        energy.total,
        expected,
        rtol=1.0e-10,
        atol=1.0e-14,
    )


def test_energy_self_terms_are_nonnegative():
    sandwich = reference_problem()

    q = solve_navier_mode_basis_aware(
        sandwich.K11,
        q0=10.0e3,
        a=0.1,
        b=0.1,
        m=1,
        n=1,
    )

    energy = modal_energy_decomposition(
        sandwich.K11,
        q,
        a=0.1,
        b=0.1,
        m=1,
        n=1,
    )

    for value in (
        energy.self_terms.values()
    ):
        assert value >= -1.0e-14