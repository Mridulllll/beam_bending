import numpy as np

from host.condensation import (
    condense_higher_order_modal_variables,
    condense_to_transverse_stiffness,
    higher_order_softening_fraction,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.navier import (
    navier_modal_stiffness_basis_aware,
    navier_modal_load,
    solve_navier_mode_basis_aware,
)

from host.sandwich import (
    build_three_layer_sandwich,
)


def reference_modal_stiffness():
    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=0.7e9,
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

    return navier_modal_stiffness_basis_aware(
        sandwich.K11,
        a=0.1,
        b=0.1,
        m=1,
        n=1,
    )


def test_condensed_matrix_shape_and_symmetry():
    K = reference_modal_stiffness()

    result = (
        condense_higher_order_modal_variables(
            K
        )
    )

    assert result.K_effective.shape == (
        3,
        3,
    )

    np.testing.assert_allclose(
        result.K_effective,
        result.K_effective.T,
        rtol=1.0e-12,
        atol=1.0e-10,
    )


def test_relaxation_is_positive_semidefinite():
    K = reference_modal_stiffness()

    result = (
        condense_higher_order_modal_variables(
            K
        )
    )

    eigvals = np.linalg.eigvalsh(
        result.relaxation
    )

    assert np.min(
        eigvals
    ) >= -1.0e-10 * max(
        1.0,
        np.max(
            np.abs(eigvals)
        ),
    )


def test_condensation_reproduces_full_solution_W():
    C_face = isotropic_stiffness_3d(
        E=70.0e9,
        nu=0.33,
    )

    C_core = isotropic_stiffness_3d(
        E=0.7e9,
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

    a = 0.1
    b = 0.1
    q0 = 10.0e3

    K = navier_modal_stiffness_basis_aware(
        sandwich.K11,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    full = solve_navier_mode_basis_aware(
        sandwich.K11,
        q0=q0,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    f = navier_modal_load(
        q0=q0,
        a=a,
        b=b,
        m=1,
        n=1,
    )

    k_w = condense_to_transverse_stiffness(
        K
    )

    W_condensed = (
        f[2]
        / k_w
    )

    np.testing.assert_allclose(
        W_condensed,
        full[2],
        rtol=1.0e-11,
        atol=1.0e-14,
    )


def test_higher_order_relaxation_softens_transverse_response():
    K = reference_modal_stiffness()

    fraction = (
        higher_order_softening_fraction(
            K
        )
    )

    assert fraction >= 0.0
    assert fraction < 1.0


def test_effective_transverse_stiffness_is_positive():
    K = reference_modal_stiffness()

    k_w = condense_to_transverse_stiffness(
        K
    )

    assert k_w > 0.0