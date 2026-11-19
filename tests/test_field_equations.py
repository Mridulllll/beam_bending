import sympy as sp

from host.field_equations import (
    build_symbolic_field_equations,
)


def test_membrane_equilibrium_operators():
    eq = build_symbolic_field_equations()

    expected_u = (
        sp.diff(eq.Nxx, eq.x)
        + sp.diff(eq.Nxy, eq.y)
    )

    expected_v = (
        sp.diff(eq.Nxy, eq.x)
        + sp.diff(eq.Nyy, eq.y)
    )

    assert sp.simplify(
        eq.eq_u - expected_u
    ) == 0

    assert sp.simplify(
        eq.eq_v - expected_v
    ) == 0


def test_transverse_equilibrium_operator():
    eq = build_symbolic_field_equations()

    expected = (
        sp.diff(eq.Mxx, eq.x, 2)
        + 2 * sp.diff(
            eq.Mxy,
            eq.x,
            eq.y,
        )
        + sp.diff(eq.Myy, eq.y, 2)
        + eq.q
    )

    assert sp.simplify(
        eq.eq_w - expected
    ) == 0


def test_beta_x_equilibrium_operator():
    eq = build_symbolic_field_equations()

    expected = (
        sp.diff(eq.Pxx, eq.x)
        + sp.diff(eq.Pxy, eq.y)
        - eq.Qx
    )

    assert sp.simplify(
        eq.eq_beta_x - expected
    ) == 0


def test_beta_y_equilibrium_operator():
    eq = build_symbolic_field_equations()

    expected = (
        sp.diff(eq.Pxy, eq.x)
        + sp.diff(eq.Pyy, eq.y)
        - eq.Qy
    )

    assert sp.simplify(
        eq.eq_beta_y - expected
    ) == 0