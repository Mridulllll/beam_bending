import sympy as sp

from host.boundary_operators import (
    build_symbolic_boundary_operators,
)


def test_right_edge_membrane_tractions():
    bc = build_symbolic_boundary_operators(
        1,
        0,
    )

    eq = bc.field_equations

    assert sp.simplify(
        bc.membrane_u - eq.Nxx
    ) == 0

    assert sp.simplify(
        bc.membrane_v - eq.Nxy
    ) == 0


def test_top_edge_membrane_tractions():
    bc = build_symbolic_boundary_operators(
        0,
        1,
    )

    eq = bc.field_equations

    assert sp.simplify(
        bc.membrane_u - eq.Nxy
    ) == 0

    assert sp.simplify(
        bc.membrane_v - eq.Nyy
    ) == 0


def test_right_edge_higher_order_tractions():
    bc = build_symbolic_boundary_operators(
        1,
        0,
    )

    eq = bc.field_equations

    assert sp.simplify(
        bc.warping_beta_x - eq.Pxx
    ) == 0

    assert sp.simplify(
        bc.warping_beta_y - eq.Pxy
    ) == 0


def test_top_edge_higher_order_tractions():
    bc = build_symbolic_boundary_operators(
        0,
        1,
    )

    eq = bc.field_equations

    assert sp.simplify(
        bc.warping_beta_x - eq.Pxy
    ) == 0

    assert sp.simplify(
        bc.warping_beta_y - eq.Pyy
    ) == 0


def test_right_edge_bending_operators():
    bc = build_symbolic_boundary_operators(
        1,
        0,
    )

    eq = bc.field_equations

    expected_V = (
        sp.diff(eq.Mxx, eq.x)
        + 2 * sp.diff(eq.Mxy, eq.y)
    )

    assert sp.simplify(
        bc.M_nn - eq.Mxx
    ) == 0

    assert sp.simplify(
        bc.M_ns - eq.Mxy
    ) == 0

    assert sp.simplify(
        bc.V_n - expected_V
    ) == 0


def test_top_edge_bending_operators():
    bc = build_symbolic_boundary_operators(
        0,
        1,
    )

    eq = bc.field_equations

    expected_V = (
        sp.diff(eq.Myy, eq.y)
        + 2 * sp.diff(eq.Mxy, eq.x)
    )

    assert sp.simplify(
        bc.M_nn - eq.Myy
    ) == 0

    # With s=(-1,0), M_ns = -M_xy on the top edge.
    assert sp.simplify(
        bc.M_ns + eq.Mxy
    ) == 0

    assert sp.simplify(
        bc.V_n - expected_V
    ) == 0


def test_opposite_edge_tractions_change_sign():
    right = build_symbolic_boundary_operators(
        1,
        0,
    )

    left = build_symbolic_boundary_operators(
        -1,
        0,
    )

    assert sp.simplify(
        right.membrane_u
        + left.membrane_u
    ) == 0

    assert sp.simplify(
        right.membrane_v
        + left.membrane_v
    ) == 0

    assert sp.simplify(
        right.warping_beta_x
        + left.warping_beta_x
    ) == 0

    assert sp.simplify(
        right.warping_beta_y
        + left.warping_beta_y
    ) == 0