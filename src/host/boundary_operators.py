"""
Natural boundary operators for the five-field HOST HSDT.

For an edge with outward unit normal

    n = [nx, ny]^T

the in-plane membrane traction is

    t_N = N n,

and the higher-order traction is

    t_P = P n.

For the transverse displacement w0, the bending contribution produces

    M_nn = n^T M n

and the effective transverse edge force

    V_n = n . div(M) + d(M_ns)/ds,

where

    M_ns = n^T M s

and

    s = [-ny, nx]^T

is the tangent to a straight edge.

A corner contribution associated with M_ns also emerges from the
second integration by parts. It is irrelevant for edges on which
w0 is prescribed, but must be retained for completely free boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from host.field_equations import (
    SymbolicFieldEquations,
    build_symbolic_field_equations,
)


@dataclass(frozen=True)
class SymbolicBoundaryOperators:
    field_equations: SymbolicFieldEquations

    nx: sp.Expr
    ny: sp.Expr

    sx: sp.Expr
    sy: sp.Expr

    membrane_u: sp.Expr
    membrane_v: sp.Expr

    warping_beta_x: sp.Expr
    warping_beta_y: sp.Expr

    M_nn: sp.Expr
    M_ns: sp.Expr
    V_n: sp.Expr


def build_symbolic_boundary_operators(
    nx,
    ny,
) -> SymbolicBoundaryOperators:
    """
    Construct natural boundary operators for a straight edge.

    Parameters
    ----------
    nx, ny
        Components of the outward unit normal.

    Notes
    -----
    The routine assumes nx and ny are constant along the straight edge.
    """

    eq = build_symbolic_field_equations()

    x = eq.x
    y = eq.y

    nx = sp.sympify(nx)
    ny = sp.sympify(ny)

    # Positive 90-degree tangent rotation.
    sx = -ny
    sy = nx

    # --------------------------------------------------------------
    # Membrane traction: N n
    # --------------------------------------------------------------

    membrane_u = (
        eq.Nxx * nx
        + eq.Nxy * ny
    )

    membrane_v = (
        eq.Nxy * nx
        + eq.Nyy * ny
    )

    # --------------------------------------------------------------
    # Higher-order traction: P n
    # --------------------------------------------------------------

    warping_beta_x = (
        eq.Pxx * nx
        + eq.Pxy * ny
    )

    warping_beta_y = (
        eq.Pxy * nx
        + eq.Pyy * ny
    )

    # --------------------------------------------------------------
    # Bending tensor projections
    # --------------------------------------------------------------

    M_nn = (
        nx**2 * eq.Mxx
        + 2 * nx * ny * eq.Mxy
        + ny**2 * eq.Myy
    )

    M_ns = (
        nx * sx * eq.Mxx
        + (nx * sy + ny * sx) * eq.Mxy
        + ny * sy * eq.Myy
    )

    # div(M)
    div_M_x = (
        sp.diff(eq.Mxx, x)
        + sp.diff(eq.Mxy, y)
    )

    div_M_y = (
        sp.diff(eq.Mxy, x)
        + sp.diff(eq.Myy, y)
    )

    n_dot_div_M = (
        nx * div_M_x
        + ny * div_M_y
    )

    # Tangential derivative d()/ds.
    dMns_ds = (
        sx * sp.diff(M_ns, x)
        + sy * sp.diff(M_ns, y)
    )

    V_n = sp.simplify(
        n_dot_div_M
        + dMns_ds
    )

    return SymbolicBoundaryOperators(
        field_equations=eq,
        nx=nx,
        ny=ny,
        sx=sx,
        sy=sy,
        membrane_u=sp.simplify(membrane_u),
        membrane_v=sp.simplify(membrane_v),
        warping_beta_x=sp.simplify(warping_beta_x),
        warping_beta_y=sp.simplify(warping_beta_y),
        M_nn=sp.simplify(M_nn),
        M_ns=sp.simplify(M_ns),
        V_n=V_n,
    )