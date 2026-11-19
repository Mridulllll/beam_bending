"""
Strong-form equilibrium equations for the five-field HOST HSDT.

They follow from the stationary total potential

    delta Pi = delta U - delta W = 0

with

    delta U = integral_A [
        delta(eps0)^T N
        + delta(kappa)^T M
        + delta(chi)^T P
        + delta(beta)^T Q
    ] dA.

Curvature convention:

    kappa_xx = -w0,xx
    kappa_yy = -w0,yy
    kappa_xy = -2*w0,xy
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class SymbolicFieldEquations:
    x: sp.Symbol
    y: sp.Symbol

    Nxx: sp.Expr
    Nyy: sp.Expr
    Nxy: sp.Expr

    Mxx: sp.Expr
    Myy: sp.Expr
    Mxy: sp.Expr

    Pxx: sp.Expr
    Pyy: sp.Expr
    Pxy: sp.Expr

    Qx: sp.Expr
    Qy: sp.Expr

    q: sp.Expr

    eq_u: sp.Expr
    eq_v: sp.Expr
    eq_w: sp.Expr
    eq_beta_x: sp.Expr
    eq_beta_y: sp.Expr


def build_symbolic_field_equations() -> SymbolicFieldEquations:
    """
    Construct the five HSDT equilibrium residuals.

    A residual equal to zero defines each strong-form equation.
    """

    x, y = sp.symbols(
        "x y",
        real=True,
    )

    Nxx = sp.Function("N_xx")(x, y)
    Nyy = sp.Function("N_yy")(x, y)
    Nxy = sp.Function("N_xy")(x, y)

    Mxx = sp.Function("M_xx")(x, y)
    Myy = sp.Function("M_yy")(x, y)
    Mxy = sp.Function("M_xy")(x, y)

    Pxx = sp.Function("P_xx")(x, y)
    Pyy = sp.Function("P_yy")(x, y)
    Pxy = sp.Function("P_xy")(x, y)

    Qx = sp.Function("Q_x")(x, y)
    Qy = sp.Function("Q_y")(x, y)

    q = sp.Function("q")(x, y)

    # --------------------------------------------------------------
    # Membrane equilibrium
    # --------------------------------------------------------------

    eq_u = (
        sp.diff(Nxx, x)
        + sp.diff(Nxy, y)
    )

    eq_v = (
        sp.diff(Nxy, x)
        + sp.diff(Nyy, y)
    )

    # --------------------------------------------------------------
    # Transverse equilibrium
    #
    # kappa = [-w,xx, -w,yy, -2*w,xy]
    #
    # therefore the internal virtual-work contribution after
    # integration by parts is
    #
    # -(Mxx,xx + 2*Mxy,xy + Myy,yy) delta(w).
    #
    # With external load potential -integral(q*w)dA, stationarity gives
    #
    # Mxx,xx + 2*Mxy,xy + Myy,yy + q = 0.
    # --------------------------------------------------------------

    eq_w = (
        sp.diff(Mxx, x, 2)
        + 2 * sp.diff(Mxy, x, y)
        + sp.diff(Myy, y, 2)
        + q
    )

    # --------------------------------------------------------------
    # Higher-order / transverse-shear equilibrium
    # --------------------------------------------------------------

    eq_beta_x = (
        sp.diff(Pxx, x)
        + sp.diff(Pxy, y)
        - Qx
    )

    eq_beta_y = (
        sp.diff(Pxy, x)
        + sp.diff(Pyy, y)
        - Qy
    )

    return SymbolicFieldEquations(
        x=x,
        y=y,
        Nxx=Nxx,
        Nyy=Nyy,
        Nxy=Nxy,
        Mxx=Mxx,
        Myy=Myy,
        Mxy=Mxy,
        Pxx=Pxx,
        Pyy=Pyy,
        Pxy=Pxy,
        Qx=Qx,
        Qy=Qy,
        q=q,
        eq_u=sp.simplify(eq_u),
        eq_v=sp.simplify(eq_v),
        eq_w=sp.simplify(eq_w),
        eq_beta_x=sp.simplify(eq_beta_x),
        eq_beta_y=sp.simplify(eq_beta_y),
    )