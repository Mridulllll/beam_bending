"""
Directional higher-order kinematics for HOST sandwich theory.

Displacement field
------------------
u(x,y,z) = u0(x,y) - z*w0,x(x,y) + fx(z)*beta_x(x,y)
v(x,y,z) = v0(x,y) - z*w0,y(x,y) + fy(z)*beta_y(x,y)
w(x,y,z) = w0(x,y) + g(z)*eta(x,y)

The classical scalar HSDT formulation is recovered exactly for

    fx(z) = fy(z) = f(z).
"""

from dataclasses import dataclass
from typing import Tuple

import sympy as sp


@dataclass(frozen=True)
class SymbolicKinematics:
    x: sp.Symbol
    y: sp.Symbol
    z: sp.Symbol

    u0: sp.Expr
    v0: sp.Expr
    w0: sp.Expr
    beta_x: sp.Expr
    beta_y: sp.Expr
    eta: sp.Expr

    fx: sp.Expr
    fy: sp.Expr
    g: sp.Expr

    u: sp.Expr
    v: sp.Expr
    w: sp.Expr

    strain: sp.Matrix

    @property
    def f(self) -> sp.Expr:
        """
        Backward-compatible scalar warping alias.

        This property is meaningful only for code that assumes the
        scalar limit fx = fy. New directional code should use fx/fy.
        """
        return self.fx


def build_symbolic_kinematics() -> SymbolicKinematics:
    """
    Construct the symbolic displacement and engineering-strain fields.

    Strain ordering:

        [eps_xx,
         eps_yy,
         eps_zz,
         gamma_xy,
         gamma_xz,
         gamma_yz]^T
    """

    x, y, z = sp.symbols("x y z", real=True)

    u0 = sp.Function("u_0")(x, y)
    v0 = sp.Function("v_0")(x, y)
    w0 = sp.Function("w_0")(x, y)

    beta_x = sp.Function("beta_x")(x, y)
    beta_y = sp.Function("beta_y")(x, y)
    eta = sp.Function("eta")(x, y)

    fx = sp.Function("f_x")(z)
    fy = sp.Function("f_y")(z)
    g = sp.Function("g")(z)

    u = (
        u0
        - z * sp.diff(w0, x)
        + fx * beta_x
    )

    v = (
        v0
        - z * sp.diff(w0, y)
        + fy * beta_y
    )

    w = w0 + g * eta

    eps_xx = sp.diff(u, x)
    eps_yy = sp.diff(v, y)
    eps_zz = sp.diff(w, z)

    gamma_xy = (
        sp.diff(u, y)
        + sp.diff(v, x)
    )

    gamma_xz = (
        sp.diff(u, z)
        + sp.diff(w, x)
    )

    gamma_yz = (
        sp.diff(v, z)
        + sp.diff(w, y)
    )

    strain = sp.Matrix(
        [
            sp.simplify(eps_xx),
            sp.simplify(eps_yy),
            sp.simplify(eps_zz),
            sp.simplify(gamma_xy),
            sp.simplify(gamma_xz),
            sp.simplify(gamma_yz),
        ]
    )

    return SymbolicKinematics(
        x=x,
        y=y,
        z=z,
        u0=u0,
        v0=v0,
        w0=w0,
        beta_x=beta_x,
        beta_y=beta_y,
        eta=eta,
        fx=fx,
        fy=fy,
        g=g,
        u=sp.simplify(u),
        v=sp.simplify(v),
        w=sp.simplify(w),
        strain=strain,
    )


def expected_symbolic_strains(
    kin: SymbolicKinematics,
) -> Tuple[sp.Expr, ...]:
    """
    Return the analytically expected directional HOST strains.
    """

    x = kin.x
    y = kin.y
    z = kin.z

    eps_xx = (
        sp.diff(kin.u0, x)
        - z * sp.diff(kin.w0, x, 2)
        + kin.fx * sp.diff(kin.beta_x, x)
    )

    eps_yy = (
        sp.diff(kin.v0, y)
        - z * sp.diff(kin.w0, y, 2)
        + kin.fy * sp.diff(kin.beta_y, y)
    )

    eps_zz = (
        sp.diff(kin.g, z)
        * kin.eta
    )

    gamma_xy = (
        sp.diff(kin.u0, y)
        + sp.diff(kin.v0, x)
        - 2 * z * sp.diff(kin.w0, x, y)
        + kin.fx * sp.diff(kin.beta_x, y)
        + kin.fy * sp.diff(kin.beta_y, x)
    )

    gamma_xz = (
        sp.diff(kin.fx, z) * kin.beta_x
        + kin.g * sp.diff(kin.eta, x)
    )

    gamma_yz = (
        sp.diff(kin.fy, z) * kin.beta_y
        + kin.g * sp.diff(kin.eta, y)
    )

    return (
        sp.simplify(eps_xx),
        sp.simplify(eps_yy),
        sp.simplify(eps_zz),
        sp.simplify(gamma_xy),
        sp.simplify(gamma_xz),
        sp.simplify(gamma_yz),
    )