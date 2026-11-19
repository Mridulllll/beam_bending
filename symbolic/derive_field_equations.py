#!/usr/bin/env python3

"""
Independent symbolic verification of the five HOST HSDT field equations.

The script constructs a virtual-work-like Lagrangian density in terms
of arbitrary generalized stress-resultant fields and applies the
Euler-Lagrange operator to the five generalized displacement fields.

The result is compared against the strong-form operators implemented
in host.field_equations.
"""

from pathlib import Path
import sys

import sympy as sp
from sympy.calculus.euler import euler_equations


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from host.field_equations import (  # noqa: E402
    build_symbolic_field_equations,
)


def main():
    reference = build_symbolic_field_equations()

    x = reference.x
    y = reference.y

    u = sp.Function("u")(x, y)
    v = sp.Function("v")(x, y)
    w = sp.Function("w")(x, y)

    beta_x = sp.Function("beta_x")(x, y)
    beta_y = sp.Function("beta_y")(x, y)

    # --------------------------------------------------------------
    # Generalized strains
    # --------------------------------------------------------------

    eps_xx = sp.diff(u, x)
    eps_yy = sp.diff(v, y)
    gamma_xy = (
        sp.diff(u, y)
        + sp.diff(v, x)
    )

    kappa_xx = -sp.diff(w, x, 2)
    kappa_yy = -sp.diff(w, y, 2)
    kappa_xy = -2 * sp.diff(w, x, y)

    chi_xx = sp.diff(beta_x, x)
    chi_yy = sp.diff(beta_y, y)

    chi_xy = (
        sp.diff(beta_x, y)
        + sp.diff(beta_y, x)
    )

    # --------------------------------------------------------------
    # Potential-energy density expressed using arbitrary resultants.
    #
    # This form is linear in the generalized strains because the
    # resultants are treated as independent coefficient fields for
    # verification of the differential adjoint operators.
    # --------------------------------------------------------------

    L = (
        reference.Nxx * eps_xx
        + reference.Nyy * eps_yy
        + reference.Nxy * gamma_xy

        + reference.Mxx * kappa_xx
        + reference.Myy * kappa_yy
        + reference.Mxy * kappa_xy

        + reference.Pxx * chi_xx
        + reference.Pyy * chi_yy
        + reference.Pxy * chi_xy

        + reference.Qx * beta_x
        + reference.Qy * beta_y

        - reference.q * w
    )

    fields = (
        u,
        v,
        w,
        beta_x,
        beta_y,
    )

    equations = euler_equations(
        L,
        fields,
        (x, y),
    )

    # Euler equations are returned as Eq(lhs, rhs).
    residuals = [
        sp.simplify(eq.lhs - eq.rhs)
        for eq in equations
    ]

    # Depending on the Euler-Lagrange sign convention, each equation
    # may be the negative of our equilibrium residual. Zero sets are
    # equivalent. Determine the expected correspondence explicitly.
    expected = [
        reference.eq_u,
        reference.eq_v,
        reference.eq_w,
        reference.eq_beta_x,
        reference.eq_beta_y,
    ]

    names = (
        "u0",
        "v0",
        "w0",
        "beta_x",
        "beta_y",
    )

    print("=" * 72)
    print("HOST HSDT Euler-Lagrange verification")
    print("=" * 72)

    all_ok = True

    for name, derived, target in zip(
        names,
        residuals,
        expected,
    ):
        same = (
            sp.simplify(derived - target)
            == 0
        )

        opposite = (
            sp.simplify(derived + target)
            == 0
        )

        ok = same or opposite

        print(
            "{:10s}: {}".format(
                name,
                "PASS" if ok else "FAIL",
            )
        )

        if not ok:
            print("\nDerived:")
            sp.pprint(derived)

            print("\nExpected:")
            sp.pprint(target)

            print()

            all_ok = False

    if not all_ok:
        raise RuntimeError(
            "Euler-Lagrange verification failed."
        )

    print("\nStrong-form equilibrium equations:\n")

    labels = (
        "u0-equilibrium",
        "v0-equilibrium",
        "w0-equilibrium",
        "beta_x-equilibrium",
        "beta_y-equilibrium",
    )

    for label, expression in zip(
        labels,
        expected,
    ):
        print(label)
        sp.pprint(sp.Eq(expression, 0))
        print()

    print("All field-equation checks passed.")


if __name__ == "__main__":
    main()