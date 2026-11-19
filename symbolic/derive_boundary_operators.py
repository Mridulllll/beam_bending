#!/usr/bin/env python3

"""
Display and verify HOST HSDT natural boundary operators.
"""

from pathlib import Path
import sys

import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from host.boundary_operators import (  # noqa: E402
    build_symbolic_boundary_operators,
)


def show_edge(name, nx, ny):
    bc = build_symbolic_boundary_operators(
        nx,
        ny,
    )

    print("=" * 72)
    print(name)
    print("=" * 72)

    quantities = (
        ("t_N,u", bc.membrane_u),
        ("t_N,v", bc.membrane_v),
        ("t_P,beta_x", bc.warping_beta_x),
        ("t_P,beta_y", bc.warping_beta_y),
        ("M_nn", bc.M_nn),
        ("M_ns", bc.M_ns),
        ("V_n", bc.V_n),
    )

    for label, expression in quantities:
        print("\n{} =".format(label))
        sp.pprint(expression)

    print()


def main():
    show_edge(
        "Right edge: n = (+1, 0)",
        1,
        0,
    )

    show_edge(
        "Left edge: n = (-1, 0)",
        -1,
        0,
    )

    show_edge(
        "Top edge: n = (0, +1)",
        0,
        1,
    )

    show_edge(
        "Bottom edge: n = (0, -1)",
        0,
        -1,
    )


if __name__ == "__main__":
    main()