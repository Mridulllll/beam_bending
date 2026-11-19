#!/usr/bin/env python3

from pathlib import Path
import sys

import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from host.kinematics import (
    build_symbolic_kinematics,
    expected_symbolic_strains,
)


def main():
    kin = build_symbolic_kinematics()
    expected = expected_symbolic_strains(kin)

    names = (
        "epsilon_xx",
        "epsilon_yy",
        "epsilon_zz",
        "gamma_xy",
        "gamma_xz",
        "gamma_yz",
    )

    print("=" * 72)
    print("HOST analytical kinematics")
    print("=" * 72)

    print("\nDisplacement field\n")

    print("u =")
    sp.pprint(kin.u)

    print("\nv =")
    sp.pprint(kin.v)

    print("\nw =")
    sp.pprint(kin.w)

    print("\n" + "=" * 72)
    print("Engineering strains")
    print("=" * 72)

    for name, expr in zip(names, kin.strain):
        print("\n{} =".format(name))
        sp.pprint(expr)

    print("\n" + "=" * 72)
    print("Symbolic consistency check")
    print("=" * 72)

    all_ok = True

    for name, derived, reference in zip(
        names,
        kin.strain,
        expected,
    ):
        residual = sp.simplify(derived - reference)
        ok = residual == 0

        print("{:12s}: {}".format(
            name,
            "PASS" if ok else "FAIL",
        ))

        if not ok:
            print("Residual:")
            sp.pprint(residual)
            all_ok = False

    if not all_ok:
        raise RuntimeError(
            "Symbolic kinematics consistency check failed."
        )

    print("\nAll symbolic checks passed.")


if __name__ == "__main__":
    main()
