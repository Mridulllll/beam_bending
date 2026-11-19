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


def test_symbolic_strains_match_closed_form():
    kin = build_symbolic_kinematics()
    expected = expected_symbolic_strains(kin)

    assert len(kin.strain) == 6

    for derived, reference in zip(
        kin.strain,
        expected,
    ):
        assert sp.simplify(derived - reference) == 0


def test_kirchhoff_limit():
    kin = build_symbolic_kinematics()

    substitutions = {
        kin.beta_x: 0,
        kin.beta_y: 0,
        kin.eta: 0,
    }

    strain = [
        sp.simplify(expr.subs(substitutions))
        for expr in kin.strain
    ]

    assert strain[2] == 0
    assert strain[4] == 0
    assert strain[5] == 0


def test_transverse_normal_strain_origin():
    kin = build_symbolic_kinematics()

    expected = (
        sp.diff(kin.g, kin.z)
        * kin.eta
    )

    assert sp.simplify(
        kin.strain[2] - expected
    ) == 0


def test_shear_cancellation_of_kirchhoff_term():
    kin = build_symbolic_kinematics()

    expected_xz = (
        sp.diff(kin.fx, kin.z) * kin.beta_x
        + kin.g * sp.diff(kin.eta, kin.x)
    )

    expected_yz = (
        sp.diff(kin.fy, kin.z) * kin.beta_y
        + kin.g * sp.diff(kin.eta, kin.y)
    )

    assert sp.simplify(
        kin.strain[4] - expected_xz
    ) == 0

    assert sp.simplify(
        kin.strain[5] - expected_yz
    ) == 0


def test_pure_rigid_translation_has_zero_strain():
    kin = build_symbolic_kinematics()

    c_u, c_v, c_w = sp.symbols(
        "c_u c_v c_w",
        real=True,
    )

    substitutions = {
        kin.u0: c_u,
        kin.v0: c_v,
        kin.w0: c_w,
        kin.beta_x: 0,
        kin.beta_y: 0,
        kin.eta: 0,
    }

    strain = [
        sp.simplify(expr.subs(substitutions))
        for expr in kin.strain
    ]

    assert all(expr == 0 for expr in strain)
