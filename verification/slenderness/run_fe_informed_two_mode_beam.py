#!/usr/bin/env python3
"""
Proof-of-mechanism: solve the 4PB beam with the FE/SVD residual shear mode as
a second through-thickness warping mode.

Purpose
-------
The analytic polynomial family tested in
fit_analytic_second_shear_mode.py does not yet represent the FE/SVD residual
mode accurately enough, and its *local* beta-beta condensation gives almost no
softening.  Before designing a better analytic f2(z), this script answers a
more fundamental question:

    Does the full two-mode beam, including the beta2'(x)=chi2 gradient
    mechanics and its higher-order boundary layers, explain the global
    3-D-FE/HOST discrepancy when the actual FE-informed residual mode is used?

This is deliberately a verification/proof script.  It does NOT modify the
production HOST theory and it does NOT make the discrete FE/SVD mode a final
model ingredient.

Kinematics
----------
    u = u0 - z*w' + f1*beta1 + f2*beta2
    w = w

    eps_xx  = eps0 + z*kappa + f1*chi1 + f2*chi2
    gamma_xz = f1'*beta1 + f2'*beta2

    chi1 = beta1'
    chi2 = beta2'

The full section stiffness K6 is integrated from the canonical free-stress
beam constitutive matrices and then eps0 is condensed under N=0 to K5:

    [M,H1,H2,Q1,Q2]^T
        = K5 [kappa,chi1,chi2,beta1,beta2]^T.

The eight-state beam is

    y = [w,theta,beta1,beta2,M,V,H1,H2]^T

with

    w'      = theta
    theta'  = -kappa
    beta1'  = chi1
    beta2'  = chi2
    M'      = V
    V'      = -q
    H1'     = Q1
    H2'     = Q2.

The point-load BVP uses exact matrix-exponential propagation and equilibrated
multiple shooting, with auxiliary knots no farther apart than one unit-cell
size by default.

Boundary conditions match the one-mode point-load solver:

    left free physical end:
        M=V=H1=H2=0

    simple supports:
        w(xA)=w(xB)=0

    right higher-order natural conditions:
        H1=H2=0

Right-end M=V=0 are post-solution equilibrium checks.

Second-mode construction
------------------------
The common FE/SVD residual *derivative* f2'(z) is read from

    shear_profile_modes.json

and symmetrized.  Separate monotone cubic interpolants are constructed in the
core and face so the derivative may jump at the material interface.  The
surface value is enforced as f2'(+-h/2)=0.  f2(z) is obtained by analytical
integration of those piecewise cubic interpolants and is therefore continuous
and odd.

The normalization of f2 is arbitrary.  The script uses max|f2'|=1; the exact
two-mode solution is invariant to this scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modes-json",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_profile_modes/"
            "shear_profile_modes.json"
        ),
    )

    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "verification/beam_reduction/"
            "plate_to_beam_audit.json"
        ),
    )

    parser.add_argument(
        "--slenderness",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/slenderness_sweep.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/fe_informed_two_mode_beam"
        ),
    )

    parser.add_argument(
        "--max-segment-mm",
        type=float,
        default=10.0,
        help="Maximum multiple-shooting segment length.",
    )

    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=80,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def isotropic_stiffness_3d(E: float, nu: float) -> np.ndarray:
    E = float(E)
    nu = float(nu)

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    C = np.zeros((6, 6), dtype=float)
    C[:3, :3] = lam
    C[0, 0] = lam + 2.0 * mu
    C[1, 1] = lam + 2.0 * mu
    C[2, 2] = lam + 2.0 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


def free_stress_x_beam_stiffness(C3: np.ndarray) -> np.ndarray:
    """
    Active physical strain components [eps_xx, gamma_xz].

    Inactive stresses:
        sigma_yy = sigma_zz = tau_xy = tau_yz = 0.
    """
    C3 = np.asarray(C3, dtype=float)

    active = (0, 4)
    inactive = (1, 2, 3, 5)

    Caa = C3[np.ix_(active, active)]
    Cai = C3[np.ix_(active, inactive)]
    Cia = C3[np.ix_(inactive, active)]
    Cii = C3[np.ix_(inactive, inactive)]

    return Caa - Cai @ np.linalg.solve(Cii, Cia)


def f1_functions(
    *,
    h: float,
    hc: float,
    G_core: float,
    G_face: float,
) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Exact current material-informed symmetric warping.
    """
    h = float(h)
    hc = float(hc)

    zi = 0.5 * hc
    ratio = float(G_core) / float(G_face)

    def R(z: float) -> float:
        z = float(z)
        return z - 4.0 * z ** 3 / (3.0 * h ** 2)

    Ri = R(zi)

    def df(z: float) -> float:
        z = float(z)

        p = 1.0 - 4.0 * z * z / (h * h)

        if abs(z) <= zi:
            return p

        return ratio * p

    def f(z: float) -> float:
        z = float(z)

        if z < 0.0:
            return -f(-z)

        if z <= zi:
            return R(z)

        return Ri + ratio * (R(z) - Ri)

    return f, df


class FEInformedSecondMode:
    """
    Piecewise PCHIP representation of the even derivative f2'(z) and its
    continuous odd primitive f2(z).
    """

    def __init__(
        self,
        z: np.ndarray,
        df_values: np.ndarray,
        *,
        h: float,
        hc: float,
    ) -> None:
        z = np.asarray(z, dtype=float)
        values = np.asarray(df_values, dtype=float)

        if z.ndim != 1 or values.shape != z.shape:
            raise ValueError("z/df_values must be matching 1-D arrays.")

        h = float(h)
        hc = float(hc)
        zi = 0.5 * hc
        zs = 0.5 * h

        # Symmetrize the sampled mode explicitly.
        order = np.argsort(z)
        z = z[order]
        values = values[order]

        even = np.empty_like(values)

        for i, zi_sample in enumerate(z):
            j = int(np.argmin(np.abs(z + zi_sample)))

            if abs(z[j] + zi_sample) > 1.0e-6:
                raise RuntimeError(
                    "SVD mode does not have a symmetric z partner at "
                    "z={:.12g}.".format(zi_sample)
                )

            even[i] = 0.5 * (values[i] + values[j])

        scale = float(np.max(np.abs(even)))

        if scale <= 0.0:
            raise RuntimeError("Degenerate FE residual mode.")

        even = even / scale

        positive = z > 0.0
        zp = z[positive]
        vp = even[positive]

        core_mask = zp < zi
        face_mask = zp > zi

        z_core_data = zp[core_mask]
        v_core_data = vp[core_mask]
        z_face_data = zp[face_mask]
        v_face_data = vp[face_mask]

        if len(z_core_data) < 2 or len(z_face_data) < 2:
            raise RuntimeError(
                "Insufficient positive-half profile points in core/face."
            )

        core_seed = PchipInterpolator(
            z_core_data,
            v_core_data,
            extrapolate=True,
        )

        face_seed = PchipInterpolator(
            z_face_data,
            v_face_data,
            extrapolate=True,
        )

        v_center = float(core_seed(0.0))
        v_interface_core = float(core_seed(zi))
        v_interface_face = float(face_seed(zi))

        # Face outer surface is traction-free.  Enforce the modal derivative
        # to vanish there rather than extrapolating an IP value to the surface.
        v_surface = 0.0

        z_core = np.concatenate(
            (
                np.asarray([0.0]),
                z_core_data,
                np.asarray([zi]),
            )
        )
        v_core = np.concatenate(
            (
                np.asarray([v_center]),
                v_core_data,
                np.asarray([v_interface_core]),
            )
        )

        z_face = np.concatenate(
            (
                np.asarray([zi]),
                z_face_data,
                np.asarray([zs]),
            )
        )
        v_face = np.concatenate(
            (
                np.asarray([v_interface_face]),
                v_face_data,
                np.asarray([v_surface]),
            )
        )

        self._core = PchipInterpolator(
            z_core,
            v_core,
            extrapolate=False,
        )
        self._face = PchipInterpolator(
            z_face,
            v_face,
            extrapolate=False,
        )

        self._core_int = self._core.antiderivative()
        self._face_int = self._face.antiderivative()

        self.h = h
        self.hc = hc
        self.zi = zi
        self.zs = zs

        self.interface_core_df = v_interface_core
        self.interface_face_df = v_interface_face
        self.center_df = v_center

        self.f_at_interface = float(
            self._core_int(zi)
            - self._core_int(0.0)
        )

    def df(self, z: float) -> float:
        z = float(z)
        za = abs(z)

        if za > self.zs + 1.0e-10:
            raise ValueError("z lies outside the section.")

        if za <= self.zi:
            return float(self._core(za))

        return float(self._face(za))

    def f(self, z: float) -> float:
        z = float(z)

        if z < 0.0:
            return -self.f(-z)

        if z <= self.zi:
            return float(
                self._core_int(z)
                - self._core_int(0.0)
            )

        return self.f_at_interface + float(
            self._face_int(z)
            - self._face_int(self.zi)
        )


def build_K6(
    *,
    width: float,
    h: float,
    hc: float,
    Cb_core: np.ndarray,
    Cb_face: np.ndarray,
    f1: Callable[[float], float],
    df1: Callable[[float], float],
    f2: Callable[[float], float],
    df2: Callable[[float], float],
    quadrature_order: int,
) -> np.ndarray:
    """
    Generalized order:
        [eps0, kappa, chi1, chi2, beta1, beta2].
    """
    h = float(h)
    hc = float(hc)

    layers = (
        (-0.5 * h, -0.5 * hc, Cb_face),
        (-0.5 * hc, +0.5 * hc, Cb_core),
        (+0.5 * hc, +0.5 * h, Cb_face),
    )

    points, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )

    K = np.zeros((6, 6), dtype=float)

    for z0, z1, Cb in layers:
        jac = 0.5 * (z1 - z0)
        mid = 0.5 * (z1 + z0)

        for point, weight in zip(points, weights):
            z = mid + jac * float(point)

            B = np.asarray(
                [
                    [
                        1.0,
                        z,
                        float(f1(z)),
                        float(f2(z)),
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        float(df1(z)),
                        float(df2(z)),
                    ],
                ],
                dtype=float,
            )

            K += (
                float(width)
                * jac
                * float(weight)
                * (B.T @ Cb @ B)
            )

    return 0.5 * (K + K.T)


def condense_N_zero(K6: np.ndarray) -> np.ndarray:
    K6 = np.asarray(K6, dtype=float)

    K00 = float(K6[0, 0])
    K0t = K6[0, 1:]
    Ktt = K6[1:, 1:]

    return (
        Ktt
        - np.outer(K0t, K0t) / K00
    )


def generalized_from_state(
    K5: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given
        y=[w,theta,beta1,beta2,M,V,H1,H2],
    recover
        e=[kappa,chi1,chi2,beta1,beta2]
    and
        Q=[Q1,Q2].
    """
    S = np.linalg.inv(
        np.asarray(K5, dtype=float)
    )

    beta = np.asarray(
        [y[2], y[3]],
        dtype=float,
    )

    r_a = np.asarray(
        [y[4], y[6], y[7]],
        dtype=float,
    )

    S_beta_a = S[3:5, 0:3]
    S_beta_Q = S[3:5, 3:5]

    Q = np.linalg.solve(
        S_beta_Q,
        beta - S_beta_a @ r_a,
    )

    r = np.concatenate(
        (r_a, Q)
    )

    e = S @ r

    return e, Q


def state_derivative(
    K5: np.ndarray,
    y: np.ndarray,
    *,
    q: float = 0.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    e, Q = generalized_from_state(
        K5,
        y,
    )

    kappa = float(e[0])
    chi1 = float(e[1])
    chi2 = float(e[2])

    derivative = np.zeros(
        8,
        dtype=float,
    )

    derivative[0] = y[1]
    derivative[1] = -kappa
    derivative[2] = chi1
    derivative[3] = chi2
    derivative[4] = y[5]
    derivative[5] = -float(q)
    derivative[6] = Q[0]
    derivative[7] = Q[1]

    return derivative


def state_matrix(K5: np.ndarray) -> np.ndarray:
    A = np.zeros((8, 8), dtype=float)

    for j in range(8):
        basis = np.zeros(8, dtype=float)
        basis[j] = 1.0

        A[:, j] = state_derivative(
            K5,
            basis,
            q=0.0,
        )

    return A


def equilibrated_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)

    row_norms = np.linalg.norm(A, axis=1)

    if np.any(row_norms == 0.0):
        raise RuntimeError("Zero equation row in multiple shooting.")

    row_scale = 1.0 / row_norms

    Ar = row_scale[:, None] * A
    br = row_scale * b

    column_norms = np.linalg.norm(Ar, axis=0)

    if np.any(column_norms == 0.0):
        raise RuntimeError("Unconstrained state variable.")

    column_scale = 1.0 / column_norms

    As = Ar * column_scale[None, :]

    xs = np.linalg.solve(
        As,
        br,
    )

    return column_scale * xs


def unique_sorted(values: Sequence[float], tolerance: float = 1.0e-10) -> np.ndarray:
    result: List[float] = []

    for value in sorted(float(v) for v in values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)

    return np.asarray(result, dtype=float)


def build_knots(
    *,
    length: float,
    force_positions: Sequence[float],
    max_segment_mm: float,
) -> np.ndarray:
    base = unique_sorted(
        [0.0, float(length), *force_positions]
    )

    values: List[float] = [float(base[0])]

    for x0, x1 in zip(base[:-1], base[1:]):
        dx = float(x1 - x0)

        n = max(
            1,
            int(
                math.ceil(
                    dx / float(max_segment_mm)
                )
            ),
        )

        for j in range(1, n + 1):
            values.append(
                float(x0 + dx * j / n)
            )

    return unique_sorted(values)


def knot_index(knots: np.ndarray, x: float) -> int:
    indices = np.where(
        np.abs(knots - float(x)) <= 1.0e-9
    )[0]

    if len(indices) != 1:
        raise RuntimeError(
            "Could not identify unique knot at x={}".format(x)
        )

    return int(indices[0])


def solve_two_mode_4pb(
    K5: np.ndarray,
    *,
    length: float,
    support_left: float,
    support_right: float,
    load_left: float,
    load_right: float,
    total_load: float,
    max_segment_mm: float,
) -> dict:
    """
    Sign convention follows the current HOST point-load solver:
    positive applied force produces jump V+ = V- - P.

    Symmetric downward loading therefore uses +P/2 at each load roller and
    -P/2 support reactions.
    """
    half_load = 0.5 * float(total_load)

    forces = {
        float(support_left): -half_load,
        float(load_left): +half_load,
        float(load_right): +half_load,
        float(support_right): -half_load,
    }

    force_positions = tuple(forces.keys())

    knots = build_knots(
        length=length,
        force_positions=force_positions,
        max_segment_mm=max_segment_mm,
    )

    n = len(knots)
    ndof = 8 * n

    A = state_matrix(K5)

    system = np.zeros(
        (ndof, ndof),
        dtype=float,
    )
    rhs = np.zeros(
        ndof,
        dtype=float,
    )

    row = 0

    for j in range(n - 1):
        dx = float(
            knots[j + 1] - knots[j]
        )

        T = expm(
            A * dx
        )

        left = slice(
            8 * j,
            8 * j + 8,
        )
        right = slice(
            8 * (j + 1),
            8 * (j + 1) + 8,
        )

        system[
            row:row + 8,
            left,
        ] = -T

        system[
            row:row + 8,
            right,
        ] = np.eye(8)

        force = float(
            forces.get(
                float(knots[j + 1]),
                0.0,
            )
        )

        jump = np.zeros(
            8,
            dtype=float,
        )
        jump[5] = -force

        rhs[
            row:row + 8
        ] = jump

        row += 8

    # Left free physical end:
    # M=V=H1=H2=0.
    for component in (4, 5, 6, 7):
        system[row, component] = 1.0
        row += 1

    iA = knot_index(
        knots,
        support_left,
    )
    iB = knot_index(
        knots,
        support_right,
    )

    # Simple-support transverse displacement.
    system[
        row,
        8 * iA + 0,
    ] = 1.0
    row += 1

    system[
        row,
        8 * iB + 0,
    ] = 1.0
    row += 1

    # Right higher-order natural conditions.
    system[
        row,
        8 * (n - 1) + 6,
    ] = 1.0
    row += 1

    system[
        row,
        8 * (n - 1) + 7,
    ] = 1.0
    row += 1

    if row != ndof:
        raise RuntimeError(
            "Multiple-shooting equation count mismatch: {} != {}".format(
                row,
                ndof,
            )
        )

    solution = equilibrated_solve(
        system,
        rhs,
    ).reshape((n, 8))

    def state_at(x: float) -> np.ndarray:
        x = float(x)

        matches = np.where(
            np.abs(knots - x) <= 1.0e-9
        )[0]

        if len(matches) == 1:
            return solution[int(matches[0])].copy()

        j = int(
            np.searchsorted(
                knots,
                x,
                side="right",
            )
            - 1
        )

        if j < 0 or j >= n - 1:
            raise RuntimeError(
                "Evaluation coordinate is outside beam segments."
            )

        dx = x - float(knots[j])

        return expm(A * dx) @ solution[j]

    mid = state_at(
        0.5 * float(length)
    )

    left_support_state = state_at(
        support_left
    )
    right_support_state = state_at(
        support_right
    )
    right_end = state_at(
        length
    )

    return {
        "knots": knots,
        "states_right": solution,
        "state_matrix": A,
        "midspan": mid,
        "left_support": left_support_state,
        "right_support": right_support_state,
        "right_end": right_end,
        "right_end_M": float(right_end[4]),
        "right_end_V": float(right_end[5]),
        "right_end_H1": float(right_end[6]),
        "right_end_H2": float(right_end[7]),
        "support_left_w": float(left_support_state[0]),
        "support_right_w": float(right_support_state[0]),
    }


def main() -> int:
    args = parse_args()

    modes_path = args.modes_json.resolve()
    audit_path = args.audit.resolve()
    slenderness_path = args.slenderness.resolve()
    output_dir = args.output_dir.resolve()

    for path in (modes_path, audit_path, slenderness_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    modes = load_json(modes_path)
    audit = load_json(audit_path)
    slenderness = load_json(slenderness_path)

    source_manifest_path = Path(
        audit["manifest"]
    )

    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            source_manifest_path
        )

    source_manifest = load_json(
        source_manifest_path
    )

    C_core = np.asarray(
        source_manifest["constitutive"]["C_hom_model_MPa"],
        dtype=float,
    )

    face = audit["face_material"]

    C_face = isotropic_stiffness_3d(
        float(face["E_MPa"]),
        float(face["nu"]),
    )

    Cb_core = free_stress_x_beam_stiffness(
        C_core
    )
    Cb_face = free_stress_x_beam_stiffness(
        C_face
    )

    G_core = float(
        audit["shear_moduli"]["G_xz_core_MPa"]
    )
    G_face = float(
        audit["shear_moduli"]["G_xz_face_MPa"]
    )

    rows_source = [
        row
        for row in slenderness["rows"]
        if row.get("FE_3D_abs_w_mid_mm") not in ("", None)
    ]

    if not rows_source:
        raise RuntimeError(
            "Slenderness JSON contains no FE-populated rows."
        )

    # Section geometry is constant over the sweep.
    first = rows_source[0]

    h = float(first["total_height_mm"])
    hc = float(first["core_height_mm"])
    width = float(first["width_mm"])

    f1, df1 = f1_functions(
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
    )

    residual = modes["residual_svd"]

    z_mode = np.asarray(
        residual["common_residual_mode_z_mm"],
        dtype=float,
    )
    df2_mode = np.asarray(
        residual["common_residual_mode"],
        dtype=float,
    )

    second_mode = FEInformedSecondMode(
        z_mode,
        df2_mode,
        h=h,
        hc=hc,
    )

    K6 = build_K6(
        width=width,
        h=h,
        hc=hc,
        Cb_core=Cb_core,
        Cb_face=Cb_face,
        f1=f1,
        df1=df1,
        f2=second_mode.f,
        df2=second_mode.df,
        quadrature_order=int(args.quadrature_order),
    )

    eig_K6 = np.linalg.eigvalsh(K6)

    K5 = condense_N_zero(
        K6
    )

    eig_K5 = np.linalg.eigvalsh(K5)

    if float(np.min(eig_K6)) <= 0.0:
        raise RuntimeError(
            "K6 is not positive definite."
        )

    if float(np.min(eig_K5)) <= 0.0:
        raise RuntimeError(
            "K5 is not positive definite."
        )

    # Current one-mode condensed stiffness for an exact implementation check.
    K3_current = np.asarray(
        audit["models"]["beam_free_stress"]["K3_N_zero"],
        dtype=float,
    )

    # Local beta-beta block and local Q2=0 condensation, retained as a useful
    # comparison with the earlier analytic-mode test.
    beta_block = K5[3:5, 3:5]

    local_alpha = float(
        (
            beta_block[0, 0]
            - beta_block[0, 1] ** 2 / beta_block[1, 1]
        )
        / beta_block[0, 0]
    )

    result_rows = []

    print()
    print("=" * 126)
    print("HOST FE-INFORMED TWO-MODE BEAM PROOF-OF-MECHANISM")
    print("=" * 126)
    print(
        "Residual SVD first-mode fraction : {:.8%}".format(
            float(
                residual["first_mode_explained_fraction"]
            )
        )
    )
    print(
        "Second-mode df interface core/face: {:+.6e} / {:+.6e}".format(
            second_mode.interface_core_df,
            second_mode.interface_face_df,
        )
    )
    print(
        "K6 min/max eigenvalue            : {:.6e} / {:.6e}".format(
            float(np.min(eig_K6)),
            float(np.max(eig_K6)),
        )
    )
    print(
        "K5 min/max eigenvalue            : {:.6e} / {:.6e}".format(
            float(np.min(eig_K5)),
            float(np.max(eig_K5)),
        )
    )
    print(
        "Local beta-block condensed alpha : {:.9f}".format(
            local_alpha
        )
    )
    print()
    print(
        "{:>6s} {:>8s} {:>13s} {:>13s} {:>13s} {:>11s} {:>11s}".format(
            "N_A",
            "a/h",
            "HOST 1mode",
            "HOST 2mode",
            "FE",
            "1mode err",
            "2mode err",
        )
    )

    for row in rows_source:
        length = float(row["length_mm"])
        support_left = float(row["support_left_mm"])
        support_right = float(row["support_right_mm"])
        load_left = float(row["load_left_mm"])
        load_right = float(row["load_right_mm"])

        solution = solve_two_mode_4pb(
            K5,
            length=length,
            support_left=support_left,
            support_right=support_right,
            load_left=load_left,
            load_right=load_right,
            total_load=1000.0,
            max_segment_mm=float(args.max_segment_mm),
        )

        w_two = abs(
            float(
                solution["midspan"][0]
            )
        )

        w_one = float(
            row["HOST_1D_abs_w_mid_mm"]
        )

        w_fe = float(
            row["FE_3D_abs_w_mid_mm"]
        )

        err_one = (
            w_one - w_fe
        ) / w_fe

        err_two = (
            w_two - w_fe
        ) / w_fe

        result = {
            "n_shear": int(row["n_shear"]),
            "a_over_h": float(row["a_over_h"]),
            "HOST_one_mode_abs_w_mid_mm": w_one,
            "HOST_two_mode_FE_informed_abs_w_mid_mm": w_two,
            "FE_abs_w_mid_mm": w_fe,
            "one_mode_error_relative": float(err_one),
            "two_mode_error_relative": float(err_two),
            "two_minus_one_relative_to_one": float(
                (w_two - w_one) / w_one
            ),
            "midspan_beta1": float(
                solution["midspan"][2]
            ),
            "midspan_beta2": float(
                solution["midspan"][3]
            ),
            "right_end_M": solution["right_end_M"],
            "right_end_V": solution["right_end_V"],
            "right_end_H1": solution["right_end_H1"],
            "right_end_H2": solution["right_end_H2"],
            "support_left_w": solution["support_left_w"],
            "support_right_w": solution["support_right_w"],
            "knot_count": int(
                len(solution["knots"])
            ),
        }

        result_rows.append(result)

        print(
            "{:6d} {:8.3f} {:13.8f} {:13.8f} {:13.8f} {:10.3%} {:10.3%}".format(
                result["n_shear"],
                result["a_over_h"],
                w_one,
                w_two,
                w_fe,
                err_one,
                err_two,
            )
        )

    errors_one = np.asarray(
        [row["one_mode_error_relative"] for row in result_rows],
        dtype=float,
    )
    errors_two = np.asarray(
        [row["two_mode_error_relative"] for row in result_rows],
        dtype=float,
    )

    rms_one = float(
        math.sqrt(
            np.mean(
                errors_one ** 2
            )
        )
    )
    rms_two = float(
        math.sqrt(
            np.mean(
                errors_two ** 2
            )
        )
    )

    max_one = float(
        np.max(
            np.abs(errors_one)
        )
    )
    max_two = float(
        np.max(
            np.abs(errors_two)
        )
    )

    max_support = float(
        max(
            max(abs(row["support_left_w"]), abs(row["support_right_w"]))
            for row in result_rows
        )
    )

    max_right_natural = float(
        max(
            max(
                abs(row["right_end_H1"]),
                abs(row["right_end_H2"]),
            )
            for row in result_rows
        )
    )

    max_right_equilibrium = float(
        max(
            max(
                abs(row["right_end_M"]),
                abs(row["right_end_V"]),
            )
            for row in result_rows
        )
    )

    print()
    print(
        "RMS deflection error 1mode -> 2mode: {:.4f}% -> {:.4f}%".format(
            100.0 * rms_one,
            100.0 * rms_two,
        )
    )
    print(
        "Max deflection error 1mode -> 2mode: {:.4f}% -> {:.4f}%".format(
            100.0 * max_one,
            100.0 * max_two,
        )
    )
    print(
        "Max support |w|                    : {:.3e} mm".format(
            max_support
        )
    )
    print(
        "Max right-end |H1,H2|              : {:.3e}".format(
            max_right_natural
        )
    )
    print(
        "Max right-end |M,V| equilibrium    : {:.3e}".format(
            max_right_equilibrium
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "fe_informed_two_mode_beam.csv"
    json_path = output_dir / "fe_informed_two_mode_beam.json"

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            result_rows
        )

    report = {
        "schema": "host_fe_informed_two_mode_beam_proof_v1",
        "warning": (
            "The second mode is directly FE/SVD-informed and is used only as "
            "a proof-of-mechanism, not as a final analytical HOST mode."
        ),
        "section": {
            "h_mm": h,
            "hc_mm": hc,
            "width_mm": width,
            "K6": K6.tolist(),
            "K5_N_zero": K5.tolist(),
            "K3_current_one_mode": K3_current.tolist(),
            "K6_eigenvalues": eig_K6.tolist(),
            "K5_eigenvalues": eig_K5.tolist(),
            "local_beta_block": beta_block.tolist(),
            "local_beta_condensed_alpha": local_alpha,
        },
        "second_mode": {
            "z_source_mm": z_mode.tolist(),
            "df_source_SVD": df2_mode.tolist(),
            "normalization": "symmetrized and max(abs(df2))=1",
            "df_center": second_mode.center_df,
            "df_interface_core": second_mode.interface_core_df,
            "df_interface_face": second_mode.interface_face_df,
            "f_interface": second_mode.f_at_interface,
        },
        "results": result_rows,
        "summary": {
            "RMS_one_mode_relative_error": rms_one,
            "RMS_two_mode_relative_error": rms_two,
            "max_one_mode_relative_error": max_one,
            "max_two_mode_relative_error": max_two,
            "max_support_abs_w_mm": max_support,
            "max_right_natural_abs": max_right_natural,
            "max_right_equilibrium_abs": max_right_equilibrium,
        },
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )
        handle.write("\n")

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        a = np.asarray(
            [row["a_over_h"] for row in result_rows],
            dtype=float,
        )

        ax.plot(
            a,
            100.0 * errors_one,
            marker="o",
            label="current one-mode HOST",
        )
        ax.plot(
            a,
            100.0 * errors_two,
            marker="s",
            label="FE-informed two-mode HOST",
        )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("a/h")
        ax.set_ylabel("deflection error [%]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "two_mode_error_vs_slenderness.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

        z_plot = np.linspace(
            -0.5 * h,
            +0.5 * h,
            501,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(
            [df1(z) for z in z_plot],
            z_plot / h,
            label="f1'",
        )
        ax.plot(
            [second_mode.df(z) for z in z_plot],
            z_plot / h,
            label="FE/SVD f2'",
        )

        ax.set_xlabel("normalized shear-mode derivative")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "f1_f2_derivatives.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

        report["plots"] = plot_paths

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                indent=2,
            )
            handle.write("\n")

    except Exception as exc:
        plot_paths.append(
            "Plotting skipped: {}".format(exc)
        )

    print()
    print("Wrote:")
    print(" ", csv_path)
    print(" ", json_path)
    for path in plot_paths:
        print(" ", path)
    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
