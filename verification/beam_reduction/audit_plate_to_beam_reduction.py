#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit the HOST directional five-field plate -> 1-D beam reduction.

Purpose
-------
The current HOST beam is NOT obtained by simply taking an x-only block of the
five-field plate stiffness.  It performs a pointwise free-stress reduction of
the full 3-D constitutive tensor to the active beam stresses

    (sigma_xx, tau_xz),

then integrates the resulting 2x2 constitutive law through the sandwich
thickness.

This audit compares that current beam reduction with two plate-derived
diagnostics built from the directional five-field K12 section stiffness:

1. "plate_cylindrical"
   Strict 4-field cylindrical x-bending sector

       [eps0_xx, kappa_xx, chi_xx, beta_x]

   with all other y/twist generalized fields constrained to zero.

2. "plate_local_relaxed"
   A LOCAL Schur-complement diagnostic that additionally relaxes

       [gamma0_xy, chi_xy_y, beta_y]

   against zero conjugate generalized resultants.

   IMPORTANT: this is not the exact y-invariant plate boundary-value problem,
   because gamma0_xy and chi_xy_y contain x-derivatives of secondary fields.
   It is only a constitutive/local-relaxation diagnostic.  If this correction
   is appreciable, the next step is a coupled 1-D variational/BVP reduction.

The script also solves the same four-point-bending point-load BVP for all three
3x3 reduced section matrices using HOST's existing multiple-shooting solver.

Permanent Voigt ordering
------------------------
    [xx, yy, zz, xy, xz, yz]

Engineering shear strains are assumed.

Run from repository root
------------------------
python verification/beam_reduction/audit_plate_to_beam_reduction.py

Optional:
python verification/beam_reduction/audit_plate_to_beam_reduction.py ^
    --manifest reference_results/4pb/Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1/source_case_manifest.json ^
    --output verification/beam_reduction/plate_to_beam_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from host.beam_point_loads import (
    solve_simply_supported_point_loads,
    symmetric_four_point_bending_loads,
    symmetric_support_positions,
)


VOIGT = ("xx", "yy", "zz", "xy", "xz", "yz")

# Directional five-field generalized-strain ordering:
#
#  0 eps0_xx
#  1 eps0_yy
#  2 gamma0_xy
#  3 kappa_xx
#  4 kappa_yy
#  5 kappa_xy
#  6 chi_xx
#  7 chi_yy
#  8 chi_xy_x = beta_x,y
#  9 chi_xy_y = beta_y,x
# 10 beta_x
# 11 beta_y
DIR12_LABELS = (
    "eps0_xx",
    "eps0_yy",
    "gamma0_xy",
    "kappa_xx",
    "kappa_yy",
    "kappa_xy",
    "chi_xx",
    "chi_yy",
    "chi_xy_x",
    "chi_xy_y",
    "beta_x",
    "beta_y",
)

# Strict x-bending sector retained by the current 4-field beam kinematics.
IDX_PRIMARY_X = (0, 3, 6, 10)

# Secondary fields that can remain nonzero when d()/dy = 0:
# gamma0_xy = v0,x, chi_xy_y = beta_y,x, beta_y.
IDX_SECONDARY_YINV = (2, 9, 11)

# These vanish kinematically under strict d()/dy = 0.
IDX_ZERO_YINV = (1, 4, 5, 7, 8)


def _as_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def isotropic_stiffness_3d(E: float, nu: float) -> np.ndarray:
    """3-D isotropic stiffness in HOST engineering-shear Voigt ordering."""
    E = float(E)
    nu = float(nu)

    if E <= 0.0:
        raise ValueError("E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must satisfy -1 < nu < 0.5.")

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


def engineering_shear_modulus(C: np.ndarray, shear_index: int) -> float:
    """
    Engineering shear modulus from the full compliance matrix.

    shear_index:
        3 -> xy
        4 -> xz
        5 -> yz
    """
    C = np.asarray(C, dtype=float)
    S = np.linalg.inv(C)
    return float(1.0 / S[shear_index, shear_index])


def sigma_zz_zero_stiffness(C3: np.ndarray) -> np.ndarray:
    """
    Five-field plate material reduction:

        active = [xx, yy, xy, xz, yz]
        sigma_zz = 0

    The reduced 5x5 law is embedded back into a 6x6 matrix with a zero zz
    row and column.
    """
    C3 = np.asarray(C3, dtype=float)
    active = (0, 1, 3, 4, 5)
    eliminated = (2,)

    Caa = C3[np.ix_(active, active)]
    Cab = C3[np.ix_(active, eliminated)]
    Cba = C3[np.ix_(eliminated, active)]
    Cbb = C3[np.ix_(eliminated, eliminated)]

    C5 = Caa - Cab @ np.linalg.solve(Cbb, Cba)

    out = np.zeros((6, 6), dtype=float)
    out[np.ix_(active, active)] = C5
    return _as_symmetric(out)


def free_stress_x_beam_stiffness(C3: np.ndarray) -> np.ndarray:
    """
    Current HOST x-beam physical constitutive reduction.

    Active physical strains/stresses:
        [eps_xx, gamma_xz]

    Inactive stresses are set to zero:
        [sigma_yy, sigma_zz, tau_xy, tau_yz]
    """
    C3 = np.asarray(C3, dtype=float)

    active = (0, 4)
    inactive = (1, 2, 3, 5)

    Caa = C3[np.ix_(active, active)]
    Cai = C3[np.ix_(active, inactive)]
    Cia = C3[np.ix_(inactive, active)]
    Cii = C3[np.ix_(inactive, inactive)]

    Cb = Caa - Cai @ np.linalg.solve(Cii, Cia)
    return _as_symmetric(Cb)


def make_directional_material_informed_warping(
    *,
    h_total: float,
    h_core: float,
    G_xz_core: float,
    G_yz_core: float,
    G_xz_face: float,
    G_yz_face: float,
) -> Tuple[
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
]:
    """
    Reconstruct the current symmetric directional material-informed warping.

    p(z) = 1 - 4 z^2 / h^2

    In the core:
        f'_x = p,  f'_y = p

    In a face:
        f'_x = (G_xz_core/G_xz_face) p
        f'_y = (G_yz_core/G_yz_face) p

    f_x and f_y are odd and continuous at the interfaces.
    """
    h = float(h_total)
    hc = float(h_core)

    if h <= 0.0 or hc <= 0.0 or hc >= h:
        raise ValueError("Require 0 < h_core < h_total.")

    zi = 0.5 * hc

    def R(z: float) -> float:
        z = float(z)
        return z - 4.0 * z**3 / (3.0 * h**2)

    R_i = R(zi)

    def make_pair(Gc: float, Gf: float):
        Gc = float(Gc)
        Gf = float(Gf)

        if Gc <= 0.0 or Gf <= 0.0:
            raise ValueError("Shear moduli must be positive.")

        ratio = Gc / Gf

        def df(z: float) -> float:
            z = float(z)
            p = 1.0 - 4.0 * z**2 / h**2
            if abs(z) <= zi:
                return p
            return ratio * p

        def f_positive(z: float) -> float:
            z = float(z)
            if z <= zi:
                return R(z)
            return R_i + ratio * (R(z) - R_i)

        def f(z: float) -> float:
            z = float(z)
            if z >= 0.0:
                return f_positive(z)
            return -f_positive(-z)

        return f, df

    fx, dfx = make_pair(G_xz_core, G_xz_face)
    fy, dfy = make_pair(G_yz_core, G_yz_face)
    return fx, dfx, fy, dfy


def gauss_integrate_matrix(
    z_bottom: float,
    z_top: float,
    order: int,
    integrand: Callable[[float], np.ndarray],
) -> np.ndarray:
    xi, wi = np.polynomial.legendre.leggauss(int(order))

    midpoint = 0.5 * (float(z_bottom) + float(z_top))
    jac = 0.5 * (float(z_top) - float(z_bottom))

    result = None

    for xii, wii in zip(xi, wi):
        z = midpoint + jac * float(xii)
        value = np.asarray(integrand(z), dtype=float)

        if result is None:
            result = np.zeros_like(value, dtype=float)

        result += float(wii) * jac * value

    if result is None:
        raise RuntimeError("Gauss integration produced no values.")

    return result


def layer_bounds(
    *,
    h_core: float,
    h_face_bottom: float,
    h_face_top: float,
) -> Tuple[Tuple[str, float, float], ...]:
    h = float(h_core) + float(h_face_bottom) + float(h_face_top)

    z0 = -0.5 * h
    z1 = z0 + float(h_face_bottom)
    z2 = z1 + float(h_core)
    z3 = z2 + float(h_face_top)

    return (
        ("bottom_face", z0, z1),
        ("core", z1, z2),
        ("top_face", z2, z3),
    )


def beam_B4(z: float, fx: Callable[[float], float], dfx: Callable[[float], float]) -> np.ndarray:
    """
    [eps_xx, gamma_xz]^T = B4 [eps0, kappa, chi, beta_x]^T
    """
    return np.asarray(
        [
            [1.0, float(z), float(fx(z)), 0.0],
            [0.0, 0.0, 0.0, float(dfx(z))],
        ],
        dtype=float,
    )


def directional_F12(
    z: float,
    fx: Callable[[float], float],
    dfx: Callable[[float], float],
    fy: Callable[[float], float],
    dfy: Callable[[float], float],
) -> np.ndarray:
    """
    Directional five-field physical strain operator:

        eps_3D = F12 E12

    using permanent physical Voigt order
        [xx, yy, zz, xy, xz, yz].
    """
    z = float(z)

    F = np.zeros((6, 12), dtype=float)

    # eps_xx = eps0_xx + z*kappa_xx + fx*chi_xx
    F[0, 0] = 1.0
    F[0, 3] = z
    F[0, 6] = float(fx(z))

    # eps_yy = eps0_yy + z*kappa_yy + fy*chi_yy
    F[1, 1] = 1.0
    F[1, 4] = z
    F[1, 7] = float(fy(z))

    # gamma_xy = gamma0_xy + z*kappa_xy
    #          + fx*chi_xy_x + fy*chi_xy_y
    F[3, 2] = 1.0
    F[3, 5] = z
    F[3, 8] = float(fx(z))
    F[3, 9] = float(fy(z))

    # gamma_xz = f'_x beta_x
    F[4, 10] = float(dfx(z))

    # gamma_yz = f'_y beta_y
    F[5, 11] = float(dfy(z))

    return F


def build_section_matrices(
    *,
    C_face: np.ndarray,
    C_core: np.ndarray,
    width: float,
    h_core: float,
    h_face_bottom: float,
    h_face_top: float,
    quadrature_order: int,
) -> Dict[str, np.ndarray]:
    h_total = float(h_core) + float(h_face_bottom) + float(h_face_top)

    G_xz_core = engineering_shear_modulus(C_core, 4)
    G_yz_core = engineering_shear_modulus(C_core, 5)
    G_xz_face = engineering_shear_modulus(C_face, 4)
    G_yz_face = engineering_shear_modulus(C_face, 5)

    fx, dfx, fy, dfy = make_directional_material_informed_warping(
        h_total=h_total,
        h_core=h_core,
        G_xz_core=G_xz_core,
        G_yz_core=G_yz_core,
        G_xz_face=G_xz_face,
        G_yz_face=G_yz_face,
    )

    materials = {
        "bottom_face": C_face,
        "core": C_core,
        "top_face": C_face,
    }

    K4_beam = np.zeros((4, 4), dtype=float)
    K12_plate = np.zeros((12, 12), dtype=float)

    for name, za, zb in layer_bounds(
        h_core=h_core,
        h_face_bottom=h_face_bottom,
        h_face_top=h_face_top,
    ):
        C3 = np.asarray(materials[name], dtype=float)

        Cb = free_stress_x_beam_stiffness(C3)
        C5 = sigma_zz_zero_stiffness(C3)

        def beam_integrand(z: float) -> np.ndarray:
            B = beam_B4(z, fx, dfx)
            return float(width) * (B.T @ Cb @ B)

        def plate_integrand(z: float) -> np.ndarray:
            F = directional_F12(z, fx, dfx, fy, dfy)
            return float(width) * (F.T @ C5 @ F)

        K4_beam += gauss_integrate_matrix(
            za,
            zb,
            quadrature_order,
            beam_integrand,
        )

        K12_plate += gauss_integrate_matrix(
            za,
            zb,
            quadrature_order,
            plate_integrand,
        )

    K4_beam = _as_symmetric(K4_beam)
    K12_plate = _as_symmetric(K12_plate)

    A = IDX_PRIMARY_X
    Bidx = IDX_SECONDARY_YINV

    KAA = K12_plate[np.ix_(A, A)]
    KAB = K12_plate[np.ix_(A, Bidx)]
    KBB = K12_plate[np.ix_(Bidx, Bidx)]

    K4_plate_cyl = _as_symmetric(KAA)

    # Local constitutive relaxation only.  This is NOT the exact coupled
    # y-invariant plate BVP.
    K4_plate_local_relaxed = _as_symmetric(
        KAA - KAB @ np.linalg.solve(KBB, KAB.T)
    )

    coupling_norm = float(
        np.linalg.norm(KAB, ord="fro")
        / math.sqrt(
            np.linalg.norm(KAA, ord="fro")
            * np.linalg.norm(KBB, ord="fro")
        )
    )

    return {
        "K4_beam_free_stress": K4_beam,
        "K12_plate": K12_plate,
        "K4_plate_cylindrical": K4_plate_cyl,
        "K4_plate_local_relaxed": K4_plate_local_relaxed,
        "KAB_primary_secondary": KAB,
        "KBB_secondary": KBB,
        "primary_secondary_coupling_norm": np.asarray([coupling_norm]),
        "shear_moduli_MPa": np.asarray(
            [
                G_xz_core,
                G_yz_core,
                G_xz_face,
                G_yz_face,
            ],
            dtype=float,
        ),
    }


def condense_zero_axial_force(K4: np.ndarray) -> np.ndarray:
    """
    Condense eps0 under N = 0.

    K4 generalized order:
        [eps0, kappa, chi, beta]

    K3 order:
        [kappa, chi, beta]

    resultants:
        [M, H, Q]
    """
    K4 = np.asarray(K4, dtype=float)

    if K4.shape != (4, 4):
        raise ValueError("K4 must have shape (4, 4).")

    KNN = float(K4[0, 0])
    if KNN <= 0.0:
        raise ValueError("K4[0,0] must be positive.")

    KNq = K4[0, 1:]
    KqN = K4[1:, 0]
    Kqq = K4[1:, 1:]

    return _as_symmetric(
        Kqq - np.outer(KqN, KNq) / KNN
    )


def solve_four_point_midspan(
    K3: np.ndarray,
    *,
    length: float,
    support_span: float,
    load_span: float,
    total_load: float,
) -> Dict[str, float]:
    xA, xB = symmetric_support_positions(
        specimen_length=float(length),
        support_span=float(support_span),
    )

    loads = symmetric_four_point_bending_loads(
        total_load=float(total_load),
        specimen_length=float(length),
        load_span=float(load_span),
    )

    solution = solve_simply_supported_point_loads(
        np.asarray(K3, dtype=float),
        length=float(length),
        support_left=float(xA),
        support_right=float(xB),
        applied_forces=loads,
    )

    mid = solution.state(0.5 * float(length))

    return {
        "w_mid_mm": float(mid[0]),
        "theta_mid": float(mid[1]),
        "beta_mid": float(mid[2]),
        "M_mid_Nmm": float(mid[3]),
        "V_mid_N": float(mid[4]),
        "H_mid": float(mid[5]),
        "reaction_left_N": float(solution.reactions.left),
        "reaction_right_N": float(solution.reactions.right),
        "support_left_mm": float(xA),
        "support_right_mm": float(xB),
        "load_left_mm": float(loads[0].x),
        "load_right_mm": float(loads[1].x),
    }


def relative_change(new: float, reference: float) -> float:
    if reference == 0.0:
        return float("nan")
    return (float(new) - float(reference)) / float(reference)


def matrix_to_list(matrix: np.ndarray) -> List[List[float]]:
    return np.asarray(matrix, dtype=float).tolist()


def vector_to_list(vector: np.ndarray) -> List[float]:
    return np.asarray(vector, dtype=float).reshape(-1).tolist()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    try:
        C = data["constitutive"]["C_hom_model_MPa"]
    except KeyError as exc:
        raise KeyError(
            "Manifest does not contain constitutive.C_hom_model_MPa"
        ) from exc

    C = np.asarray(C, dtype=float)

    if C.shape != (6, 6):
        raise ValueError(
            "constitutive.C_hom_model_MPa must have shape (6, 6)."
        )

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit HOST directional plate -> beam reduction."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "reference_results/4pb/"
            "Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1/"
            "source_case_manifest.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "verification/beam_reduction/"
            "plate_to_beam_audit.json"
        ),
    )

    parser.add_argument("--length", type=float, default=240.0)
    parser.add_argument("--width", type=float, default=60.0)
    parser.add_argument("--core-height", type=float, default=60.0)
    parser.add_argument("--face-bottom", type=float, default=1.0)
    parser.add_argument("--face-top", type=float, default=1.0)

    parser.add_argument("--face-E", type=float, default=70000.0)
    parser.add_argument("--face-nu", type=float, default=0.33)

    parser.add_argument("--support-span", type=float, default=200.0)
    parser.add_argument("--load-span", type=float, default=80.0)
    parser.add_argument("--load", type=float, default=1000.0)

    parser.add_argument("--quadrature-order", type=int, default=16)

    # Known comparison values can be overridden without changing the script.
    parser.add_argument(
        "--expected-host-midspan",
        type=float,
        default=0.019197035266,
        help="Current HOST |w_mid| used only as an implementation cross-check.",
    )
    parser.add_argument(
        "--fe-midspan",
        type=float,
        default=0.0201772058725,
        help="Clean section-distributed 3-D FE |w_mid| for comparison.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Manifest not found: {}".format(manifest_path)
        )

    manifest = load_manifest(manifest_path)

    C_core = np.asarray(
        manifest["constitutive"]["C_hom_model_MPa"],
        dtype=float,
    )
    C_face = isotropic_stiffness_3d(
        args.face_E,
        args.face_nu,
    )

    matrices = build_section_matrices(
        C_face=C_face,
        C_core=C_core,
        width=args.width,
        h_core=args.core_height,
        h_face_bottom=args.face_bottom,
        h_face_top=args.face_top,
        quadrature_order=args.quadrature_order,
    )

    names = (
        "beam_free_stress",
        "plate_cylindrical",
        "plate_local_relaxed",
    )

    K4_by_name = {
        "beam_free_stress": matrices["K4_beam_free_stress"],
        "plate_cylindrical": matrices["K4_plate_cylindrical"],
        "plate_local_relaxed": matrices["K4_plate_local_relaxed"],
    }

    results = {}

    for name in names:
        K4 = K4_by_name[name]
        K3 = condense_zero_axial_force(K4)

        response = solve_four_point_midspan(
            K3,
            length=args.length,
            support_span=args.support_span,
            load_span=args.load_span,
            total_load=args.load,
        )

        results[name] = {
            "K4": matrix_to_list(K4),
            "K3_N_zero": matrix_to_list(K3),
            "K4_eigenvalues": vector_to_list(np.linalg.eigvalsh(K4)),
            "K3_eigenvalues": vector_to_list(np.linalg.eigvalsh(K3)),
            "response": response,
        }

    w_beam = abs(results["beam_free_stress"]["response"]["w_mid_mm"])
    w_cyl = abs(results["plate_cylindrical"]["response"]["w_mid_mm"])
    w_local = abs(results["plate_local_relaxed"]["response"]["w_mid_mm"])

    expected_host = abs(float(args.expected_host_midspan))
    fe_midspan = abs(float(args.fe_midspan))

    beam_vs_expected = relative_change(w_beam, expected_host)
    beam_vs_fe = relative_change(w_beam, fe_midspan)
    cyl_vs_beam = relative_change(w_cyl, w_beam)
    local_vs_cyl = relative_change(w_local, w_cyl)
    local_vs_beam = relative_change(w_local, w_beam)

    coupling_norm = float(
        matrices["primary_secondary_coupling_norm"][0]
    )

    # Decision logic is deliberately conservative.
    independent_beam_matches = abs(beam_vs_expected) <= 5.0e-3
    local_relaxation_small = abs(local_vs_cyl) <= 5.0e-3

    if not independent_beam_matches:
        decision = (
            "INCONCLUSIVE: the independent reconstruction does not reproduce "
            "the current HOST beam response closely enough. Audit the exact "
            "beam implementation/warping inputs before interpreting plate-to-"
            "beam effects."
        )
    elif local_relaxation_small:
        decision = (
            "REDUCTION_NOT_PRIMARY: the current free-stress beam reduction is "
            "reproduced, while the y-invariant secondary plate relaxation is "
            "small. The 1-D reduction is therefore unlikely to explain the "
            "remaining FE deflection gap. Proceed to slenderness/mechanism "
            "diagnostics and targeted theory refinement."
        )
    else:
        decision = (
            "SECONDARY_COUPLING_MATTERS: the local y-invariant secondary-field "
            "relaxation is appreciable. Before adding new kinematics, derive "
            "and solve the exact coupled 1-D y-invariant plate BVP for "
            "(u0, w0, beta_x, v0, beta_y)."
        )

    report = {
        "schema": "host_plate_to_beam_reduction_audit_v1",
        "manifest": str(manifest_path),
        "voigt_order": list(VOIGT),
        "engineering_shear": True,
        "geometry": {
            "length_mm": float(args.length),
            "width_mm": float(args.width),
            "core_height_mm": float(args.core_height),
            "face_bottom_mm": float(args.face_bottom),
            "face_top_mm": float(args.face_top),
            "total_height_mm": float(
                args.core_height + args.face_bottom + args.face_top
            ),
            "support_span_mm": float(args.support_span),
            "load_span_mm": float(args.load_span),
            "total_load_N": float(args.load),
        },
        "face_material": {
            "E_MPa": float(args.face_E),
            "nu": float(args.face_nu),
        },
        "directional_generalized_order": list(DIR12_LABELS),
        "index_sets": {
            "primary_x": list(IDX_PRIMARY_X),
            "primary_x_labels": [
                DIR12_LABELS[i] for i in IDX_PRIMARY_X
            ],
            "secondary_y_invariant": list(IDX_SECONDARY_YINV),
            "secondary_y_invariant_labels": [
                DIR12_LABELS[i] for i in IDX_SECONDARY_YINV
            ],
            "kinematically_zero_y_invariant": list(IDX_ZERO_YINV),
            "kinematically_zero_y_invariant_labels": [
                DIR12_LABELS[i] for i in IDX_ZERO_YINV
            ],
        },
        "shear_moduli": {
            "G_xz_core_MPa": float(matrices["shear_moduli_MPa"][0]),
            "G_yz_core_MPa": float(matrices["shear_moduli_MPa"][1]),
            "G_xz_face_MPa": float(matrices["shear_moduli_MPa"][2]),
            "G_yz_face_MPa": float(matrices["shear_moduli_MPa"][3]),
        },
        "K12_plate": matrix_to_list(matrices["K12_plate"]),
        "primary_secondary_coupling": {
            "normalized_frobenius": coupling_norm,
            "K_AB": matrix_to_list(
                matrices["KAB_primary_secondary"]
            ),
            "K_BB": matrix_to_list(
                matrices["KBB_secondary"]
            ),
        },
        "models": results,
        "comparisons": {
            "expected_current_HOST_abs_w_mm": expected_host,
            "clean_3D_FE_abs_w_mm": fe_midspan,
            "independent_beam_vs_expected_HOST_relative": beam_vs_expected,
            "independent_beam_vs_FE_relative": beam_vs_fe,
            "plate_cylindrical_vs_beam_relative": cyl_vs_beam,
            "plate_local_relaxed_vs_cylindrical_relative": local_vs_cyl,
            "plate_local_relaxed_vs_beam_relative": local_vs_beam,
        },
        "interpretation": {
            "independent_beam_matches_expected_within_0p5pct": bool(
                independent_beam_matches
            ),
            "local_secondary_relaxation_below_0p5pct": bool(
                local_relaxation_small
            ),
            "decision": decision,
            "important_note": (
                "plate_local_relaxed is a local Schur-complement diagnostic, "
                "not the exact y-invariant plate BVP."
            ),
        },
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print()
    print("=" * 76)
    print("HOST PLATE -> BEAM REDUCTION AUDIT")
    print("=" * 76)
    print("Manifest                  :", manifest_path)
    print("Output                    :", output_path)
    print()

    print("Directional K12 index sets")
    print("  primary x               :", IDX_PRIMARY_X)
    print(
        "  primary labels          :",
        [DIR12_LABELS[i] for i in IDX_PRIMARY_X],
    )
    print("  secondary y-invariant   :", IDX_SECONDARY_YINV)
    print(
        "  secondary labels        :",
        [DIR12_LABELS[i] for i in IDX_SECONDARY_YINV],
    )
    print("  forced zero for d/dy=0  :", IDX_ZERO_YINV)
    print()

    print("Engineering shear moduli [MPa]")
    print(
        "  core Gxz / Gyz          : {:.9f} / {:.9f}".format(
            matrices["shear_moduli_MPa"][0],
            matrices["shear_moduli_MPa"][1],
        )
    )
    print(
        "  face Gxz / Gyz          : {:.9f} / {:.9f}".format(
            matrices["shear_moduli_MPa"][2],
            matrices["shear_moduli_MPa"][3],
        )
    )
    print()

    print("Normalized primary-secondary coupling")
    print("  ||K_AB||_F/sqrt(...)    : {:.9e}".format(coupling_norm))
    print()

    print("Four-point-bending midspan |w| [mm]")
    print("  current beam free-stress : {:.12e}".format(w_beam))
    print("  plate cylindrical        : {:.12e}".format(w_cyl))
    print("  plate local relaxed      : {:.12e}".format(w_local))
    print("  expected current HOST    : {:.12e}".format(expected_host))
    print("  clean 3-D FE             : {:.12e}".format(fe_midspan))
    print()

    print("Relative differences")
    print(
        "  independent beam / HOST : {:+.6f} %".format(
            100.0 * beam_vs_expected
        )
    )
    print(
        "  independent beam / FE   : {:+.6f} %".format(
            100.0 * beam_vs_fe
        )
    )
    print(
        "  cylindrical / beam      : {:+.6f} %".format(
            100.0 * cyl_vs_beam
        )
    )
    print(
        "  local relax / cylindrical: {:+.6f} %".format(
            100.0 * local_vs_cyl
        )
    )
    print()

    print("DECISION")
    print(" ", decision)
    print()
    print(
        "NOTE: 'plate local relaxed' is a constitutive diagnostic only; "
        "it is not the exact coupled y-invariant plate BVP."
    )
    print("=" * 76)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
