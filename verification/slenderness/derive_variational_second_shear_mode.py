#!/usr/bin/env python3
"""
Derive a physics-based second HOST shear/warping mode from a 1-D section
eigenproblem and test it in the full eight-state 4PB beam.

Motivation
----------
The FE-informed two-mode beam reduced the 4PB RMS error from ~2.47% to ~0.41%,
but the direct FE/SVD mode must not become part of the final theory.  Smooth
global and layerwise polynomial fits also required high order.

This script therefore derives f2(z) from the section properties alone.

Variational section problem
---------------------------
On the symmetric half-section z in [0,h/2], consider odd warping functions

    f(0) = 0,

with natural outer-surface condition

    G_xz(z) f'(z) = 0 at z=h/2,

and weak continuity of

    f
    and
    G_xz f'

at material interfaces.

The diagonal warping energies motivate the Sturm-Liouville problem

    int G_xz f' v' dz
        = lambda int E_x f v dz.

Here E_x and G_xz are taken from the same free-stress-condensed 3-D->beam
constitutive matrices used by HOST.  The full two-mode section integration
later includes the complete 2x2 condensed constitutive matrix, including any
normal/shear coupling; the eigenproblem is only used to choose a compact
physics-based shape.

To retain the current HOST one-mode theory exactly, the new mode is not simply
the first eigenfunction.  Instead it is the minimum-energy admissible mode
subject to

    int E_x f1 f2 dz = 0,

where f1 is the current material-informed HOST warping function.

Thus f2 is defined by the section itself, not fitted to the 4PB deflection or
to the FE residual profile.

Implementation
--------------
A linear 1-D FE discretization is used on the half thickness.  The constrained
generalized eigenproblem is solved in the null space of the f1-orthogonality
constraint.  The mode is then mirrored oddly, inserted into the exact K6/K5
two-mode beam section formulation, and used in the same eight-state
multiple-shooting 4PB solver as the FE-informed proof.

The FE/SVD residual is used only AFTER derivation as an independent diagnostic
of the resulting f2' shape.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh, null_space

from run_fe_informed_two_mode_beam import (
    condense_N_zero,
    f1_functions,
    free_stress_x_beam_stiffness,
    isotropic_stiffness_3d,
    solve_two_mode_4pb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--modes-json",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_profile_modes/"
            "shear_profile_modes.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/variational_second_shear_mode"
        ),
    )

    parser.add_argument(
        "--core-elements",
        type=int,
        default=240,
    )

    parser.add_argument(
        "--face-elements",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-segment-mm",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return math.sqrt(
        float(
            np.sum(
                weights * values * values
            )
        )
    )


def weighted_relative_rmse(
    reference: np.ndarray,
    approximation: np.ndarray,
    weights: np.ndarray,
) -> float:
    denominator = weighted_norm(reference, weights)

    if denominator <= 0.0:
        return float("nan")

    return (
        weighted_norm(
            np.asarray(reference) - np.asarray(approximation),
            weights,
        )
        / denominator
    )


def weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    total = float(np.sum(w))

    mx = float(np.sum(w * x) / total)
    my = float(np.sum(w * y) / total)

    dx = x - mx
    dy = y - my

    vx = float(np.sum(w * dx * dx) / total)
    vy = float(np.sum(w * dy * dy) / total)

    if vx <= 0.0 or vy <= 0.0:
        return float("nan")

    return (
        float(np.sum(w * dx * dy) / total)
        / math.sqrt(vx * vy)
    )


def build_half_mesh(
    *,
    h: float,
    hc: float,
    n_core: int,
    n_face: int,
) -> np.ndarray:
    if n_core < 2 or n_face < 1:
        raise ValueError("Need n_core>=2 and n_face>=1.")

    zi = 0.5 * float(hc)
    zs = 0.5 * float(h)

    core = np.linspace(
        0.0,
        zi,
        int(n_core) + 1,
    )

    face = np.linspace(
        zi,
        zs,
        int(n_face) + 1,
    )[1:]

    return np.concatenate(
        (core, face)
    )


def assemble_eigen_matrices(
    nodes: np.ndarray,
    *,
    hc: float,
    Cb_core: np.ndarray,
    Cb_face: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K = int G_xz f' v' dz
    M = int E_x  f  v  dz
    """
    nodes = np.asarray(nodes, dtype=float)

    n = len(nodes)

    K = np.zeros((n, n), dtype=float)
    M = np.zeros((n, n), dtype=float)

    zi = 0.5 * float(hc)

    for e in range(n - 1):
        z0 = float(nodes[e])
        z1 = float(nodes[e + 1])
        le = z1 - z0
        zm = 0.5 * (z0 + z1)

        Cb = (
            Cb_core
            if zm < zi
            else Cb_face
        )

        Eeff = float(Cb[0, 0])
        Geff = float(Cb[1, 1])

        Ke = (
            Geff
            / le
            * np.asarray(
                [
                    [1.0, -1.0],
                    [-1.0, 1.0],
                ],
                dtype=float,
            )
        )

        Me = (
            Eeff
            * le
            / 6.0
            * np.asarray(
                [
                    [2.0, 1.0],
                    [1.0, 2.0],
                ],
                dtype=float,
            )
        )

        ix = np.asarray(
            [e, e + 1],
            dtype=int,
        )

        K[np.ix_(ix, ix)] += Ke
        M[np.ix_(ix, ix)] += Me

    return K, M


def solve_constrained_mode(
    nodes: np.ndarray,
    K: np.ndarray,
    M: np.ndarray,
    f1_values: np.ndarray,
) -> dict:
    """
    Enforce:
        f2(0)=0,
        f1^T M f2 = 0.

    The natural surface condition is already contained in the weak form.
    """
    nodes = np.asarray(nodes, dtype=float)
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    f1_values = np.asarray(f1_values, dtype=float)

    # Odd symmetry: f(0)=0.
    free = np.arange(1, len(nodes), dtype=int)

    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    f1f = f1_values[free]

    # f1(0)=0, so no eliminated-node contribution exists.
    c = Mf @ f1f

    Z = null_space(
        c.reshape(1, -1),
        rcond=1.0e-12,
    )

    Kr = Z.T @ Kf @ Z
    Mr = Z.T @ Mf @ Z

    eigenvalues, eigenvectors = eigh(
        Kr,
        Mr,
    )

    positive = np.where(
        eigenvalues > 1.0e-12
    )[0]

    if len(positive) == 0:
        raise RuntimeError(
            "No positive constrained section eigenvalue found."
        )

    index = int(positive[0])

    lam = float(
        eigenvalues[index]
    )

    vf = Z @ eigenvectors[:, index]

    values = np.zeros(
        len(nodes),
        dtype=float,
    )

    values[free] = vf

    # Normalize using max absolute element slope = 1.
    slopes = np.diff(values) / np.diff(nodes)

    scale = float(
        np.max(
            np.abs(slopes)
        )
    )

    if scale <= 0.0:
        raise RuntimeError("Degenerate constrained mode.")

    values /= scale
    slopes /= scale

    orthogonality = float(
        f1_values @ M @ values
    )

    modal_mass = float(
        values @ M @ values
    )

    modal_shear = float(
        values @ K @ values
    )

    return {
        "lambda_per_mm2": lam,
        "characteristic_length_mm": 1.0 / math.sqrt(lam),
        "values": values,
        "slopes": slopes,
        "orthogonality_E_inner_product": orthogonality,
        "modal_E_norm": modal_mass,
        "modal_G_gradient_norm": modal_shear,
    }


class PiecewiseLinearOddMode:
    def __init__(
        self,
        nodes_half: np.ndarray,
        values_half: np.ndarray,
    ) -> None:
        self.nodes = np.asarray(
            nodes_half,
            dtype=float,
        )
        self.values = np.asarray(
            values_half,
            dtype=float,
        )

        if len(self.nodes) != len(self.values):
            raise ValueError("Node/value size mismatch.")

        self.slopes = (
            np.diff(self.values)
            / np.diff(self.nodes)
        )

        self.zs = float(
            self.nodes[-1]
        )

    def _element(self, za: float) -> int:
        za = float(za)

        if za < -1.0e-12 or za > self.zs + 1.0e-12:
            raise ValueError("z outside section.")

        if za >= self.zs:
            return len(self.slopes) - 1

        return int(
            np.searchsorted(
                self.nodes,
                za,
                side="right",
            )
            - 1
        )

    def f(self, z: float) -> float:
        z = float(z)

        if z < 0.0:
            return -self.f(-z)

        e = self._element(z)

        z0 = float(self.nodes[e])
        v0 = float(self.values[e])

        return (
            v0
            + float(self.slopes[e])
            * (z - z0)
        )

    def df(self, z: float) -> float:
        za = abs(float(z))
        e = self._element(za)

        return float(
            self.slopes[e]
        )


def build_K6_piecewise(
    *,
    width: float,
    nodes_half: np.ndarray,
    hc: float,
    Cb_core: np.ndarray,
    Cb_face: np.ndarray,
    f1,
    df1,
    f2: PiecewiseLinearOddMode,
) -> np.ndarray:
    """
    Exact-enough section integration on every eigenmesh segment.

    Generalized order:
        [eps0, kappa, chi1, chi2, beta1, beta2].
    """
    nodes = np.asarray(
        nodes_half,
        dtype=float,
    )

    zi = 0.5 * float(hc)

    gauss_x = np.asarray(
        [-1.0 / math.sqrt(3.0), +1.0 / math.sqrt(3.0)],
        dtype=float,
    )

    gauss_w = np.asarray(
        [1.0, 1.0],
        dtype=float,
    )

    K = np.zeros(
        (6, 6),
        dtype=float,
    )

    # Integrate positive half and mirror explicitly.
    for e in range(len(nodes) - 1):
        a = float(nodes[e])
        b = float(nodes[e + 1])
        jac = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        zm = mid

        Cb = (
            Cb_core
            if zm < zi
            else Cb_face
        )

        for sign in (-1.0, +1.0):
            for point, weight in zip(
                gauss_x,
                gauss_w,
            ):
                zp = (
                    mid
                    + jac * float(point)
                )

                z = sign * zp

                B = np.asarray(
                    [
                        [
                            1.0,
                            z,
                            float(f1(z)),
                            float(f2.f(z)),
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(df1(z)),
                            float(f2.df(z)),
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


def sample_derivative(
    mode: PiecewiseLinearOddMode,
    z_values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            mode.df(float(z))
            for z in np.asarray(
                z_values,
                dtype=float,
            )
        ],
        dtype=float,
    )


def main() -> int:
    args = parse_args()

    audit_path = args.audit.resolve()
    slenderness_path = args.slenderness.resolve()
    modes_path = args.modes_json.resolve()
    output_dir = args.output_dir.resolve()

    for path in (
        audit_path,
        slenderness_path,
        modes_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    audit = load_json(audit_path)
    slenderness = load_json(slenderness_path)
    modes = load_json(modes_path)

    source_manifest_path = Path(
        audit["manifest"]
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

    coupling_core = float(
        abs(Cb_core[0, 1])
        / math.sqrt(
            Cb_core[0, 0]
            * Cb_core[1, 1]
        )
    )

    coupling_face = float(
        abs(Cb_face[0, 1])
        / math.sqrt(
            Cb_face[0, 0]
            * Cb_face[1, 1]
        )
    )

    rows_source = [
        row
        for row in slenderness["rows"]
        if row.get("FE_3D_abs_w_mid_mm") not in ("", None)
    ]

    if not rows_source:
        raise RuntimeError(
            "No FE-populated slenderness rows."
        )

    first = rows_source[0]

    h = float(
        first["total_height_mm"]
    )

    hc = float(
        first["core_height_mm"]
    )

    width = float(
        first["width_mm"]
    )

    G_core = float(
        audit["shear_moduli"]["G_xz_core_MPa"]
    )

    G_face = float(
        audit["shear_moduli"]["G_xz_face_MPa"]
    )

    f1, df1 = f1_functions(
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
    )

    nodes = build_half_mesh(
        h=h,
        hc=hc,
        n_core=int(args.core_elements),
        n_face=int(args.face_elements),
    )

    K_eig, M_eig = assemble_eigen_matrices(
        nodes,
        hc=hc,
        Cb_core=Cb_core,
        Cb_face=Cb_face,
    )

    f1_values = np.asarray(
        [
            float(f1(z))
            for z in nodes
        ],
        dtype=float,
    )

    mode_result = solve_constrained_mode(
        nodes,
        K_eig,
        M_eig,
        f1_values,
    )

    mode = PiecewiseLinearOddMode(
        nodes,
        np.asarray(
            mode_result["values"],
            dtype=float,
        ),
    )

    K6 = build_K6_piecewise(
        width=width,
        nodes_half=nodes,
        hc=hc,
        Cb_core=Cb_core,
        Cb_face=Cb_face,
        f1=f1,
        df1=df1,
        f2=mode,
    )

    K5 = condense_N_zero(
        K6
    )

    eig6 = np.linalg.eigvalsh(K6)
    eig5 = np.linalg.eigvalsh(K5)

    if float(np.min(eig6)) <= 0.0:
        raise RuntimeError(
            "K6 is not positive definite."
        )

    if float(np.min(eig5)) <= 0.0:
        raise RuntimeError(
            "K5 is not positive definite."
        )

    beta_block = K5[3:5, 3:5]

    local_alpha = float(
        (
            beta_block[0, 0]
            - beta_block[0, 1] ** 2 / beta_block[1, 1]
        )
        / beta_block[0, 0]
    )

    # Independent FE/SVD profile comparison.
    residual = modes["residual_svd"]

    z_fe = np.asarray(
        residual["common_residual_mode_z_mm"],
        dtype=float,
    )

    target_df = np.asarray(
        residual["common_residual_mode"],
        dtype=float,
    )

    profile_key = sorted(
        modes["profiles"],
        key=lambda value: int(value),
    )[0]

    weights_fe = np.asarray(
        modes["profiles"][profile_key]["area_half_mm2"],
        dtype=float,
    )

    eigen_df = sample_derivative(
        mode,
        z_fe,
    )

    # SVD sign/scale are arbitrary: fit one scalar amplitude only.
    amplitude = float(
        np.sum(
            weights_fe
            * eigen_df
            * target_df
        )
        / np.sum(
            weights_fe
            * eigen_df
            * eigen_df
        )
    )

    approximation = (
        amplitude * eigen_df
    )

    profile_rmse = weighted_relative_rmse(
        target_df,
        approximation,
        weights_fe,
    )

    profile_corr = weighted_correlation(
        target_df,
        approximation,
        weights_fe,
    )

    result_rows = []

    for row in rows_source:
        solution = solve_two_mode_4pb(
            K5,
            length=float(row["length_mm"]),
            support_left=float(row["support_left_mm"]),
            support_right=float(row["support_right_mm"]),
            load_left=float(row["load_left_mm"]),
            load_right=float(row["load_right_mm"]),
            total_load=1000.0,
            max_segment_mm=float(args.max_segment_mm),
        )

        w_one = float(
            row["HOST_1D_abs_w_mid_mm"]
        )

        w_two = abs(
            float(
                solution["midspan"][0]
            )
        )

        w_fe = float(
            row["FE_3D_abs_w_mid_mm"]
        )

        result_rows.append(
            {
                "n_shear": int(
                    row["n_shear"]
                ),
                "a_over_h": float(
                    row["a_over_h"]
                ),
                "HOST_one_mode_abs_w_mid_mm": w_one,
                "HOST_variational_two_mode_abs_w_mid_mm": w_two,
                "FE_abs_w_mid_mm": w_fe,
                "one_mode_error_relative": (
                    (w_one - w_fe) / w_fe
                ),
                "two_mode_error_relative": (
                    (w_two - w_fe) / w_fe
                ),
                "two_minus_one_relative_to_one": (
                    (w_two - w_one) / w_one
                ),
                "right_end_M": float(
                    solution["right_end_M"]
                ),
                "right_end_V": float(
                    solution["right_end_V"]
                ),
                "right_end_H1": float(
                    solution["right_end_H1"]
                ),
                "right_end_H2": float(
                    solution["right_end_H2"]
                ),
            }
        )

    one_errors = np.asarray(
        [
            row["one_mode_error_relative"]
            for row in result_rows
        ],
        dtype=float,
    )

    two_errors = np.asarray(
        [
            row["two_mode_error_relative"]
            for row in result_rows
        ],
        dtype=float,
    )

    rms_one = float(
        np.sqrt(
            np.mean(
                one_errors ** 2
            )
        )
    )

    rms_two = float(
        np.sqrt(
            np.mean(
                two_errors ** 2
            )
        )
    )

    max_one = float(
        np.max(
            np.abs(one_errors)
        )
    )

    max_two = float(
        np.max(
            np.abs(two_errors)
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "variational_second_shear_mode.json"
    )

    csv_path = (
        output_dir
        / "variational_two_mode_beam.csv"
    )

    profile_csv = (
        output_dir
        / "variational_mode_profile.csv"
    )

    report = {
        "schema": "host_variational_second_shear_mode_v1",
        "derivation": {
            "weak_problem": (
                "int G_xz f2' v' dz = lambda int E_x f2 v dz"
            ),
            "symmetry": "odd f2 on full thickness; f2(0)=0 on half section",
            "orthogonality": (
                "int E_x f1 f2 dz = 0"
            ),
            "outer_surface": (
                "natural G_xz f2' = 0"
            ),
        },
        "section": {
            "h_mm": h,
            "hc_mm": hc,
            "width_mm": width,
            "core_elements_half": int(
                args.core_elements
            ),
            "face_elements_half": int(
                args.face_elements
            ),
            "Cb_core": Cb_core.tolist(),
            "Cb_face": Cb_face.tolist(),
            "normal_shear_coupling_ratio_core": coupling_core,
            "normal_shear_coupling_ratio_face": coupling_face,
            "K6": K6.tolist(),
            "K5_N_zero": K5.tolist(),
            "K6_eigenvalues": eig6.tolist(),
            "K5_eigenvalues": eig5.tolist(),
            "local_beta_block": beta_block.tolist(),
            "local_beta_condensed_alpha": local_alpha,
        },
        "mode": {
            "lambda_per_mm2": float(
                mode_result["lambda_per_mm2"]
            ),
            "characteristic_length_mm": float(
                mode_result["characteristic_length_mm"]
            ),
            "characteristic_length_over_h": float(
                mode_result["characteristic_length_mm"] / h
            ),
            "E_inner_product_with_f1": float(
                mode_result["orthogonality_E_inner_product"]
            ),
            "nodes_half_mm": nodes.tolist(),
            "values_half": np.asarray(
                mode_result["values"],
                dtype=float,
            ).tolist(),
            "slopes_half": np.asarray(
                mode_result["slopes"],
                dtype=float,
            ).tolist(),
        },
        "FE_residual_profile_diagnostic": {
            "amplitude": amplitude,
            "relative_rmse": profile_rmse,
            "weighted_correlation": profile_corr,
        },
        "beam_results": result_rows,
        "summary": {
            "RMS_one_mode_relative_error": rms_one,
            "RMS_two_mode_relative_error": rms_two,
            "max_one_mode_relative_error": max_one,
            "max_two_mode_relative_error": max_two,
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

    with profile_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "z_half_mm",
                "f2",
                "df2_element_right",
            )
        )

        slopes = np.asarray(
            mode_result["slopes"],
            dtype=float,
        )

        values = np.asarray(
            mode_result["values"],
            dtype=float,
        )

        for i, z_value in enumerate(nodes):
            if i < len(slopes):
                slope = slopes[i]
            else:
                slope = slopes[-1]

            writer.writerow(
                (
                    float(z_value),
                    float(values[i]),
                    float(slope),
                )
            )

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(
            target_df,
            z_fe / h,
            marker="o",
            markersize=3,
            label="FE/SVD residual",
        )

        ax.plot(
            approximation,
            z_fe / h,
            linewidth=2.0,
            label="variational f2' (scaled)",
        )

        ax.set_xlabel("second shear-mode derivative")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "variational_f2_vs_FE_residual.png"
        )

        fig.savefig(
            path,
            dpi=180,
        )

        plt.close(fig)
        plot_paths.append(
            str(path)
        )

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
            label="current f1'",
        )

        ax.plot(
            [mode.df(z) for z in z_plot],
            z_plot / h,
            label="variational f2'",
        )

        ax.set_xlabel("normalized mode derivative")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "variational_f1_f2_derivatives.png"
        )

        fig.savefig(
            path,
            dpi=180,
        )

        plt.close(fig)
        plot_paths.append(
            str(path)
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        a = np.asarray(
            [
                row["a_over_h"]
                for row in result_rows
            ],
            dtype=float,
        )

        ax.plot(
            a,
            100.0 * one_errors,
            marker="o",
            label="one-mode HOST",
        )

        ax.plot(
            a,
            100.0 * two_errors,
            marker="s",
            label="variational two-mode HOST",
        )

        ax.axhline(
            0.0,
            linewidth=1.0,
        )

        ax.set_xlabel("a/h")
        ax.set_ylabel("deflection error [%]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "variational_two_mode_error.png"
        )

        fig.savefig(
            path,
            dpi=180,
        )

        plt.close(fig)
        plot_paths.append(
            str(path)
        )

    except Exception as exc:
        plot_paths.append(
            "Plotting skipped: {}".format(exc)
        )

    print()
    print("=" * 126)
    print("HOST VARIATIONAL SECOND SHEAR-MODE AUDIT")
    print("=" * 126)
    print(
        "Section diagonal coupling |C15|/sqrt(C11*C55), core : {:.6e}".format(
            coupling_core
        )
    )
    print(
        "Section diagonal coupling |C15|/sqrt(C11*C55), face : {:.6e}".format(
            coupling_face
        )
    )
    print(
        "Constrained eigenvalue lambda [1/mm^2]                : {:.9e}".format(
            float(
                mode_result["lambda_per_mm2"]
            )
        )
    )
    print(
        "Characteristic length ell [mm]                       : {:.9f}".format(
            float(
                mode_result["characteristic_length_mm"]
            )
        )
    )
    print(
        "ell/h                                                 : {:.9f}".format(
            float(
                mode_result["characteristic_length_mm"] / h
            )
        )
    )
    print(
        "E-inner-product f1/f2 orthogonality residual          : {:.3e}".format(
            float(
                mode_result["orthogonality_E_inner_product"]
            )
        )
    )
    print(
        "Local beta-block condensed alpha                      : {:.9f}".format(
            local_alpha
        )
    )
    print()
    print(
        "Independent FE residual profile comparison:"
    )
    print(
        "  relative RMSE                                       : {:.6%}".format(
            profile_rmse
        )
    )
    print(
        "  weighted correlation                                : {:.9f}".format(
            profile_corr
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

    for row in result_rows:
        print(
            "{:6d} {:8.3f} {:13.8f} {:13.8f} {:13.8f} {:10.3%} {:10.3%}".format(
                int(
                    row["n_shear"]
                ),
                float(
                    row["a_over_h"]
                ),
                float(
                    row["HOST_one_mode_abs_w_mid_mm"]
                ),
                float(
                    row["HOST_variational_two_mode_abs_w_mid_mm"]
                ),
                float(
                    row["FE_abs_w_mid_mm"]
                ),
                float(
                    row["one_mode_error_relative"]
                ),
                float(
                    row["two_mode_error_relative"]
                ),
            )
        )

    print()
    print(
        "RMS error 1mode -> 2mode                              : {:.4f}% -> {:.4f}%".format(
            100.0 * rms_one,
            100.0 * rms_two,
        )
    )
    print(
        "Max error 1mode -> 2mode                              : {:.4f}% -> {:.4f}%".format(
            100.0 * max_one,
            100.0 * max_two,
        )
    )
    print()
    print("Wrote:")
    print(" ", json_path)
    print(" ", csv_path)
    print(" ", profile_csv)
    for path in plot_paths:
        print(" ", path)
    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
