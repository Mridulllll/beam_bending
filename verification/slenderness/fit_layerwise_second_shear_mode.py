#!/usr/bin/env python3
"""
Fit a compact layer-aware analytic second shear mode and test it in the full
two-mode HOST beam.

This is the next step after the FE-informed proof-of-mechanism.  The previous
global-polynomial family

    f2' = f1' q(xi)

required degree 8 and still left ~6.5% residual-mode error.  The FE/SVD mode,
however, shows a strong face/core derivative jump consistent with inverse
shear-modulus weighting.  This script therefore fits the multiplier q
piecewise while preserving physically useful conditions.

Mode family
-----------
Let

    xi = 2 z / h
    p  = 1 - xi^2
    r  = Gc / G(z)

and use

    f2'(z) = r p q(z).

For |z| <= hc/2,

    q_c(eta) = c0 + c2 eta^2 + c4 eta^4 + ...
    eta = |z| / (hc/2).

For hc/2 < |z| <= h/2,

    q_f(t) = q_c(1) + d1 t + d2 t^2 + ...
    t = (|z|-hc/2) / ((h-hc)/2).

Therefore:

* f2' is even and f2 is odd;
* f2'(+-h/2)=0 through p;
* q is continuous at the interface;
* G(z) f2'(z) is continuous at the interface for piecewise-constant G,
  because G(z) r = Gc;
* face/core strain may jump, as it should for a compliant core bonded to
  stiff faces.

After least-squares fitting, the candidate is projected orthogonal to the
current f1' mode using the FE-profile area inner product.  This is equivalent
to shifting c0 by a constant and therefore preserves all conditions above.

The lowest-parameter candidate satisfying --target-rmse is selected and then
inserted into the complete eight-state two-mode beam from
run_fe_informed_two_mode_beam.py.  No empirical alpha_G is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.integrate import quad

from run_fe_informed_two_mode_beam import (
    build_K6,
    condense_N_zero,
    f1_functions,
    free_stress_x_beam_stiffness,
    isotropic_stiffness_3d,
    solve_two_mode_4pb,
)


DEFAULT_CORE_DEGREES = (2, 4, 6, 8)
DEFAULT_FACE_DEGREES = (1, 2, 3)


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
            "gyroid_rd12_initial/layerwise_second_shear_mode"
        ),
    )

    parser.add_argument(
        "--core-degrees",
        default=",".join(str(v) for v in DEFAULT_CORE_DEGREES),
    )

    parser.add_argument(
        "--face-degrees",
        default=",".join(str(v) for v in DEFAULT_FACE_DEGREES),
    )

    parser.add_argument(
        "--target-rmse",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max-segment-mm",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def parse_ints(text: str) -> Tuple[int, ...]:
    return tuple(
        int(token.strip())
        for token in str(text).split(",")
        if token.strip()
    )


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


def current_f1_prime(
    z: np.ndarray,
    *,
    h: float,
    hc: float,
    G_core: float,
    G_face: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)

    xi = 2.0 * z / float(h)
    p = 1.0 - xi * xi

    core = np.abs(z) <= 0.5 * float(hc) + 1.0e-12

    result = np.empty_like(z)
    result[core] = p[core]
    result[~core] = (
        float(G_core)
        / float(G_face)
        * p[~core]
    )

    return result


def build_design_matrix(
    z: np.ndarray,
    *,
    h: float,
    hc: float,
    G_core: float,
    G_face: float,
    core_degree: int,
    face_degree: int,
) -> Tuple[np.ndarray, List[str]]:
    z = np.asarray(z, dtype=float)

    za = np.abs(z)
    zi = 0.5 * float(hc)
    hf = 0.5 * (float(h) - float(hc))

    xi = 2.0 * z / float(h)
    p = 1.0 - xi * xi

    is_core = za <= zi + 1.0e-12

    eta = np.zeros_like(z)
    eta[is_core] = za[is_core] / zi

    t = np.zeros_like(z)
    t[~is_core] = (
        za[~is_core] - zi
    ) / hf

    r = np.where(
        is_core,
        1.0,
        float(G_core) / float(G_face),
    )

    base = r * p

    columns = []
    names = []

    # Core coefficients c0,c2,... also define q_face(0)=q_core(1).
    for power in range(0, int(core_degree) + 1, 2):
        column = np.empty_like(z)

        column[is_core] = (
            base[is_core]
            * eta[is_core] ** power
        )

        # At the interface eta=1, so every core coefficient contributes
        # equally to q_core(1), hence also to the face intercept.
        column[~is_core] = base[~is_core]

        columns.append(column)
        names.append("c{}".format(power))

    # Additional face shape starts from zero at the interface.
    for power in range(1, int(face_degree) + 1):
        column = np.zeros_like(z)
        column[~is_core] = (
            base[~is_core]
            * t[~is_core] ** power
        )

        columns.append(column)
        names.append("d{}".format(power))

    return np.column_stack(columns), names


def fit_candidate(
    target: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    f1: np.ndarray,
    *,
    h: float,
    hc: float,
    G_core: float,
    G_face: float,
    core_degree: int,
    face_degree: int,
) -> dict:
    A, names = build_design_matrix(
        z,
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
        core_degree=core_degree,
        face_degree=face_degree,
    )

    sqrt_w = np.sqrt(weights)

    coefficients, _, _, _ = np.linalg.lstsq(
        A * sqrt_w[:, None],
        target * sqrt_w,
        rcond=None,
    )

    fitted = A @ coefficients

    # Enforce exact area-L2 orthogonality against f1'.  c0 is the global
    # constant q shift, so modifying it by -lambda performs this projection
    # without violating interface continuity or surface conditions.
    f1_norm_sq = float(
        np.sum(
            weights * f1 * f1
        )
    )

    lam = float(
        np.sum(
            weights * fitted * f1
        )
        / f1_norm_sq
    )

    index_c0 = names.index("c0")
    coefficients[index_c0] -= lam

    fitted = A @ coefficients

    scale = float(np.max(np.abs(fitted)))

    if scale <= 0.0:
        raise RuntimeError("Degenerate fitted second mode.")

    coefficients_normalized = coefficients / scale
    fitted_normalized = fitted / scale

    # The SVD vector's scale/sign are arbitrary.  Compare after optimal target
    # amplitude so the RMSE measures shape.
    amplitude = float(
        np.sum(
            weights * fitted_normalized * target
        )
        / np.sum(
            weights * fitted_normalized * fitted_normalized
        )
    )

    approximation = amplitude * fitted_normalized

    rmse = weighted_relative_rmse(
        target,
        approximation,
        weights,
    )

    corr = weighted_correlation(
        target,
        approximation,
        weights,
    )

    parameter_count = len(names)

    return {
        "core_degree": int(core_degree),
        "face_degree": int(face_degree),
        "parameter_count": int(parameter_count),
        "names": names,
        "coefficients_maxabs_normalized": {
            name: float(value)
            for name, value in zip(
                names,
                coefficients_normalized,
            )
        },
        "target_amplitude_for_shape_fit": amplitude,
        "relative_rmse": float(rmse),
        "weighted_correlation": float(corr),
        "orthogonality_lambda_before_normalization": float(lam),
        "fitted_df2_normalized": fitted_normalized.tolist(),
        "shape_approximation": approximation.tolist(),
    }


class LayerwiseAnalyticMode:
    def __init__(
        self,
        coefficients: Dict[str, float],
        *,
        h: float,
        hc: float,
        G_core: float,
        G_face: float,
    ) -> None:
        self.coefficients = {
            str(key): float(value)
            for key, value in coefficients.items()
        }

        self.h = float(h)
        self.hc = float(hc)
        self.zi = 0.5 * self.hc
        self.hf = 0.5 * (self.h - self.hc)
        self.zs = 0.5 * self.h
        self.r_face = float(G_core) / float(G_face)

        self.core_terms = sorted(
            (
                int(name[1:]),
                value,
            )
            for name, value in self.coefficients.items()
            if name.startswith("c")
        )

        self.face_terms = sorted(
            (
                int(name[1:]),
                value,
            )
            for name, value in self.coefficients.items()
            if name.startswith("d")
        )

        self.q_interface = sum(
            value
            for _, value in self.core_terms
        )

        self._f_interface = quad(
            lambda s: self.df(s),
            0.0,
            self.zi,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=100,
        )[0]

    def q_core(self, za: float) -> float:
        eta = float(za) / self.zi

        return sum(
            value * eta ** power
            for power, value in self.core_terms
        )

    def q_face(self, za: float) -> float:
        t = (
            float(za) - self.zi
        ) / self.hf

        return self.q_interface + sum(
            value * t ** power
            for power, value in self.face_terms
        )

    def df(self, z: float) -> float:
        z = float(z)
        za = abs(z)

        xi = 2.0 * za / self.h
        p = 1.0 - xi * xi

        if za <= self.zi:
            return p * self.q_core(za)

        if za <= self.zs + 1.0e-12:
            return (
                self.r_face
                * p
                * self.q_face(za)
            )

        raise ValueError("z lies outside section.")

    def f(self, z: float) -> float:
        z = float(z)

        if z < 0.0:
            return -self.f(-z)

        if z <= self.zi:
            return quad(
                lambda s: self.df(s),
                0.0,
                z,
                epsabs=1.0e-11,
                epsrel=1.0e-11,
                limit=100,
            )[0]

        return self._f_interface + quad(
            lambda s: self.df(s),
            self.zi,
            z,
            epsabs=1.0e-11,
            epsrel=1.0e-11,
            limit=100,
        )[0]

    def interface_diagnostics(self) -> dict:
        eps = 1.0e-9

        core_value = self.df(
            self.zi - eps
        )

        face_value = self.df(
            self.zi + eps
        )

        return {
            "df_core_side": float(core_value),
            "df_face_side": float(face_value),
            "df_core_over_face": float(
                core_value / face_value
            ),
            "q_core_interface": float(
                self.q_core(self.zi)
            ),
            "q_face_interface": float(
                self.q_face(self.zi)
            ),
        }


def main() -> int:
    args = parse_args()

    modes_path = args.modes_json.resolve()
    audit_path = args.audit.resolve()
    slenderness_path = args.slenderness.resolve()
    output_dir = args.output_dir.resolve()

    for path in (
        modes_path,
        audit_path,
        slenderness_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    modes = load_json(modes_path)
    audit = load_json(audit_path)
    slenderness = load_json(slenderness_path)

    residual = modes["residual_svd"]

    z = np.asarray(
        residual["common_residual_mode_z_mm"],
        dtype=float,
    )

    target = np.asarray(
        residual["common_residual_mode"],
        dtype=float,
    )

    profile_key = sorted(
        modes["profiles"],
        key=lambda value: int(value),
    )[0]

    profile = modes["profiles"][profile_key]

    weights = np.asarray(
        profile["area_half_mm2"],
        dtype=float,
    )

    source_rows = [
        row
        for row in slenderness["rows"]
        if row.get("FE_3D_abs_w_mid_mm") not in ("", None)
    ]

    if not source_rows:
        raise RuntimeError(
            "No FE-populated slenderness rows."
        )

    first = source_rows[0]

    h = float(first["total_height_mm"])
    hc = float(first["core_height_mm"])
    width = float(first["width_mm"])

    G_core = float(
        audit["shear_moduli"]["G_xz_core_MPa"]
    )

    G_face = float(
        audit["shear_moduli"]["G_xz_face_MPa"]
    )

    f1_profile = current_f1_prime(
        z,
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
    )

    candidate_rows = []

    for core_degree in parse_ints(args.core_degrees):
        if core_degree < 0 or core_degree % 2:
            raise ValueError(
                "Core degrees must be nonnegative even integers."
            )

        for face_degree in parse_ints(args.face_degrees):
            if face_degree < 0:
                raise ValueError(
                    "Face degrees must be nonnegative integers."
                )

            candidate_rows.append(
                fit_candidate(
                    target,
                    z,
                    weights,
                    f1_profile,
                    h=h,
                    hc=hc,
                    G_core=G_core,
                    G_face=G_face,
                    core_degree=core_degree,
                    face_degree=face_degree,
                )
            )

    # Prefer the fewest parameters that reaches the target; break ties by RMSE.
    acceptable = [
        row
        for row in candidate_rows
        if float(row["relative_rmse"]) <= float(args.target_rmse)
    ]

    if acceptable:
        selected = min(
            acceptable,
            key=lambda row: (
                int(row["parameter_count"]),
                float(row["relative_rmse"]),
            ),
        )
    else:
        selected = min(
            candidate_rows,
            key=lambda row: float(row["relative_rmse"]),
        )

    mode = LayerwiseAnalyticMode(
        selected["coefficients_maxabs_normalized"],
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
    )

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

    f1, df1 = f1_functions(
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
    )

    K6 = build_K6(
        width=width,
        h=h,
        hc=hc,
        Cb_core=Cb_core,
        Cb_face=Cb_face,
        f1=f1,
        df1=df1,
        f2=mode.f,
        df2=mode.df,
        quadrature_order=int(args.quadrature_order),
    )

    K5 = condense_N_zero(
        K6
    )

    eig6 = np.linalg.eigvalsh(K6)
    eig5 = np.linalg.eigvalsh(K5)

    if np.min(eig6) <= 0.0 or np.min(eig5) <= 0.0:
        raise RuntimeError(
            "Analytic two-mode section stiffness is not positive definite."
        )

    beta_block = K5[3:5, 3:5]

    local_alpha = float(
        (
            beta_block[0, 0]
            - beta_block[0, 1] ** 2 / beta_block[1, 1]
        )
        / beta_block[0, 0]
    )

    result_rows = []

    for row in source_rows:
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
                "n_shear": int(row["n_shear"]),
                "a_over_h": float(row["a_over_h"]),
                "HOST_one_mode_abs_w_mid_mm": w_one,
                "HOST_layerwise_two_mode_abs_w_mid_mm": w_two,
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

    summary = {
        "RMS_one_mode_relative_error": float(
            np.sqrt(
                np.mean(
                    one_errors ** 2
                )
            )
        ),
        "RMS_two_mode_relative_error": float(
            np.sqrt(
                np.mean(
                    two_errors ** 2
                )
            )
        ),
        "max_one_mode_relative_error": float(
            np.max(
                np.abs(one_errors)
            )
        ),
        "max_two_mode_relative_error": float(
            np.max(
                np.abs(two_errors)
            )
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "layerwise_second_shear_mode.json"
    )

    csv_path = (
        output_dir
        / "layerwise_second_shear_mode_candidates.csv"
    )

    results_csv_path = (
        output_dir
        / "layerwise_two_mode_beam.csv"
    )

    report = {
        "schema": "host_layerwise_second_shear_mode_v1",
        "definition": {
            "core_q": (
                "q_c(eta)=sum c_(2k)*eta^(2k), "
                "eta=|z|/(hc/2)"
            ),
            "face_q": (
                "q_f(t)=q_c(1)+sum d_j*t^j, "
                "t=(|z|-hc/2)/((h-hc)/2)"
            ),
            "f2_prime": (
                "(Gc/G(z))*(1-(2z/h)^2)*q(z)"
            ),
        },
        "source_residual": {
            "first_SVD_mode_fraction": float(
                residual["first_mode_explained_fraction"]
            ),
        },
        "geometry": {
            "h_mm": h,
            "hc_mm": hc,
            "width_mm": width,
        },
        "candidate_fits": candidate_rows,
        "selected": selected,
        "interface": mode.interface_diagnostics(),
        "section": {
            "K6": K6.tolist(),
            "K5_N_zero": K5.tolist(),
            "K6_eigenvalues": eig6.tolist(),
            "K5_eigenvalues": eig5.tolist(),
            "local_beta_block": beta_block.tolist(),
            "local_beta_condensed_alpha": local_alpha,
        },
        "beam_results": result_rows,
        "summary": summary,
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
            fieldnames=(
                "core_degree",
                "face_degree",
                "parameter_count",
                "relative_rmse",
                "weighted_correlation",
            ),
        )

        writer.writeheader()

        for row in sorted(
            candidate_rows,
            key=lambda item: (
                int(item["parameter_count"]),
                float(item["relative_rmse"]),
            ),
        ):
            writer.writerow(
                {
                    key: row[key]
                    for key in writer.fieldnames
                }
            )

    with results_csv_path.open(
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

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        selected_shape = np.asarray(
            selected["shape_approximation"],
            dtype=float,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(
            target,
            z / h,
            marker="o",
            markersize=3,
            label="FE/SVD residual",
        )

        ax.plot(
            selected_shape,
            z / h,
            linewidth=2.0,
            label="selected layerwise analytic",
        )

        ax.set_xlabel("second shear-mode derivative")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "selected_layerwise_mode_fit.png"
        )
        fig.savefig(
            path,
            dpi=180,
        )
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
            [df1(value) for value in z_plot],
            z_plot / h,
            label="f1'",
        )

        ax.plot(
            [mode.df(value) for value in z_plot],
            z_plot / h,
            label="analytic f2'",
        )

        ax.set_xlabel("normalized mode derivative")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "analytic_f1_f2_derivatives.png"
        )
        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(fig)
        plot_paths.append(str(path))

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
            label="layerwise two-mode HOST",
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
            / "layerwise_two_mode_error.png"
        )
        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(fig)
        plot_paths.append(str(path))

    except Exception as exc:
        plot_paths.append(
            "Plotting skipped: {}".format(exc)
        )

    print()
    print("=" * 124)
    print("HOST LAYER-AWARE ANALYTIC SECOND SHEAR MODE")
    print("=" * 124)
    print(
        "Residual SVD first-mode fraction : {:.8%}".format(
            float(
                residual["first_mode_explained_fraction"]
            )
        )
    )
    print()
    print(
        "{:>8s} {:>8s} {:>8s} {:>14s} {:>14s}".format(
            "coredeg",
            "facedeg",
            "npar",
            "shape RMSE",
            "correlation",
        )
    )

    for row in sorted(
        candidate_rows,
        key=lambda item: (
            int(item["parameter_count"]),
            float(item["relative_rmse"]),
        ),
    ):
        print(
            "{:8d} {:8d} {:8d} {:13.5%} {:14.9f}".format(
                int(row["core_degree"]),
                int(row["face_degree"]),
                int(row["parameter_count"]),
                float(row["relative_rmse"]),
                float(row["weighted_correlation"]),
            )
        )

    print()
    print(
        "Selected core/face degree       : {} / {}".format(
            selected["core_degree"],
            selected["face_degree"],
        )
    )
    print(
        "Selected shape RMSE             : {:.6%}".format(
            float(selected["relative_rmse"])
        )
    )
    print("Selected coefficients:")
    for name, value in selected[
        "coefficients_maxabs_normalized"
    ].items():
        print(
            "  {:>4s} = {:+.12e}".format(
                name,
                float(value),
            )
        )

    interface = mode.interface_diagnostics()

    print()
    print(
        "df2 core/face interface         : {:+.6e} / {:+.6e}".format(
            interface["df_core_side"],
            interface["df_face_side"],
        )
    )
    print(
        "df2 core/face ratio             : {:.6f}".format(
            interface["df_core_over_face"]
        )
    )
    print(
        "Gface/Gcore                     : {:.6f}".format(
            G_face / G_core
        )
    )
    print(
        "Local beta-block condensed alpha: {:.9f}".format(
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

    for row in result_rows:
        print(
            "{:6d} {:8.3f} {:13.8f} {:13.8f} {:13.8f} {:10.3%} {:10.3%}".format(
                int(row["n_shear"]),
                float(row["a_over_h"]),
                float(row["HOST_one_mode_abs_w_mid_mm"]),
                float(row["HOST_layerwise_two_mode_abs_w_mid_mm"]),
                float(row["FE_abs_w_mid_mm"]),
                float(row["one_mode_error_relative"]),
                float(row["two_mode_error_relative"]),
            )
        )

    print()
    print(
        "RMS error 1mode -> 2mode        : {:.4f}% -> {:.4f}%".format(
            100.0 * summary["RMS_one_mode_relative_error"],
            100.0 * summary["RMS_two_mode_relative_error"],
        )
    )
    print(
        "Max error 1mode -> 2mode        : {:.4f}% -> {:.4f}%".format(
            100.0 * summary["max_one_mode_relative_error"],
            100.0 * summary["max_two_mode_relative_error"],
        )
    )
    print()
    print("Wrote:")
    print(" ", json_path)
    print(" ", csv_path)
    print(" ", results_csv_path)
    for path in plot_paths:
        print(" ", path)
    print("=" * 124)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
