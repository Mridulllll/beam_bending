#!/usr/bin/env python3
"""
Audit whether the remaining HOST-vs-3D-FE 4PB discrepancy is primarily a
transverse-shear compliance mismatch.

Inputs
------
1. verification/beam_reduction/plate_to_beam_audit.json
   Provides the verified HOST beam K3 matrix.

2. verification/slenderness/results/gyroid_rd12_initial/slenderness_sweep.json
   Provides the six geometries and FE midspan deflections.

Method
------
For the current symmetric benchmark the condensed beam stiffness has the
generalized order

    [kappa, chi, beta_x]

and the beta_x row/column is essentially uncoupled from [kappa, chi].

The audit therefore introduces ONE diagnostic scalar alpha_G multiplying only

    K3[2,2] -> alpha_G * K3[2,2].

alpha_G < 1 means the HOST transverse-shear sector is softened.

This is deliberately a diagnostic, not a proposed final theory correction.
If one scalar shear-softening factor collapses the entire deep-to-slender FE
trend, the next theory work should focus on the transverse-shear warping /
effective shear compliance rather than on adding epsilon_zz.

The script also fits the raw FE-HOST deflection gap to

    delta_FE - delta_HOST = c0 + c1 * a,

because a near-linear dependence on shear-span length is characteristic of a
distributed shear-compliance difference plus an approximately fixed local
boundary-layer contribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize_scalar

from host.beam_point_loads import (
    solve_simply_supported_point_loads,
    symmetric_four_point_bending_loads,
    symmetric_support_positions,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_compliance"
        ),
    )

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=1.2,
    )

    return parser.parse_args()


def solve_midspan(
    K3: np.ndarray,
    row: dict,
) -> float:
    length = float(row["length_mm"])
    support_span = float(row["support_span_mm"])
    load_span = float(row["load_span_mm"])
    total_load = 1000.0

    xA, xB = symmetric_support_positions(
        specimen_length=length,
        support_span=support_span,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=total_load,
        specimen_length=length,
        load_span=load_span,
    )

    solution = solve_simply_supported_point_loads(
        np.asarray(K3, dtype=float),
        length=length,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    return abs(
        float(
            solution.state(
                0.5 * length
            )[0]
        )
    )


def finite_rows(report: dict) -> List[dict]:
    rows = []

    for row in report.get("rows", []):
        fe = row.get("FE_3D_abs_w_mid_mm")

        if fe in ("", None):
            continue

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No FE-populated slenderness rows were found."
        )

    rows.sort(
        key=lambda item: float(item["a_over_h"])
    )

    return rows


def scaled_K3(
    K3: np.ndarray,
    alpha: float,
) -> np.ndarray:
    result = np.asarray(
        K3,
        dtype=float,
    ).copy()

    result[2, 2] *= float(alpha)

    return result


def objective(
    alpha: float,
    K3: np.ndarray,
    rows: List[dict],
) -> float:
    K = scaled_K3(
        K3,
        alpha,
    )

    residuals = []

    for row in rows:
        prediction = solve_midspan(
            K,
            row,
        )
        reference = float(
            row["FE_3D_abs_w_mid_mm"]
        )

        residuals.append(
            (prediction - reference)
            / reference
        )

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    return float(
        np.mean(
            residuals ** 2
        )
    )


def linear_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    coefficients = np.polyfit(
        x,
        y,
        1,
    )

    slope = float(
        coefficients[0]
    )
    intercept = float(
        coefficients[1]
    )

    predicted = (
        slope * x
        + intercept
    )

    ss_res = float(
        np.sum(
            (y - predicted) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    if ss_tot > 0.0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")

    return {
        "intercept_mm": intercept,
        "slope_mm_per_mm": slope,
        "r_squared": float(r2),
        "predicted_mm": predicted.tolist(),
    }


def best_additive_offset_mm(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Least-squares constant offset c0 in

        reference ~= prediction + c0.

    This is used only as a diagnostic for an approximately fixed local
    support/load boundary-layer compliance.
    """
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)

    return float(
        np.mean(
            reference - prediction
        )
    )


def objective_with_offset(
    alpha: float,
    K3: np.ndarray,
    rows: List[dict],
) -> float:
    """
    Absolute-deflection least-squares objective with the best constant offset.

    For each trial alpha_G:
        FE ~= HOST(alpha_G) + c_local

    where c_local is obtained analytically as the mean residual in mm.
    """
    K = scaled_K3(
        K3,
        alpha,
    )

    prediction = np.asarray(
        [
            solve_midspan(
                K,
                row,
            )
            for row in rows
        ],
        dtype=float,
    )

    reference = np.asarray(
        [
            float(
                row[
                    "FE_3D_abs_w_mid_mm"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    c_local = best_additive_offset_mm(
        prediction,
        reference,
    )

    residual = (
        prediction
        + c_local
        - reference
    )

    return float(
        np.mean(
            residual ** 2
        )
    )


def solve_casewise_alpha(
    K3: np.ndarray,
    row: dict,
    *,
    alpha_min: float,
    alpha_max: float,
    additive_offset_mm: float = 0.0,
) -> dict:
    """
    Infer the diagnostic alpha_G that matches one FE deflection.

    No claim is made that this is a material parameter.  Constancy of the
    inferred alpha over slenderness would support a true shear-stiffness
    mismatch; strong variation would indicate additional mechanisms.
    """
    reference = float(
        row[
            "FE_3D_abs_w_mid_mm"
        ]
    )

    def case_objective(alpha: float) -> float:
        prediction = solve_midspan(
            scaled_K3(
                K3,
                alpha,
            ),
            row,
        )

        residual = (
            prediction
            + float(additive_offset_mm)
            - reference
        )

        return residual * residual

    result = minimize_scalar(
        case_objective,
        bounds=(
            float(alpha_min),
            float(alpha_max),
        ),
        method="bounded",
        options={
            "xatol": 1.0e-13,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Casewise alpha fit failed for N_SHEAR={}: {}".format(
                row.get(
                    "n_shear"
                ),
                result.message,
            )
        )

    alpha = float(
        result.x
    )

    prediction = solve_midspan(
        scaled_K3(
            K3,
            alpha,
        ),
        row,
    )

    return {
        "alpha_G": alpha,
        "softening_percent": 100.0 * (1.0 - alpha),
        "prediction_plus_offset_mm": (
            prediction
            + float(additive_offset_mm)
        ),
        "residual_mm": (
            prediction
            + float(additive_offset_mm)
            - reference
        ),
    }


def maybe_plot(
    rows: List[dict],
    output_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []

    a_over_h = np.asarray(
        [float(row["a_over_h"]) for row in rows],
        dtype=float,
    )

    baseline = np.asarray(
        [float(row["HOST_baseline_abs_w_mm"]) for row in rows],
        dtype=float,
    )

    fitted = np.asarray(
        [float(row["HOST_shear_fitted_abs_w_mm"]) for row in rows],
        dtype=float,
    )

    fe = np.asarray(
        [float(row["FE_abs_w_mm"]) for row in rows],
        dtype=float,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        a_over_h,
        fe,
        marker="o",
        label="3D FE",
    )
    ax.plot(
        a_over_h,
        baseline,
        marker="s",
        label="HOST baseline",
    )
    ax.plot(
        a_over_h,
        fitted,
        marker="^",
        label="HOST fitted shear",
    )

    ax.set_xlabel("a/h")
    ax.set_ylabel("|w_mid| [mm]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    path = (
        output_dir
        / "shear_fitted_deflection.png"
    )
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)
    paths.append(str(path))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    baseline_error = 100.0 * (
        baseline - fe
    ) / fe

    fitted_error = 100.0 * (
        fitted - fe
    ) / fe

    ax.plot(
        a_over_h,
        baseline_error,
        marker="o",
        label="baseline HOST",
    )
    ax.plot(
        a_over_h,
        fitted_error,
        marker="s",
        label="shear-fitted HOST",
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
        / "shear_fitted_error.png"
    )
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(fig)
    paths.append(str(path))

    return paths


def main() -> int:
    args = parse_args()

    audit_path = args.audit.resolve()
    slenderness_path = (
        args.slenderness.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    if not audit_path.is_file():
        raise FileNotFoundError(
            audit_path
        )

    if not slenderness_path.is_file():
        raise FileNotFoundError(
            slenderness_path
        )

    audit = load_json(
        audit_path
    )

    slenderness = load_json(
        slenderness_path
    )

    K3 = np.asarray(
        audit[
            "models"
        ][
            "beam_free_stress"
        ][
            "K3_N_zero"
        ],
        dtype=float,
    )

    if K3.shape != (3, 3):
        raise ValueError(
            "Expected beam K3 shape (3,3)."
        )

    coupling_ratio = float(
        np.linalg.norm(
            K3[2, :2]
        )
        / max(
            abs(float(K3[2, 2])),
            1.0e-30,
        )
    )

    rows_source = finite_rows(
        slenderness
    )

    baseline_values = np.asarray(
        [
            solve_midspan(
                K3,
                row,
            )
            for row in rows_source
        ],
        dtype=float,
    )

    fe_values = np.asarray(
        [
            float(
                row[
                    "FE_3D_abs_w_mid_mm"
                ]
            )
            for row in rows_source
        ],
        dtype=float,
    )

    source_host_values = np.asarray(
        [
            float(
                row[
                    "HOST_1D_abs_w_mid_mm"
                ]
            )
            for row in rows_source
        ],
        dtype=float,
    )

    reconstruction_relative = float(
        np.max(
            np.abs(
                baseline_values
                - source_host_values
            )
            / np.maximum(
                np.abs(
                    source_host_values
                ),
                1.0e-30,
            )
        )
    )

    optimizer = minimize_scalar(
        objective,
        bounds=(
            float(args.alpha_min),
            float(args.alpha_max),
        ),
        args=(
            K3,
            rows_source,
        ),
        method="bounded",
        options={
            "xatol": 1.0e-12,
        },
    )

    if not optimizer.success:
        raise RuntimeError(
            "Shear-factor fit failed: {}".format(
                optimizer.message
            )
        )

    alpha = float(
        optimizer.x
    )

    K3_fitted = scaled_K3(
        K3,
        alpha,
    )

    fitted_values = np.asarray(
        [
            solve_midspan(
                K3_fitted,
                row,
            )
            for row in rows_source
        ],
        dtype=float,
    )

    # Second diagnostic: allow a constant additive compliance contribution in
    # addition to the distributed shear-stiffness change.  The nearly linear
    # raw FE-HOST gap versus shear-span length motivates this decomposition.
    optimizer_offset = minimize_scalar(
        objective_with_offset,
        bounds=(
            float(args.alpha_min),
            float(args.alpha_max),
        ),
        args=(
            K3,
            rows_source,
        ),
        method="bounded",
        options={
            "xatol": 1.0e-13,
        },
    )

    if not optimizer_offset.success:
        raise RuntimeError(
            "Shear+offset fit failed: {}".format(
                optimizer_offset.message
            )
        )

    alpha_offset = float(
        optimizer_offset.x
    )

    K3_offset = scaled_K3(
        K3,
        alpha_offset,
    )

    offset_predictions_raw = np.asarray(
        [
            solve_midspan(
                K3_offset,
                row,
            )
            for row in rows_source
        ],
        dtype=float,
    )

    c_local_mm = best_additive_offset_mm(
        offset_predictions_raw,
        fe_values,
    )

    offset_predictions = (
        offset_predictions_raw
        + c_local_mm
    )

    shear_spans = np.asarray(
        [
            float(
                row[
                    "shear_span_mm"
                ]
            )
            for row in rows_source
        ],
        dtype=float,
    )

    raw_gap = (
        fe_values
        - baseline_values
    )

    fitted_gap = (
        fe_values
        - fitted_values
    )

    offset_gap = (
        fe_values
        - offset_predictions
    )

    casewise_alpha_raw = [
        solve_casewise_alpha(
            K3,
            row,
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
            additive_offset_mm=0.0,
        )
        for row in rows_source
    ]

    casewise_alpha_offset = [
        solve_casewise_alpha(
            K3,
            row,
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
            additive_offset_mm=c_local_mm,
        )
        for row in rows_source
    ]

    raw_linear = linear_fit(
        shear_spans,
        raw_gap,
    )

    fitted_linear = linear_fit(
        shear_spans,
        fitted_gap,
    )

    output_rows = []

    for i, source in enumerate(
        rows_source
    ):
        baseline_error = (
            baseline_values[i]
            - fe_values[i]
        ) / fe_values[i]

        fitted_error = (
            fitted_values[i]
            - fe_values[i]
        ) / fe_values[i]

        output_rows.append(
            {
                "n_shear": int(
                    source[
                        "n_shear"
                    ]
                ),
                "a_over_h": float(
                    source[
                        "a_over_h"
                    ]
                ),
                "shear_span_mm": float(
                    source[
                        "shear_span_mm"
                    ]
                ),
                "HOST_baseline_abs_w_mm": float(
                    baseline_values[i]
                ),
                "HOST_shear_fitted_abs_w_mm": float(
                    fitted_values[i]
                ),
                "HOST_shear_plus_local_abs_w_mm": float(
                    offset_predictions[i]
                ),
                "FE_abs_w_mm": float(
                    fe_values[i]
                ),
                "baseline_error_relative": float(
                    baseline_error
                ),
                "fitted_error_relative": float(
                    fitted_error
                ),
                "shear_plus_local_error_relative": float(
                    (
                        offset_predictions[i]
                        - fe_values[i]
                    )
                    / fe_values[i]
                ),
                "casewise_alpha_G_no_offset": float(
                    casewise_alpha_raw[i]["alpha_G"]
                ),
                "casewise_alpha_G_with_global_local_offset": float(
                    casewise_alpha_offset[i]["alpha_G"]
                ),
                "raw_FE_minus_HOST_mm": float(
                    raw_gap[i]
                ),
                "fitted_FE_minus_HOST_mm": float(
                    fitted_gap[i]
                ),
                "shear_plus_local_FE_minus_HOST_mm": float(
                    offset_gap[i]
                ),
            }
        )

    baseline_rms_error = float(
        np.sqrt(
            np.mean(
                (
                    (
                        baseline_values
                        - fe_values
                    )
                    / fe_values
                ) ** 2
            )
        )
    )

    fitted_rms_error = float(
        np.sqrt(
            np.mean(
                (
                    (
                        fitted_values
                        - fe_values
                    )
                    / fe_values
                ) ** 2
            )
        )
    )

    baseline_max_error = float(
        np.max(
            np.abs(
                (
                    baseline_values
                    - fe_values
                )
                / fe_values
            )
        )
    )

    fitted_max_error = float(
        np.max(
            np.abs(
                (
                    fitted_values
                    - fe_values
                )
                / fe_values
            )
        )
    )

    offset_rms_error = float(
        np.sqrt(
            np.mean(
                (
                    (
                        offset_predictions
                        - fe_values
                    )
                    / fe_values
                ) ** 2
            )
        )
    )

    offset_max_error = float(
        np.max(
            np.abs(
                (
                    offset_predictions
                    - fe_values
                )
                / fe_values
            )
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "shear_compliance_audit.csv"
    )

    json_path = (
        output_dir
        / "shear_compliance_audit.json"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            output_rows
        )

    plots = maybe_plot(
        output_rows,
        output_dir,
    )

    report = {
        "schema": "host_4pb_shear_compliance_audit_v1",
        "warning": (
            "alpha_G is a diagnostic scalar applied to K3[2,2], "
            "not a calibrated final theory parameter."
        ),
        "K3": K3.tolist(),
        "beta_coupling_ratio": coupling_ratio,
        "beam_reconstruction_max_relative_error": (
            reconstruction_relative
        ),
        "fit_relative_error_only": {
            "alpha_G": alpha,
            "effective_shear_softening_percent": (
                100.0
                * (1.0 - alpha)
            ),
            "baseline_rms_relative_error": (
                baseline_rms_error
            ),
            "fitted_rms_relative_error": (
                fitted_rms_error
            ),
            "baseline_max_relative_error": (
                baseline_max_error
            ),
            "fitted_max_relative_error": (
                fitted_max_error
            ),
        },
        "fit_shear_plus_constant_local": {
            "alpha_G": alpha_offset,
            "effective_shear_softening_percent": (
                100.0
                * (1.0 - alpha_offset)
            ),
            "constant_local_compliance_mm": c_local_mm,
            "rms_relative_error": offset_rms_error,
            "max_relative_error": offset_max_error,
        },
        "casewise_alpha_no_offset": casewise_alpha_raw,
        "casewise_alpha_with_global_local_offset": casewise_alpha_offset,
        "raw_gap_linear_fit": raw_linear,
        "fitted_gap_linear_fit": fitted_linear,
        "rows": output_rows,
        "plots": plots,
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
        handle.write(
            "\n"
        )

    print()
    print("=" * 108)
    print("HOST 4PB TRANSVERSE-SHEAR COMPLIANCE AUDIT")
    print("=" * 108)
    print(
        "beta_x coupling ratio          : {:.3e}".format(
            coupling_ratio
        )
    )
    print(
        "beam reconstruction max error : {:.3e}".format(
            reconstruction_relative
        )
    )
    print()
    print(
        "Best diagnostic alpha_G       : {:.9f}".format(
            alpha
        )
    )
    print(
        "Equivalent shear softening    : {:.4f} %".format(
            100.0
            * (1.0 - alpha)
        )
    )
    print(
        "RMS error baseline -> fitted  : {:.4f}% -> {:.4f}%".format(
            100.0 * baseline_rms_error,
            100.0 * fitted_rms_error,
        )
    )
    print(
        "Max error baseline -> fitted  : {:.4f}% -> {:.4f}%".format(
            100.0 * baseline_max_error,
            100.0 * fitted_max_error,
        )
    )
    print()
    print(
        "Best alpha_G with local offset: {:.9f}".format(
            alpha_offset
        )
    )
    print(
        "Equivalent shear softening    : {:.4f} %".format(
            100.0
            * (1.0 - alpha_offset)
        )
    )
    print(
        "Constant local compliance     : {:.9e} mm".format(
            c_local_mm
        )
    )
    print(
        "RMS error shear+local         : {:.4f} %".format(
            100.0 * offset_rms_error
        )
    )
    print(
        "Max error shear+local         : {:.4f} %".format(
            100.0 * offset_max_error
        )
    )
    print()
    print(
        "Raw FE-HOST gap ~= c0 + c1*a:"
    )
    print(
        "  c0 = {:.9e} mm".format(
            raw_linear[
                "intercept_mm"
            ]
        )
    )
    print(
        "  c1 = {:.9e} mm/mm".format(
            raw_linear[
                "slope_mm_per_mm"
            ]
        )
    )
    print(
        "  R2 = {:.9f}".format(
            raw_linear[
                "r_squared"
            ]
        )
    )
    print()
    print(
        "{:>6s} {:>8s} {:>13s} {:>13s} {:>13s} {:>13s} {:>10s} {:>10s} {:>10s}".format(
            "N_A",
            "a/h",
            "HOST",
            "shear fit",
            "shear+loc",
            "FE",
            "err base",
            "err shear",
            "err +loc",
        )
    )

    for row in output_rows:
        print(
            "{:6d} {:8.3f} {:13.8f} {:13.8f} {:13.8f} {:13.8f} {:9.3%} {:9.3%} {:9.3%}".format(
                int(
                    row[
                        "n_shear"
                    ]
                ),
                float(
                    row[
                        "a_over_h"
                    ]
                ),
                float(
                    row[
                        "HOST_baseline_abs_w_mm"
                    ]
                ),
                float(
                    row[
                        "HOST_shear_fitted_abs_w_mm"
                    ]
                ),
                float(
                    row[
                        "HOST_shear_plus_local_abs_w_mm"
                    ]
                ),
                float(
                    row[
                        "FE_abs_w_mm"
                    ]
                ),
                float(
                    row[
                        "baseline_error_relative"
                    ]
                ),
                float(
                    row[
                        "fitted_error_relative"
                    ]
                ),
                float(
                    row[
                        "shear_plus_local_error_relative"
                    ]
                ),
            )
        )

    print()
    print("Casewise diagnostic alpha_G values:")
    print(
        "{:>6s} {:>8s} {:>16s} {:>18s}".format(
            "N_A",
            "a/h",
            "alpha no offset",
            "alpha with offset",
        )
    )

    for i, row in enumerate(output_rows):
        print(
            "{:6d} {:8.3f} {:16.9f} {:18.9f}".format(
                int(row["n_shear"]),
                float(row["a_over_h"]),
                float(
                    casewise_alpha_raw[i]["alpha_G"]
                ),
                float(
                    casewise_alpha_offset[i]["alpha_G"]
                ),
            )
        )

    print()
    print("Wrote:")
    print(" ", csv_path)
    print(" ", json_path)

    for path in plots:
        print(" ", path)

    print("=" * 108)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
