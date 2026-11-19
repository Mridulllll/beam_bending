#!/usr/bin/env python3
"""
Finite-width 2-D HOST plate solution for the canonical four-point-bending case.

This verification runner deliberately consumes the output of

    verification/beam_reduction/audit_plate_to_beam_reduction.py

because that audit already reconstructs and records the exact directional
five-field 12x12 plate section stiffness used in the current theory branch.

The audit matrix K12_plate contains integration across the physical benchmark
width because it was constructed for direct beam/plate-section comparison.
For the genuine 2-D plate problem it is divided by the recorded width before
the planform Ritz integration.

The loading/support idealization matches the theory-compatible Abaqus
"section_distributing" reference:

* two transverse load resultants, each P/2, distributed uniformly across width;
* supports constrain width-average w0 at the two support sections;
* outer x and y edges are free natural boundaries;
* the same minimal width-average in-plane stabilization is used only to remove
  rigid-body modes.

The solver uses tensor-product orthonormal Legendre Ritz bases and performs
an explicit basis-order convergence study.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from host.ritz_plate import (
    solve_directional_five_field_ritz_4pb,
)


DEFAULT_ORDERS = (
    (16, 8),
    (20, 10),
    (24, 10),
    (28, 10),
    (30, 10),
    (32, 10),
)


def parse_orders(
    text: str,
) -> Tuple[Tuple[int, int], ...]:
    pairs = []

    for token in str(text).split(","):
        token = token.strip().lower()

        if not token:
            continue

        parts = token.split("x")

        if len(parts) != 2:
            raise ValueError(
                "Invalid order token {!r}; expected e.g. 20x10.".format(
                    token
                )
            )

        pairs.append(
            (
                int(parts[0]),
                int(parts[1]),
            )
        )

    if not pairs:
        raise ValueError(
            "At least one Ritz order pair is required."
        )

    return tuple(pairs)


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
        "--output-dir",
        type=Path,
        default=Path(
            "verification/plate_4pb/"
            "gyroid_rd12_initial"
        ),
    )

    parser.add_argument(
        "--orders",
        default=",".join(
            "{}x{}".format(nx, ny)
            for nx, ny in DEFAULT_ORDERS
        ),
        help=(
            "Comma-separated Ritz degree pairs, "
            "e.g. 12x6,16x8,20x10."
        ),
    )

    parser.add_argument(
        "--profile-points",
        type=int,
        default=61,
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def relative_error(
    value: float,
    reference: float,
) -> float:
    return (
        abs(
            float(value)
            - float(reference)
        )
        / max(
            abs(float(reference)),
            1.0e-30,
        )
    )


def main() -> int:
    args = parse_args()

    audit_path = args.audit.resolve()

    if not audit_path.is_file():
        raise FileNotFoundError(
            "Reduction-audit JSON not found: {}".format(
                audit_path
            )
        )

    audit = load_json(
        audit_path
    )

    geometry = audit[
        "geometry"
    ]

    length = float(
        geometry[
            "length_mm"
        ]
    )
    width = float(
        geometry[
            "width_mm"
        ]
    )
    support_span = float(
        geometry[
            "support_span_mm"
        ]
    )
    load_span = float(
        geometry[
            "load_span_mm"
        ]
    )
    total_load = float(
        geometry[
            "total_load_N"
        ]
    )

    support_left = 0.5 * (
        length - support_span
    )
    support_right = 0.5 * (
        length + support_span
    )

    load_left = 0.5 * (
        length - load_span
    )
    load_right = 0.5 * (
        length + load_span
    )

    K12_width_integrated = np.asarray(
        audit[
            "K12_plate"
        ],
        dtype=float,
    )

    if K12_width_integrated.shape != (12, 12):
        raise ValueError(
            "audit K12_plate must have shape (12,12)."
        )

    # The reduction audit multiplied the directional plate constitutive
    # stiffness by the benchmark width to obtain beam-section units.
    # Undo exactly that factor before the genuine 2-D midsurface integration.
    K12_per_unit_width = (
        K12_width_integrated
        / width
    )

    comparisons = audit.get(
        "comparisons",
        {},
    )

    beam_abs_w = float(
        comparisons.get(
            "expected_current_HOST_abs_w_mm",
            float("nan"),
        )
    )

    fe_abs_w = float(
        comparisons.get(
            "clean_3D_FE_abs_w_mm",
            float("nan"),
        )
    )

    orders = parse_orders(
        args.orders
    )

    rows = []
    solutions = []

    print()
    print("=" * 78)
    print("HOST FINITE-WIDTH 2-D FOUR-POINT-BENDING RITZ SOLUTION")
    print("=" * 78)
    print("Audit       :", audit_path)
    print(
        "Geometry    : L={:.6g} mm, b={:.6g} mm".format(
            length,
            width,
        )
    )
    print(
        "Supports    : x={:.6g}, {:.6g} mm".format(
            support_left,
            support_right,
        )
    )
    print(
        "Loads       : x={:.6g}, {:.6g} mm, P={:.6g} N".format(
            load_left,
            load_right,
            total_load,
        )
    )
    print(
        "HOST beam   : |w_mid|={:.12g} mm".format(
            beam_abs_w
        )
    )
    print(
        "3-D FE      : |w_mid|={:.12g} mm".format(
            fe_abs_w
        )
    )
    print()
    print(
        "{:>8s} {:>8s} {:>15s} {:>15s} {:>15s} {:>13s} {:>13s}".format(
            "nx",
            "ny",
            "w_avg [mm]",
            "w_ctr [mm]",
            "w_edge [mm]",
            "vs beam",
            "vs FE",
        )
    )

    for nx_degree, ny_degree in orders:
        solution = (
            solve_directional_five_field_ritz_4pb(
                K12_per_unit_width,
                length=length,
                width=width,
                support_left=support_left,
                support_right=support_right,
                load_left=load_left,
                load_right=load_right,
                total_load=total_load,
                nx_degree=nx_degree,
                ny_degree=ny_degree,
            )
        )

        solutions.append(
            solution
        )

        x_mid = 0.5 * length

        w_average = solution.width_average(
            "w0",
            x_mid,
        )

        w_center = solution.field(
            "w0",
            x_mid,
            0.0,
        )

        w_edge_positive = solution.field(
            "w0",
            x_mid,
            +0.5 * width,
        )

        w_edge_negative = solution.field(
            "w0",
            x_mid,
            -0.5 * width,
        )

        beam_relative = relative_error(
            abs(w_average),
            beam_abs_w,
        )

        fe_relative = relative_error(
            abs(w_average),
            fe_abs_w,
        )

        row = {
            "nx_degree": int(nx_degree),
            "ny_degree": int(ny_degree),
            "dof_count": int(
                5
                * (nx_degree + 1)
                * (ny_degree + 1)
            ),
            "midspan_width_average_w_mm": float(
                w_average
            ),
            "midspan_center_w_mm": float(
                w_center
            ),
            "midspan_edge_positive_w_mm": float(
                w_edge_positive
            ),
            "midspan_edge_negative_w_mm": float(
                w_edge_negative
            ),
            "midspan_edge_minus_center_mm": float(
                0.5
                * (
                    w_edge_positive
                    + w_edge_negative
                )
                - w_center
            ),
            "relative_to_beam": float(
                beam_relative
            ),
            "relative_to_FE": float(
                fe_relative
            ),
            "support_reaction_left_N": float(
                solution.support_reaction_left
            ),
            "support_reaction_right_N": float(
                solution.support_reaction_right
            ),
            "reaction_sum_N": float(
                solution.support_reaction_left
                + solution.support_reaction_right
            ),
            "strain_energy_Nmm": float(
                solution.strain_energy
            ),
            "linear_ramp_external_work_Nmm": float(
                solution.linear_ramp_external_work
            ),
            "energy_relative_difference": float(
                abs(
                    solution.strain_energy
                    - solution.linear_ramp_external_work
                )
                / max(
                    abs(
                        solution.strain_energy
                    ),
                    1.0e-30,
                )
            ),
            "projected_residual_relative": float(
                solution.projected_residual_relative
            ),
            "constraint_residual_max": float(
                solution.constraint_residual_max
            ),
            "scaled_reduced_condition_number": float(
                solution.scaled_reduced_condition_number
            ),
        }

        rows.append(
            row
        )

        print(
            "{:8d} {:8d} {:15.9e} {:15.9e} {:15.9e} {:12.5%} {:12.5%}".format(
                nx_degree,
                ny_degree,
                w_average,
                w_center,
                0.5
                * (
                    w_edge_positive
                    + w_edge_negative
                ),
                beam_relative,
                fe_relative,
            )
        )

    if len(rows) >= 3:
        last_three = np.asarray(
            [
                abs(
                    row[
                        "midspan_width_average_w_mm"
                    ]
                )
                for row in rows[-3:]
            ],
            dtype=float,
        )

        convergence_spread = float(
            (
                np.max(last_three)
                - np.min(last_three)
            )
            / np.mean(last_three)
        )
    else:
        convergence_spread = float(
            "nan"
        )

    final_row = rows[-1]
    final_solution = solutions[-1]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    convergence_csv = (
        output_dir
        / "ritz_convergence.csv"
    )

    with convergence_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    profile_csv = (
        output_dir
        / "midspan_width_profile.csv"
    )

    y_values = np.linspace(
        -0.5 * width,
        +0.5 * width,
        int(args.profile_points),
    )

    with profile_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "y_mm",
                "w0_mm",
                "u0_mm",
                "v0_mm",
                "beta_x",
                "beta_y",
            ),
        )
        writer.writeheader()

        for y in y_values:
            writer.writerow(
                {
                    "y_mm": float(y),
                    "w0_mm": final_solution.field(
                        "w0",
                        0.5 * length,
                        float(y),
                    ),
                    "u0_mm": final_solution.field(
                        "u0",
                        0.5 * length,
                        float(y),
                    ),
                    "v0_mm": final_solution.field(
                        "v0",
                        0.5 * length,
                        float(y),
                    ),
                    "beta_x": final_solution.field(
                        "beta_x",
                        0.5 * length,
                        float(y),
                    ),
                    "beta_y": final_solution.field(
                        "beta_y",
                        0.5 * length,
                        float(y),
                    ),
                }
            )

    final_abs_w = abs(
        float(
            final_row[
                "midspan_width_average_w_mm"
            ]
        )
    )

    report = {
        "schema": "host_2d_4pb_ritz_v1",
        "source_audit": str(
            audit_path
        ),
        "geometry": {
            "length_mm": length,
            "width_mm": width,
            "support_left_mm": support_left,
            "support_right_mm": support_right,
            "load_left_mm": load_left,
            "load_right_mm": load_right,
            "total_load_N": total_load,
        },
        "fixture": {
            "support_type": (
                "width-average generalized section translation"
            ),
            "load_type": (
                "uniform transverse line resultant across width"
            ),
            "outer_edges": (
                "free natural Ritz boundaries"
            ),
            "in_plane_stabilization": (
                "width-average pin/guide matching full fxsec gauge"
            ),
        },
        "constitutive": {
            "directional_size": 12,
            "source_K12_was_width_integrated": True,
            "division_by_width_mm": width,
        },
        "reference": {
            "HOST_beam_abs_midspan_w_mm": beam_abs_w,
            "clean_3D_FE_abs_midspan_w_mm": fe_abs_w,
        },
        "convergence": {
            "rows": rows,
            "last_three_relative_spread": convergence_spread,
        },
        "final": {
            **final_row,
            "abs_midspan_width_average_w_mm": final_abs_w,
            "plate_minus_beam_relative_signed": (
                (
                    final_abs_w
                    - beam_abs_w
                )
                / beam_abs_w
            ),
            "plate_minus_FE_relative_signed": (
                (
                    final_abs_w
                    - fe_abs_w
                )
                / fe_abs_w
            ),
        },
    }

    report_json = (
        output_dir
        / "host_2d_4pb_ritz.json"
    )

    with report_json.open(
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
    print("-" * 78)
    print(
        "Final width-average |w_mid| : {:.12g} mm".format(
            final_abs_w
        )
    )
    print(
        "Plate vs HOST beam          : {:+.6%}".format(
            report[
                "final"
            ][
                "plate_minus_beam_relative_signed"
            ]
        )
    )
    print(
        "Plate vs clean 3-D FE       : {:+.6%}".format(
            report[
                "final"
            ][
                "plate_minus_FE_relative_signed"
            ]
        )
    )
    print(
        "Last-three Ritz spread      : {:.6%}".format(
            convergence_spread
        )
    )
    print(
        "Support reactions           : {:.9f}, {:.9f} N".format(
            final_solution.support_reaction_left,
            final_solution.support_reaction_right,
        )
    )
    print(
        "Internal/external energy err: {:.3e}".format(
            final_row[
                "energy_relative_difference"
            ]
        )
    )
    print(
        "Projected residual          : {:.3e}".format(
            final_solution.projected_residual_relative
        )
    )
    print(
        "Constraint residual         : {:.3e}".format(
            final_solution.constraint_residual_max
        )
    )
    print("-" * 78)
    print("Wrote:")
    print(" ", report_json)
    print(" ", convergence_csv)
    print(" ", profile_csv)
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
