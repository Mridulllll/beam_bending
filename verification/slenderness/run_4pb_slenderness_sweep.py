#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOST 4PB deep-to-slender theory/FE comparison sweep.

The sweep varies only N_SHEAR, i.e. the support-to-nearest-load distance

    a = N_SHEAR * l_c,

while keeping fixed

    l_c, core height, width, face thickness,
    loading-line separation, overhang,
    material, load magnitude.

For each requested N_SHEAR the script evaluates:

1. the current 1-D HOST beam using the verified K3_N_zero matrix from
   verification/beam_reduction/plate_to_beam_audit.json;

2. the finite-width 2-D directional HOST plate using the Ritz solver;

3. the postprocessed quarter-symmetry Abaqus FE campaign result, when found.

The intended FE campaign definition is

    C3D20R
    HOST_4PB_NE_CELL = 2
    HOST_4PB_FACE_NE_Z = 2
    HOST_4PB_FIXTURE_MODE = section_distributing
    HOST_4PB_DOMAIN_MODE = quarter_xy

Missing FE cases are permitted: theory rows are still written and the FE
columns remain blank.  Re-run the script after the Abaqus cases have been
solved/postprocessed.

Example
-------
python verification/slenderness/run_4pb_slenderness_sweep.py ^
    --run-root "C:/AbaqusScratch/HOST/4pb_homogenized"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from host.beam_point_loads import (
    solve_simply_supported_point_loads,
    symmetric_four_point_bending_loads,
    symmetric_support_positions,
)
from host.ritz_plate import (
    solve_directional_five_field_ritz_4pb,
)


DEFAULT_N_SHEAR = (6, 9, 12, 18, 24, 30)
DEFAULT_RITZ_ORDERS = ((28, 10), (32, 10))


def parse_int_list(text: str) -> Tuple[int, ...]:
    values = []

    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("N_SHEAR values must be positive.")
        values.append(value)

    if not values:
        raise ValueError("At least one N_SHEAR value is required.")

    return tuple(values)


def parse_order_list(text: str) -> Tuple[Tuple[int, int], ...]:
    values = []

    for token in str(text).split(","):
        token = token.strip().lower()
        if not token:
            continue

        parts = token.split("x")
        if len(parts) != 2:
            raise ValueError(
                "Invalid Ritz order {!r}; use e.g. 28x10.".format(token)
            )

        nx = int(parts[0])
        ny = int(parts[1])

        if nx < 2 or ny < 2:
            raise ValueError("Ritz degrees must both be >= 2.")

        values.append((nx, ny))

    if not values:
        raise ValueError("At least one Ritz order is required.")

    return tuple(values)


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
        "--run-root",
        type=Path,
        default=Path(
            r"C:\AbaqusScratch\HOST\4pb_homogenized"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial"
        ),
    )

    parser.add_argument(
        "--n-shear",
        default=",".join(str(v) for v in DEFAULT_N_SHEAR),
    )

    parser.add_argument(
        "--ritz-orders",
        default=",".join(
            "{}x{}".format(nx, ny)
            for nx, ny in DEFAULT_RITZ_ORDERS
        ),
    )

    parser.add_argument("--cell-size", type=float, default=10.0)
    parser.add_argument("--n-height", type=int, default=6)
    parser.add_argument("--n-width", type=int, default=6)
    parser.add_argument("--n-moment", type=int, default=8)
    parser.add_argument("--n-overhang", type=int, default=2)
    parser.add_argument("--face-bottom", type=float, default=1.0)
    parser.add_argument("--face-top", type=float, default=1.0)

    parser.add_argument("--total-load", type=float, default=1000.0)

    parser.add_argument("--fe-element-type", default="C3D20R")
    parser.add_argument("--fe-ne-cell", type=int, default=2)
    parser.add_argument("--fe-face-ne-z", type=int, default=2)
    parser.add_argument("--fe-fixture-mode", default="section_distributing")
    parser.add_argument("--fe-domain-mode", default="quarter_xy")

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def geometry_from_counts(
    *,
    cell_size: float,
    n_height: int,
    n_width: int,
    n_shear: int,
    n_moment: int,
    n_overhang: int,
    face_bottom: float,
    face_top: float,
) -> dict:
    lc = float(cell_size)
    nh = int(n_height)
    nb = int(n_width)
    na = int(n_shear)
    nm = int(n_moment)
    no = int(n_overhang)

    hc = nh * lc
    width = nb * lc

    shear_span = na * lc
    load_span = nm * lc
    support_span = (2 * na + nm) * lc
    overhang = no * lc
    length = (2 * no + 2 * na + nm) * lc
    total_height = float(face_bottom) + hc + float(face_top)

    return {
        "cell_size_mm": lc,
        "n_height": nh,
        "n_width": nb,
        "n_shear": na,
        "n_moment": nm,
        "n_overhang": no,
        "core_height_mm": hc,
        "width_mm": width,
        "face_bottom_mm": float(face_bottom),
        "face_top_mm": float(face_top),
        "total_height_mm": total_height,
        "shear_span_mm": shear_span,
        "load_span_mm": load_span,
        "support_span_mm": support_span,
        "overhang_mm": overhang,
        "length_mm": length,
        "support_left_mm": overhang,
        "support_right_mm": length - overhang,
        "load_left_mm": overhang + shear_span,
        "load_right_mm": overhang + shear_span + load_span,
        "midspan_mm": 0.5 * length,
        "a_over_h": shear_span / total_height,
        "S_over_h": support_span / total_height,
        "L_over_h": length / total_height,
        "b_over_h": width / total_height,
    }


def solve_beam(
    K3: np.ndarray,
    geometry: dict,
    total_load: float,
) -> float:
    length = float(geometry["length_mm"])
    support_span = float(geometry["support_span_mm"])
    load_span = float(geometry["load_span_mm"])

    xA, xB = symmetric_support_positions(
        specimen_length=length,
        support_span=support_span,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=float(total_load),
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
            solution.state(0.5 * length)[0]
        )
    )


def solve_plate_orders(
    K12_per_unit_width: np.ndarray,
    geometry: dict,
    total_load: float,
    orders: Sequence[Tuple[int, int]],
) -> dict:
    length = float(geometry["length_mm"])
    width = float(geometry["width_mm"])

    values = []

    for nx_degree, ny_degree in orders:
        solution = solve_directional_five_field_ritz_4pb(
            K12_per_unit_width,
            length=length,
            width=width,
            support_left=float(geometry["support_left_mm"]),
            support_right=float(geometry["support_right_mm"]),
            load_left=float(geometry["load_left_mm"]),
            load_right=float(geometry["load_right_mm"]),
            total_load=float(total_load),
            nx_degree=int(nx_degree),
            ny_degree=int(ny_degree),
        )

        x_mid = 0.5 * length

        values.append(
            {
                "nx_degree": int(nx_degree),
                "ny_degree": int(ny_degree),
                "abs_w_mid_width_average_mm": abs(
                    float(
                        solution.width_average(
                            "w0",
                            x_mid,
                        )
                    )
                ),
                "abs_w_mid_center_mm": abs(
                    float(
                        solution.field(
                            "w0",
                            x_mid,
                            0.0,
                        )
                    )
                ),
                "abs_w_mid_edge_mm": abs(
                    0.5
                    * (
                        float(
                            solution.field(
                                "w0",
                                x_mid,
                                +0.5 * width,
                            )
                        )
                        + float(
                            solution.field(
                                "w0",
                                x_mid,
                                -0.5 * width,
                            )
                        )
                    )
                ),
                "reaction_sum_N": float(
                    solution.support_reaction_left
                    + solution.support_reaction_right
                ),
                "energy_error": abs(
                    float(solution.strain_energy)
                    - float(solution.linear_ramp_external_work)
                )
                / max(
                    abs(float(solution.strain_energy)),
                    1.0e-30,
                ),
                "projected_residual_relative": float(
                    solution.projected_residual_relative
                ),
            }
        )

    final = values[-1]

    if len(values) >= 2:
        previous = values[-2]["abs_w_mid_width_average_mm"]
        current = final["abs_w_mid_width_average_mm"]
        order_change = abs(current - previous) / max(abs(current), 1.0e-30)
    else:
        order_change = float("nan")

    return {
        "orders": values,
        "final": final,
        "last_order_relative_change": float(order_change),
    }


def candidate_fe_cases(
    run_root: Path,
    requested_n_shear: Sequence[int],
    *,
    element_type: str,
    ne_cell: int,
    face_ne_z: int,
    fixture_mode: str,
    domain_mode: str,
) -> Dict[int, dict]:
    requested = set(int(v) for v in requested_n_shear)
    matches: Dict[int, List[dict]] = {
        value: []
        for value in requested
    }

    if not run_root.is_dir():
        return {}

    for case_dir in run_root.iterdir():
        if not case_dir.is_dir():
            continue

        manifest_path = case_dir / "manifest.json"
        summary_path = case_dir / "post" / "summary.json"

        if not manifest_path.is_file() or not summary_path.is_file():
            continue

        try:
            manifest = load_json(manifest_path)
            summary = load_json(summary_path)
        except Exception:
            continue

        try:
            n_shear = int(manifest["geometry"]["n_shear"])
            mesh = manifest["mesh"]
            fixture = manifest["fixture"]
            domain = manifest["domain"]
        except Exception:
            continue

        if n_shear not in requested:
            continue

        if str(mesh.get("element_type", "")).upper() != str(element_type).upper():
            continue

        if int(mesh.get("core_elements_per_cell_edge", -1)) != int(ne_cell):
            continue

        if int(mesh.get("face_elements_through_thickness", -1)) != int(face_ne_z):
            continue

        if str(fixture.get("mode", "")).lower() != str(fixture_mode).lower():
            continue

        if str(domain.get("mode", "")).lower() != str(domain_mode).lower():
            continue

        # The 4PB postprocessor prints a global QA status to the console, but
        # older/newer summary schemas do not necessarily expose it under
        # summary["qa"]["overall_status"].  Do not discard an otherwise valid
        # postprocessed FE case merely because that optional label is absent.
        qa_block = summary.get("qa", {})
        if not isinstance(qa_block, dict):
            qa_block = {}

        qa_status = str(
            qa_block.get("overall_status")
            or qa_block.get("status")
            or qa_block.get("overall")
            or summary.get("qa_overall")
            or summary.get("overall_status")
            or "UNKNOWN"
        )

        try:
            w_mid = abs(
                float(
                    summary["midspan"]["U3_midplane_avg_mm"]
                )
            )
        except Exception:
            continue

        item = {
            "path": str(case_dir),
            "case_hash": manifest.get("case", {}).get("case_hash"),
            "qa_status": qa_status,
            "abs_w_mid_mm": w_mid,
            "modified_time": case_dir.stat().st_mtime,
        }

        matches[n_shear].append(item)

    selected = {}

    for n_shear, items in matches.items():
        if not items:
            continue

        passing = [
            item
            for item in items
            if item["qa_status"].upper() == "PASS"
        ]

        # If the summary schema has no explicit overall QA label, keep the
        # case.  Its presence here still requires a valid manifest, post
        # summary, matching campaign definition, and a readable midspan U3.
        pool = passing if passing else items
        pool.sort(
            key=lambda item: item["modified_time"],
            reverse=True,
        )

        selected[n_shear] = pool[0]

    return selected


def maybe_float(value):
    if value is None:
        return ""
    return float(value)


def main() -> int:
    args = parse_args()

    audit_path = args.audit.resolve()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()

    if not audit_path.is_file():
        raise FileNotFoundError(
            "Reduction audit not found: {}".format(audit_path)
        )

    audit = load_json(audit_path)

    K3_beam = np.asarray(
        audit["models"]["beam_free_stress"]["K3_N_zero"],
        dtype=float,
    )

    K12_width_integrated = np.asarray(
        audit["K12_plate"],
        dtype=float,
    )

    n_shear_values = parse_int_list(args.n_shear)
    ritz_orders = parse_order_list(args.ritz_orders)

    width = float(args.n_width) * float(args.cell_size)
    K12_per_unit_width = K12_width_integrated / width

    fe_by_n_shear = candidate_fe_cases(
        run_root,
        n_shear_values,
        element_type=args.fe_element_type,
        ne_cell=args.fe_ne_cell,
        face_ne_z=args.fe_face_ne_z,
        fixture_mode=args.fe_fixture_mode,
        domain_mode=args.fe_domain_mode,
    )

    rows = []
    detailed_plate = {}

    print()
    print("=" * 92)
    print("HOST 4PB DEEP-TO-SLENDER SWEEP")
    print("=" * 92)
    print("Audit      :", audit_path)
    print("FE run root:", run_root)
    print("N_SHEAR    :", n_shear_values)
    print("Ritz orders:", ritz_orders)
    print()
    print(
        "{:>7s} {:>9s} {:>9s} {:>14s} {:>14s} {:>14s} {:>11s} {:>11s}".format(
            "N_A",
            "a/h",
            "S/h",
            "HOST-1D",
            "HOST-2D",
            "FE-3D",
            "1D err",
            "2D err",
        )
    )

    for n_shear in n_shear_values:
        geometry = geometry_from_counts(
            cell_size=args.cell_size,
            n_height=args.n_height,
            n_width=args.n_width,
            n_shear=n_shear,
            n_moment=args.n_moment,
            n_overhang=args.n_overhang,
            face_bottom=args.face_bottom,
            face_top=args.face_top,
        )

        w_beam = solve_beam(
            K3_beam,
            geometry,
            args.total_load,
        )

        plate = solve_plate_orders(
            K12_per_unit_width,
            geometry,
            args.total_load,
            ritz_orders,
        )

        detailed_plate[str(n_shear)] = plate

        w_plate = float(
            plate["final"]["abs_w_mid_width_average_mm"]
        )

        fe = fe_by_n_shear.get(n_shear)
        w_fe = None if fe is None else float(fe["abs_w_mid_mm"])

        if w_fe is None:
            err_1d = None
            err_2d = None
        else:
            err_1d = (w_beam - w_fe) / w_fe
            err_2d = (w_plate - w_fe) / w_fe

        row = {
            "n_shear": int(n_shear),
            **geometry,
            "HOST_1D_abs_w_mid_mm": float(w_beam),
            "HOST_2D_abs_w_mid_width_average_mm": float(w_plate),
            "HOST_2D_abs_w_mid_center_mm": float(
                plate["final"]["abs_w_mid_center_mm"]
            ),
            "HOST_2D_abs_w_mid_edge_mm": float(
                plate["final"]["abs_w_mid_edge_mm"]
            ),
            "HOST_2D_vs_1D_relative": float(
                (w_plate - w_beam) / w_beam
            ),
            "HOST_2D_last_order_relative_change": float(
                plate["last_order_relative_change"]
            ),
            "FE_3D_abs_w_mid_mm": (
                ""
                if w_fe is None
                else float(w_fe)
            ),
            "HOST_1D_vs_FE_relative": (
                ""
                if err_1d is None
                else float(err_1d)
            ),
            "HOST_2D_vs_FE_relative": (
                ""
                if err_2d is None
                else float(err_2d)
            ),
            "FE_case_hash": (
                ""
                if fe is None
                else fe["case_hash"]
            ),
            "FE_QA": (
                ""
                if fe is None
                else fe["qa_status"]
            ),
            "FE_case_path": (
                ""
                if fe is None
                else fe["path"]
            ),
        }

        rows.append(row)

        print(
            "{:7d} {:9.3f} {:9.3f} {:14.8f} {:14.8f} {:>14s} {:>11s} {:>11s}".format(
                int(n_shear),
                float(geometry["a_over_h"]),
                float(geometry["S_over_h"]),
                float(w_beam),
                float(w_plate),
                (
                    "-"
                    if w_fe is None
                    else "{:.8f}".format(w_fe)
                ),
                (
                    "-"
                    if err_1d is None
                    else "{:+.3%}".format(err_1d)
                ),
                (
                    "-"
                    if err_2d is None
                    else "{:+.3%}".format(err_2d)
                ),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "slenderness_sweep.csv"
    json_path = output_dir / "slenderness_sweep.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema": "host_4pb_slenderness_sweep_v1",
        "source_audit": str(audit_path),
        "fe_run_root": str(run_root),
        "n_shear_values": list(n_shear_values),
        "ritz_orders": [list(v) for v in ritz_orders],
        "fe_campaign_filter": {
            "element_type": args.fe_element_type,
            "core_elements_per_cell_edge": int(args.fe_ne_cell),
            "face_elements_through_thickness": int(args.fe_face_ne_z),
            "fixture_mode": args.fe_fixture_mode,
            "domain_mode": args.fe_domain_mode,
        },
        "rows": rows,
        "plate_details": detailed_plate,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print()
    print("FE cases found: {}/{}".format(
        len(fe_by_n_shear),
        len(n_shear_values),
    ))
    print("Wrote:")
    print(" ", csv_path)
    print(" ", json_path)
    print("=" * 92)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
