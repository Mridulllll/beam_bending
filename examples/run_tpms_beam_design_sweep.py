#!/usr/bin/env python3

"""
HOST four-point-bending specimen design sweep.

The purpose is not to prescribe a validity threshold, but to expose
the important dimensionless fixture ratios and the separation between

    Euler-Bernoulli,
    Reddy HSDT,
    material-informed HOST.

Output is ranked by the absolute HOST/Reddy difference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from host.beam import (
    SandwichBeamGeometry,
    build_three_layer_host_beam,
    free_beam_constitutive,
)

from host.beam_bending import (
    condense_zero_axial_force,
)

from host.beam_point_loads import (
    solve_simply_supported_point_loads,
    symmetric_four_point_bending_loads,
    symmetric_support_positions,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.sandwich_warping import (
    material_informed_generalized_warping,
)

from host.tpms_io import (
    load_tpms_homogenization_json,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def parse_values(text):
    return [
        float(value)
        for value in text.split(",")
        if value.strip()
    ]


def eb_four_point_midspan(
    P,
    support_span,
    load_span,
    D,
):
    a = 0.5 * (
        support_span
        - load_span
    )

    return (
        P
        * a
        * (
            3.0 * support_span**2
            - 4.0 * a**2
        )
        / (
            48.0 * D
        )
    )


def classical_D(K4):
    return float(
        K4[1, 1]
        - K4[1, 0]
        * K4[0, 1]
        / K4[0, 0]
    )


def solve_midspan(
    K3,
    *,
    length,
    support_span,
    load_span,
    load,
):
    xA, xB = symmetric_support_positions(
        specimen_length=length,
        support_span=support_span,
    )

    forces = symmetric_four_point_bending_loads(
        total_load=load,
        specimen_length=length,
        load_span=load_span,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=length,
        support_left=xA,
        support_right=xB,
        applied_forces=forces,
    )

    return abs(
        float(
            solution.state(
                0.5 * length
            )[0]
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
    )

    parser.add_argument(
        "--axis",
        choices=("x", "y"),
        default="x",
    )

    parser.add_argument(
        "--length",
        type=float,
        default=240.0,
    )

    parser.add_argument(
        "--widths",
        default="40,50,60",
    )

    parser.add_argument(
        "--core-heights",
        default="20,30,40,50,60",
    )

    parser.add_argument(
        "--support-spans",
        default="160,180,200,220",
    )

    parser.add_argument(
        "--load-span-ratios",
        default="0.2,0.3,0.4,0.5",
        help="load span / support span",
    )

    parser.add_argument(
        "--face-thickness",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--load",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--face-E",
        type=float,
        default=70000.0,
    )

    parser.add_argument(
        "--face-nu",
        type=float,
        default=0.33,
    )

    parser.add_argument(
        "--cell-size",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--output",
        default=(
            "verification/"
            "beam_bending_design/"
            "tpms_four_point_design_sweep.csv"
        ),
    )

    args = parser.parse_args()

    widths = parse_values(
        args.widths
    )

    core_heights = parse_values(
        args.core_heights
    )

    support_spans = parse_values(
        args.support_spans
    )

    load_span_ratios = parse_values(
        args.load_span_ratios
    )

    L = float(
        args.length
    )

    hf = float(
        args.face_thickness
    )

    P = float(
        args.load
    )

    record = load_tpms_homogenization_json(
        args.json_file
    )

    C_core = np.asarray(
        record.C_hom,
        dtype=float,
    )

    C_face = isotropic_stiffness_3d(
        E=float(args.face_E),
        nu=float(args.face_nu),
    )

    Cb_core = free_beam_constitutive(
        C_core,
        axis=args.axis,
    )

    Cb_face = free_beam_constitutive(
        C_face,
        axis=args.axis,
    )

    G_core = float(
        Cb_core[1, 1]
    )

    G_face = float(
        Cb_face[1, 1]
    )

    rows = []

    for width in widths:

        for hc in core_heights:

            geometry = SandwichBeamGeometry(
                length=L,
                width=width,
                core_height=hc,
                face_bottom=hf,
                face_top=hf,
            )

            h = geometry.total_height

            reddy_warping = (
                reddy_baseline_generalized_warping(
                    h
                )
            )

            reddy_beam = (
                build_three_layer_host_beam(
                    C_bottom=C_face,
                    C_core=C_core,
                    C_top=C_face,
                    geometry=geometry,
                    axis=args.axis,
                    warping=reddy_warping,
                    quadrature_order=12,
                )
            )

            host_warping = (
                material_informed_generalized_warping(
                    h=h,
                    hc=hc,
                    G_face=G_face,
                    G_core=G_core,
                )
            )

            host_beam = (
                build_three_layer_host_beam(
                    C_bottom=C_face,
                    C_core=C_core,
                    C_top=C_face,
                    geometry=geometry,
                    axis=args.axis,
                    warping=host_warping,
                    quadrature_order=12,
                )
            )

            K3_reddy = (
                condense_zero_axial_force(
                    reddy_beam.K4
                ).K3
            )

            K3_host = (
                condense_zero_axial_force(
                    host_beam.K4
                ).K3
            )

            D = classical_D(
                reddy_beam.K4
            )

            for support_span in support_spans:

                if support_span > L:
                    continue

                overhang = (
                    0.5
                    * (
                        L - support_span
                    )
                )

                for ratio in load_span_ratios:

                    load_span = (
                        ratio
                        * support_span
                    )

                    if (
                        load_span <= 0.0
                        or load_span >= support_span
                    ):
                        continue

                    shear_span = (
                        0.5
                        * (
                            support_span
                            - load_span
                        )
                    )

                    w_eb = eb_four_point_midspan(
                        P,
                        support_span,
                        load_span,
                        D,
                    )

                    w_reddy = solve_midspan(
                        K3_reddy,
                        length=L,
                        support_span=support_span,
                        load_span=load_span,
                        load=P,
                    )

                    w_host = solve_midspan(
                        K3_host,
                        length=L,
                        support_span=support_span,
                        load_span=load_span,
                        load=P,
                    )

                    host_reddy = (
                        100.0
                        * (
                            w_host / w_reddy
                            - 1.0
                        )
                    )

                    host_eb = (
                        100.0
                        * (
                            w_host / w_eb
                            - 1.0
                        )
                    )

                    non_eb_fraction = (
                        100.0
                        * (
                            w_host
                            - w_eb
                        )
                        / w_host
                    )

                    rows.append(
                        {
                            "axis": args.axis,
                            "length_mm": L,
                            "width_mm": width,
                            "core_height_mm": hc,
                            "face_thickness_mm": hf,
                            "total_height_mm": h,
                            "support_span_mm": support_span,
                            "load_span_mm": load_span,
                            "shear_span_mm": shear_span,
                            "overhang_mm": overhang,
                            "L_over_h": L / h,
                            "S_over_h": support_span / h,
                            "shear_span_over_h": (
                                shear_span / h
                            ),
                            "load_span_over_h": (
                                load_span / h
                            ),
                            "b_over_h": width / h,
                            "cells_width": (
                                width
                                / args.cell_size
                            ),
                            "cells_core_height": (
                                hc
                                / args.cell_size
                            ),
                            "cells_shear_span": (
                                shear_span
                                / args.cell_size
                            ),
                            "w_EB_mm": w_eb,
                            "w_Reddy_mm": w_reddy,
                            "w_HOST_mm": w_host,
                            "HOST_minus_Reddy_pct": (
                                host_reddy
                            ),
                            "HOST_minus_EB_pct": host_eb,
                            "non_EB_fraction_pct": (
                                non_eb_fraction
                            ),
                            "HOST_compliance_mm_per_N": (
                                w_host / P
                            ),
                            "width_normalized_compliance": (
                                width
                                * w_host
                                / P
                            ),
                        }
                    )


    # ==============================================================
    # WIDTH-SCALING CONSISTENCY CHECK
    #
    # For a homogenized 1-D beam, K scales linearly with width.
    # Therefore width * w must be invariant for otherwise identical
    # geometry and loading. Failure indicates a numerical solver
    # problem and the sweep must not be interpreted physically.
    # ==============================================================

    # WIDTH-SCALING CONSISTENCY CHECK

    groups = {}

    for row in rows:

        key = (
            row["axis"],
            row["core_height_mm"],
            row["face_thickness_mm"],
            row["support_span_mm"],
            row["load_span_mm"],
        )

        groups.setdefault(
            key,
            []
        ).append(
            row
        )

    max_width_spread = 0.0

    for key, group in groups.items():

        if len(group) < 2:
            continue

        for field in (
            "w_EB_mm",
            "w_Reddy_mm",
            "w_HOST_mm",
        ):

            normalized = np.asarray(
                [
                    item["width_mm"]
                    * item[field]
                    for item in group
                ],
                dtype=float,
            )

            if not np.all(
                np.isfinite(
                    normalized
                )
            ):
                raise RuntimeError(
                    "Non-finite beam response encountered "
                    "during width-scaling validation."
                )

            reference = max(
                1.0e-30,
                abs(
                    float(
                        np.mean(
                            normalized
                        )
                    )
                ),
            )

            relative_spread = (
                float(
                    np.max(
                        normalized
                    )
                    - np.min(
                        normalized
                    )
                )
                / reference
            )

            max_width_spread = max(
                max_width_spread,
                abs(
                    relative_spread
                ),
            )

            if abs(
                relative_spread
            ) > 1.0e-6:

                raise RuntimeError(
                    "Width-scaling consistency failed for "
                    "{} and field {}: relative spread = {:.6e}".format(
                        key,
                        field,
                        relative_spread,
                    )
                )

    for row in rows:

        for field in (
            "w_EB_mm",
            "w_Reddy_mm",
            "w_HOST_mm",
        ):

            value = float(
                row[field]
            )

            if (
                not np.isfinite(
                    value
                )
                or value <= 0.0
            ):
                raise RuntimeError(
                    "Invalid positive-deflection response: "
                    "{} = {}".format(
                        field,
                        value,
                    )
                )

    print()
    print(
        "Maximum width-normalized response spread: "
        "{:.6e}".format(
            max_width_spread
        )
    )

    rows.sort(
        key=lambda row: abs(
            row[
                "HOST_minus_Reddy_pct"
            ]
        ),
        reverse=True,
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
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

    print()
    print("=" * 105)
    print("Top cases ranked by |HOST - Reddy|")
    print("=" * 105)

    print(
        "{:>5s} {:>5s} {:>6s} {:>6s} {:>7s} {:>7s} {:>9s} {:>10s} {:>10s}".format(
            "b",
            "hc",
            "S",
            "SL",
            "S/h",
            "a/h",
            "HOST/R %",
            "HOST/EB %",
            "nonEB %",
        )
    )

    for row in rows[:25]:

        print(
            "{:5.0f} {:5.0f} {:6.0f} {:6.0f} {:7.3f} {:7.3f} {:9.4f} {:10.2f} {:10.2f}".format(
                row["width_mm"],
                row["core_height_mm"],
                row["support_span_mm"],
                row["load_span_mm"],
                row["S_over_h"],
                row["shear_span_over_h"],
                row["HOST_minus_Reddy_pct"],
                row["HOST_minus_EB_pct"],
                row["non_EB_fraction_pct"],
            )
        )

    print()
    print(
        "CSV:",
        output,
    )


if __name__ == "__main__":
    main()