#!/usr/bin/env python3

"""
Cell-count-based TPMS sandwich-beam design sweep.

The specimen geometry is generated from

    lc, Nh, Nb, Na, Nm, No

rather than from independent millimetre dimensions.

IMPORTANT
---------
The same homogenized C_hom is reused when cell size changes.
This corresponds to ideal geometrically similar scaling of the TPMS
geometry at fixed relative density and base material.

If absolute wall/sheet thickness is kept fixed while cell size changes,
a new homogenization calculation is required.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from host.beam import (
    build_three_layer_host_beam,
    free_beam_constitutive,
)

from host.beam_bending import (
    condense_zero_axial_force,
)

from host.beam_cell_design import (
    CellCountBeamDesign,
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


def parse_floats(
    text,
):
    return [
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    ]


def parse_ints(
    text,
):
    values = []

    for raw in text.split(","):
        raw = raw.strip()

        if not raw:
            continue

        value = int(
            raw
        )

        values.append(
            value
        )

    return values


def classical_D(
    K4,
):
    return float(
        K4[1, 1]
        - (
            K4[1, 0]
            * K4[0, 1]
            / K4[0, 0]
        )
    )


def eb_four_point(
    P,
    S,
    SL,
    D,
):
    a = 0.5 * (
        S - SL
    )

    return (
        P
        * a
        * (
            3.0 * S**2
            - 4.0 * a**2
        )
        / (
            48.0 * D
        )
    )


def solve_midspan(
    K3,
    design,
    P,
):
    xA, xB = symmetric_support_positions(
        specimen_length=design.length,
        support_span=design.support_span,
    )

    loads = symmetric_four_point_bending_loads(
        total_load=P,
        specimen_length=design.length,
        load_span=design.load_span,
    )

    # Unit-cell parameterization and generic fixture geometry must
    # produce exactly the same coordinates.
    np.testing.assert_allclose(
        [
            xA,
            xB,
        ],
        [
            design.support_left,
            design.support_right,
        ],
        rtol=0.0,
        atol=1.0e-11,
    )

    np.testing.assert_allclose(
        [
            loads[0].x,
            loads[1].x,
        ],
        [
            design.load_left,
            design.load_right,
        ],
        rtol=0.0,
        atol=1.0e-11,
    )

    solution = solve_simply_supported_point_loads(
        K3,
        length=design.length,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    state = solution.state(
        0.5 * design.length
    )

    return {
        "w": abs(
            float(
                state[0]
            )
        ),
        "M": float(
            state[3]
        ),
        "V": float(
            state[4]
        ),
        "H": float(
            state[5]
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
    )

    parser.add_argument(
        "--axis",
        choices=(
            "x",
            "y",
        ),
        default="x",
    )

    parser.add_argument(
        "--cell-sizes",
        default="7.5,10,12.5",
    )

    parser.add_argument(
        "--height-cells",
        default="6",
    )

    parser.add_argument(
        "--width-cells",
        default="6",
    )

    parser.add_argument(
        "--shear-cells",
        default="6,8,10,12",
    )

    parser.add_argument(
        "--moment-cells",
        default="4,6,8",
    )

    parser.add_argument(
        "--overhang-cells",
        default="2",
    )

    parser.add_argument(
        "--face-thickness",
        type=float,
        default=1.0,
        help=(
            "Fixed face thickness [mm]. "
            "Ignored if --face-thickness-ratio is supplied."
        ),
    )

    parser.add_argument(
        "--face-thickness-ratio",
        type=float,
        default=None,
        help=(
            "Optional geometrically scaled face thickness hf/lc."
        ),
    )

    parser.add_argument(
        "--face-boundary-cells",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--minimum-interior-height-cells",
        type=int,
        default=4,
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
        "--output",
        default=(
            "verification/"
            "beam_bending_design/"
            "tpms_cell_count_sweep.csv"
        ),
    )

    args = parser.parse_args()

    cell_sizes = parse_floats(
        args.cell_sizes
    )

    height_cells = parse_ints(
        args.height_cells
    )

    width_cells = parse_ints(
        args.width_cells
    )

    shear_cells = parse_ints(
        args.shear_cells
    )

    moment_cells = parse_ints(
        args.moment_cells
    )

    overhang_cells = parse_ints(
        args.overhang_cells
    )

    record = load_tpms_homogenization_json(
        args.json_file
    )

    C_core = np.asarray(
        record.C_hom,
        dtype=float,
    )

    C_face = isotropic_stiffness_3d(
        E=float(
            args.face_E
        ),
        nu=float(
            args.face_nu
        ),
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
    skipped_height = 0

    combinations = product(
        cell_sizes,
        height_cells,
        width_cells,
        shear_cells,
        moment_cells,
        overhang_cells,
    )

    for (
        lc,
        Nh,
        Nb,
        Na,
        Nm,
        No,
    ) in combinations:

        if (
            args.face_thickness_ratio
            is None
        ):
            hf = float(
                args.face_thickness
            )

            face_mode = (
                "fixed_mm"
            )

        else:
            hf = (
                float(
                    args.face_thickness_ratio
                )
                * lc
            )

            face_mode = (
                "scaled_with_cell"
            )

        design = CellCountBeamDesign(
            cell_size=lc,
            n_height=Nh,
            n_width=Nb,
            n_shear=Na,
            n_moment=Nm,
            n_overhang=No,
            face_bottom=hf,
            face_top=hf,
            face_boundary_cells=(
                args.face_boundary_cells
            ),
        )

        if not design.height_homogenization_ok(
            minimum_interior_cells=(
                args.minimum_interior_height_cells
            )
        ):
            skipped_height += 1
            continue

        geometry = design.beam_geometry()

        h = design.total_height

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
                hc=design.core_height,
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

        w_eb = eb_four_point(
            args.load,
            design.support_span,
            design.load_span,
            D,
        )

        reddy = solve_midspan(
            K3_reddy,
            design,
            args.load,
        )

        host = solve_midspan(
            K3_host,
            design,
            args.load,
        )

        rows.append(
            {
                "axis": args.axis,
                "cell_size_mm": lc,

                "N_height": Nh,
                "N_width": Nb,
                "N_shear": Na,
                "N_moment": Nm,
                "N_overhang": No,

                "face_boundary_cells": (
                    args.face_boundary_cells
                ),

                "interior_height_cells": (
                    design.height_interior_cells
                ),

                "face_mode": face_mode,
                "face_thickness_mm": hf,
                "face_thickness_over_cell": (
                    hf / lc
                ),

                "core_height_mm": (
                    design.core_height
                ),

                "width_mm": (
                    design.width
                ),

                "shear_span_mm": (
                    design.shear_span
                ),

                "load_span_mm": (
                    design.load_span
                ),

                "support_span_mm": (
                    design.support_span
                ),

                "overhang_mm": (
                    design.overhang
                ),

                "length_mm": (
                    design.length
                ),

                "total_height_mm": (
                    design.total_height
                ),

                "S_over_h": (
                    design.S_over_h
                ),

                "a_over_h": (
                    design.a_over_h
                ),

                "SL_over_h": (
                    design.load_span_over_h
                ),

                "L_over_h": (
                    design.length
                    / design.total_height
                ),

                "w_EB_mm": w_eb,

                "w_Reddy_mm": (
                    reddy["w"]
                ),

                "w_HOST_mm": (
                    host["w"]
                ),

                "HOST_minus_Reddy_pct": (
                    100.0
                    * (
                        host["w"]
                        / reddy["w"]
                        - 1.0
                    )
                ),

                "HOST_minus_EB_pct": (
                    100.0
                    * (
                        host["w"]
                        / w_eb
                        - 1.0
                    )
                ),

                "non_EB_fraction_pct": (
                    100.0
                    * (
                        host["w"]
                        - w_eb
                    )
                    / host["w"]
                ),

                "HOST_compliance_mm_per_N": (
                    host["w"]
                    / args.load
                ),

                "midspan_M_Nmm": (
                    host["M"]
                ),

                "midspan_V_N": (
                    host["V"]
                ),

                "midspan_H": (
                    host["H"]
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            "Cell-count sweep produced no valid designs."
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

    ranked = sorted(
        rows,
        key=lambda row: abs(
            row[
                "HOST_minus_Reddy_pct"
            ]
        ),
        reverse=True,
    )

    print()
    print("=" * 125)
    print("CELL-COUNT TPMS FOUR-POINT-BENDING STUDY")
    print("=" * 125)

    print(
        "Geometric-scaling assumption for C_hom : "
        "same homogenized tensor reused across cell sizes"
    )

    print(
        "Height screening                      : "
        "{} face-affected cell(s) each side, "
        "minimum {} interior cell(s)".format(
            args.face_boundary_cells,
            args.minimum_interior_height_cells,
        )
    )

    print(
        "Rejected by height criterion           :",
        skipped_height,
    )

    print()

    print(
        "{:>5s} {:>3s} {:>3s} {:>3s} {:>3s} "
        "{:>6s} {:>6s} {:>6s} {:>6s} "
        "{:>8s} {:>9s} {:>9s}".format(
            "lc",
            "Nh",
            "Na",
            "Nm",
            "No",
            "hc",
            "L",
            "S/h",
            "a/h",
            "HOST/R%",
            "HOST/EB%",
            "nonEB%",
        )
    )

    for row in ranked[:30]:

        print(
            "{:5.1f} {:3d} {:3d} {:3d} {:3d} "
            "{:6.1f} {:6.1f} {:6.3f} {:6.3f} "
            "{:8.4f} {:9.2f} {:9.2f}".format(
                row["cell_size_mm"],
                row["N_height"],
                row["N_shear"],
                row["N_moment"],
                row["N_overhang"],
                row["core_height_mm"],
                row["length_mm"],
                row["S_over_h"],
                row["a_over_h"],
                row[
                    "HOST_minus_Reddy_pct"
                ],
                row[
                    "HOST_minus_EB_pct"
                ],
                row[
                    "non_EB_fraction_pct"
                ],
            )
        )

    print()
    print(
        "CSV:",
        output,
    )


if __name__ == "__main__":
    main()