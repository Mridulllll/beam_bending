#!/usr/bin/env python3
"""
Canonical production-path verification of the variational two-mode HOST beam.

The script derives the second mode only from section/material properties,
builds the production two-mode section, solves the existing Gyroid RD12
slenderness sweep, and compares against the stored 3-D FE values.

No FE/SVD profile is used to construct the mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from host.beam_point_loads import (
    PointForce,
    solve_simply_supported_point_loads,
)
from host.beam_two_mode import (
    WarpingMode,
    boundary_layer_spectrum,
    build_two_mode_beam_section,
    local_shear_condensation_ratio,
)
from host.beam_two_mode_point_loads import (
    solve_simply_supported_two_mode_point_loads,
)
from host.constitutive import isotropic_stiffness_3d
from host.equilibrium_shear_modes import (
    derive_equilibrium_shear_mode,
)
from host.sandwich_warping import material_informed_symmetric_warping
from host.variational_shear_modes import (
    BeamLayer2D,
    derive_variational_second_shear_mode,
    free_stress_beam_stiffness,
)


DEFAULT_REFERENCE = Path(
    "reference_results/4pb/"
    "Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1/"
    "source_case_manifest.json"
)

DEFAULT_SWEEP = Path(
    "verification/slenderness/results/"
    "gyroid_rd12_initial/slenderness_sweep.json"
)

DEFAULT_OUTPUT = Path(
    "verification/slenderness/results/"
    "gyroid_rd12_initial/production_variational_two_mode"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
    )

    parser.add_argument(
        "--sweep",
        type=Path,
        default=DEFAULT_SWEEP,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
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
        "--face-E-MPa",
        type=float,
        default=70000.0,
    )

    parser.add_argument(
        "--face-nu",
        type=float,
        default=0.33,
    )

    parser.add_argument(
        "--primary-mode",
        choices=(
            "equilibrium",
            "material-informed",
        ),
        default="equilibrium",
        help=(
            "Primary through-thickness shear mode. "
            "'equilibrium' derives f1 from layered beam equilibrium and "
            "shear energy; 'material-informed' retains the previous HOST "
            "parabolic primary mode."
        ),
    )

    parser.add_argument(
        "--max-segment-mm",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()

    reference_path = args.reference.resolve()
    sweep_path = args.sweep.resolve()
    output_dir = args.output_dir.resolve()

    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    if not sweep_path.is_file():
        raise FileNotFoundError(sweep_path)

    reference = load_json(reference_path)
    sweep = load_json(sweep_path)

    Ccore3 = np.asarray(
        reference["constitutive"]["C_hom_model_MPa"],
        dtype=float,
    )

    Cface3 = isotropic_stiffness_3d(
        E=float(args.face_E_MPa),
        nu=float(args.face_nu),
    )

    Ccore = free_stress_beam_stiffness(
        Ccore3,
        axis="x",
    )

    Cface = free_stress_beam_stiffness(
        Cface3,
        axis="x",
    )

    rows_source = [
        row
        for row in sweep["rows"]
        if row.get("FE_3D_abs_w_mid_mm") not in ("", None)
    ]

    if not rows_source:
        raise RuntimeError("Sweep contains no FE-populated rows.")

    first = rows_source[0]

    h = float(first["total_height_mm"])
    hc = float(first["core_height_mm"])
    width = float(first["width_mm"])

    full_layers = (
        BeamLayer2D(
            "face_bottom",
            -0.5 * h,
            -0.5 * hc,
            Cface,
        ),
        BeamLayer2D(
            "core",
            -0.5 * hc,
            +0.5 * hc,
            Ccore,
        ),
        BeamLayer2D(
            "face_top",
            +0.5 * hc,
            +0.5 * h,
            Cface,
        ),
    )

    half_layers = (
        BeamLayer2D(
            "core+",
            0.0,
            0.5 * hc,
            Ccore,
        ),
        BeamLayer2D(
            "face+",
            0.5 * hc,
            0.5 * h,
            Cface,
        ),
    )

    equilibrium_primary = None

    if args.primary_mode == "equilibrium":
        equilibrium_primary = derive_equilibrium_shear_mode(
            full_layers,
            width=width,
        )

        f1 = equilibrium_primary
    else:
        f1 = material_informed_symmetric_warping(
            h=h,
            hc=hc,
            G_face=float(Cface[1, 1]),
            G_core=float(Ccore[1, 1]),
        )

    f2 = derive_variational_second_shear_mode(
        half_layers,
        f1.f,
        elements_per_layer=(
            int(args.core_elements),
            int(args.face_elements),
        ),
    )

    section = build_two_mode_beam_section(
        full_layers,
        width=width,
        mode1=WarpingMode(
            f1.f,
            f1.df,
        ),
        mode2=WarpingMode(
            f2.f,
            f2.df,
        ),
        quadrature_order=8,
        integration_breakpoints=np.unique(
            np.concatenate(
                (
                    -f2.nodes_half[::-1],
                    f2.nodes_half,
                )
            )
        ),
    )

    spectrum = boundary_layer_spectrum(
        section.K5
    )

    # Exact one-mode subspace of the selected primary basis.
    #
    # K6 order:
    #   [eps0, kappa, chi1, chi2, beta1, beta2]
    #
    # Retaining [eps0,kappa,chi1,beta1] and condensing eps0 under N=0
    # produces the K3 expected by the established one-mode HOST solver.
    one_mode_indices = np.asarray(
        [0, 1, 2, 4],
        dtype=int,
    )

    K4_selected = section.K6[
        np.ix_(
            one_mode_indices,
            one_mode_indices,
        )
    ]

    K00_selected = float(
        K4_selected[0, 0]
    )

    K0t_selected = K4_selected[
        0,
        1:,
    ]

    K3_selected = (
        K4_selected[
            1:,
            1:,
        ]
        - np.outer(
            K0t_selected,
            K0t_selected,
        )
        / K00_selected
    )

    K3_selected = 0.5 * (
        K3_selected
        + K3_selected.T
    )

    result_rows = []

    for row in rows_source:
        load_total = 1000.0

        applied_forces = (
            PointForce(
                float(row["load_left_mm"]),
                0.5 * load_total,
            ),
            PointForce(
                float(row["load_right_mm"]),
                0.5 * load_total,
            ),
        )

        solution_one = solve_simply_supported_point_loads(
            K3_selected,
            length=float(row["length_mm"]),
            support_left=float(row["support_left_mm"]),
            support_right=float(row["support_right_mm"]),
            applied_forces=applied_forces,
        )

        solution_two = solve_simply_supported_two_mode_point_loads(
            section.K5,
            length=float(row["length_mm"]),
            support_left=float(row["support_left_mm"]),
            support_right=float(row["support_right_mm"]),
            applied_forces=applied_forces,
            max_segment_length=float(args.max_segment_mm),
        )

        x_mid = 0.5 * float(
            row["length_mm"]
        )

        w_one = abs(
            float(
                solution_one.state(
                    x_mid
                )[0]
            )
        )

        w_two = abs(
            float(
                solution_two.state(
                    x_mid
                )[0]
            )
        )

        w_legacy = float(
            row["HOST_1D_abs_w_mid_mm"]
        )

        w_fe = float(
            row["FE_3D_abs_w_mid_mm"]
        )

        result_rows.append(
            {
                "n_shear": int(row["n_shear"]),
                "a_over_h": float(row["a_over_h"]),
                "HOST_one_mode_abs_w_mid_mm": w_one,
                "HOST_variational_two_mode_abs_w_mid_mm": w_two,
                "HOST_legacy_material_informed_abs_w_mid_mm": w_legacy,
                "FE_3D_abs_w_mid_mm": w_fe,
                "one_mode_error_relative": (
                    (w_one - w_fe) / w_fe
                ),
                "two_mode_error_relative": (
                    (w_two - w_fe) / w_fe
                ),
                "legacy_material_informed_error_relative": (
                    (w_legacy - w_fe) / w_fe
                ),
            }
        )

    e1 = np.asarray(
        [
            row["one_mode_error_relative"]
            for row in result_rows
        ]
    )

    e2 = np.asarray(
        [
            row["two_mode_error_relative"]
            for row in result_rows
        ]
    )

    e_legacy = np.asarray(
        [
            row["legacy_material_informed_error_relative"]
            for row in result_rows
        ]
    )

    summary = {
        "primary_mode": str(args.primary_mode),
        "eigenvalue_per_mm2": float(f2.eigenvalue),
        "section_characteristic_length_mm": float(
            f2.characteristic_length
        ),
        "local_shear_condensation_ratio": float(
            local_shear_condensation_ratio(section.K5)
        ),
        "slowest_boundary_layer_length_mm": float(
            spectrum["stable_modes"][0]["decay_length"]
        ),
        "fastest_boundary_layer_length_mm": float(
            spectrum["stable_modes"][-1]["decay_length"]
        ),
        "RMS_one_mode_percent": float(
            100.0 * np.sqrt(np.mean(e1 ** 2))
        ),
        "RMS_two_mode_percent": float(
            100.0 * np.sqrt(np.mean(e2 ** 2))
        ),
        "max_one_mode_percent": float(
            100.0 * np.max(np.abs(e1))
        ),
        "max_two_mode_percent": float(
            100.0 * np.max(np.abs(e2))
        ),
        "RMS_legacy_material_informed_percent": float(
            100.0 * np.sqrt(np.mean(e_legacy ** 2))
        ),
        "max_legacy_material_informed_percent": float(
            100.0 * np.max(np.abs(e_legacy))
        ),
    }

    if equilibrium_primary is not None:
        summary.update(
            {
                "equilibrium_primary_neutral_axis_mm": float(
                    equilibrium_primary.neutral_axis
                ),
                "equilibrium_primary_D_R_Nmm2": float(
                    equilibrium_primary.reduced_bending_stiffness
                ),
                "equilibrium_primary_S_eq_N": float(
                    equilibrium_primary.equivalent_shear_stiffness
                ),
                "equilibrium_primary_K_eq": float(
                    equilibrium_primary.shear_correction_factor
                ),
                "equilibrium_primary_unit_V_check": float(
                    equilibrium_primary.unit_shear_resultant_check
                ),
            }
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_slug = str(
        args.primary_mode
    ).replace(
        "-",
        "_",
    )

    json_path = (
        output_dir
        / (
            "production_variational_two_mode_"
            + primary_slug
            + ".json"
        )
    )

    csv_path = (
        output_dir
        / (
            "production_variational_two_mode_"
            + primary_slug
            + ".csv"
        )
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema": "host_production_variational_two_mode_v2",
                "reference": str(reference_path),
                "sweep": str(sweep_path),
                "summary": summary,
                "rows": result_rows,
            },
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
            fieldnames=list(result_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    print()
    print("=" * 110)
    print("PRODUCTION VARIATIONAL TWO-MODE HOST VERIFICATION")
    print("=" * 110)
    print(
        "primary shear mode           : {}".format(
            summary["primary_mode"]
        )
    )

    if equilibrium_primary is not None:
        print(
            "equilibrium S_eq [N]       : {:.9f}".format(
                summary["equilibrium_primary_S_eq_N"]
            )
        )
        print(
            "equilibrium K_eq           : {:.9f}".format(
                summary["equilibrium_primary_K_eq"]
            )
        )
        print(
            "equilibrium unit-V check   : {:.12f}".format(
                summary["equilibrium_primary_unit_V_check"]
            )
        )

    print(
        "lambda [1/mm^2]              : {:.9e}".format(
            summary["eigenvalue_per_mm2"]
        )
    )
    print(
        "section ell [mm]             : {:.9f}".format(
            summary["section_characteristic_length_mm"]
        )
    )
    print(
        "local condensed alpha        : {:.9f}".format(
            summary["local_shear_condensation_ratio"]
        )
    )
    print(
        "slow/fast BL length [mm]     : {:.9f} / {:.9f}".format(
            summary["slowest_boundary_layer_length_mm"],
            summary["fastest_boundary_layer_length_mm"],
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
            "err 1mode",
            "err 2mode",
        )
    )

    for row in result_rows:
        print(
            "{:6d} {:8.3f} {:13.8f} {:13.8f} {:13.8f} {:10.3%} {:10.3%}".format(
                row["n_shear"],
                row["a_over_h"],
                row["HOST_one_mode_abs_w_mid_mm"],
                row["HOST_variational_two_mode_abs_w_mid_mm"],
                row["FE_3D_abs_w_mid_mm"],
                row["one_mode_error_relative"],
                row["two_mode_error_relative"],
            )
        )

    print()
    print(
        "RMS error selected 1mode -> 2mode : {:.4f}% -> {:.4f}%".format(
            summary["RMS_one_mode_percent"],
            summary["RMS_two_mode_percent"],
        )
    )

    if args.primary_mode == "equilibrium":
        print(
            "Legacy material-informed 1mode RMS: {:.4f}%".format(
                summary["RMS_legacy_material_informed_percent"]
            )
        )
    print(
        "Max error selected 1mode -> 2mode : {:.4f}% -> {:.4f}%".format(
            summary["max_one_mode_percent"],
            summary["max_two_mode_percent"],
        )
    )
    print()
    print("Wrote:")
    print(" ", json_path)
    print(" ", csv_path)
    print("=" * 110)


if __name__ == "__main__":
    main()
