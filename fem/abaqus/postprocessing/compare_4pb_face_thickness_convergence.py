# -*- coding: utf-8 -*-
"""
Compare HOST homogenized 4PB cases for face-sheet through-thickness convergence.

Run with the normal project Python environment after each case has been
postprocessed with postprocess_4pb_homogenized.py v2 or newer:

    python fem/abaqus/postprocessing/compare_4pb_face_thickness_convergence.py ^
        --case "<face ne=1 case>" ^
        --case "<face ne=2 case>" ^
        --case "<face ne=4 case>"

The comparison keeps the core/in-plane discretization and element formulation
fixed and varies only the number of solid elements through each face sheet.

Local face S11/E11 values are compared at the SAME physical outer-surface
coordinate. They are obtained from a least-squares linear fit through the
width-averaged integration-point profile at midspan and evaluated at z_top or
z_bottom. This avoids the moving-Gauss-point bias that occurs when simply
selecting the outermost element layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


SCHEMA = "host_4pb_face_thickness_convergence_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare HOST homogenized 4PB face-sheet through-thickness "
            "discretization."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case directory. Pass once per face-through-thickness mesh level.",
    )
    parser.add_argument(
        "--reference-face-ne",
        type=int,
        default=None,
        help="Optional reference number of face elements through thickness.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <common parent>/face_thickness_convergence.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key, default=math.nan):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def relative_difference(value, reference):
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def linear_fit_value_at_z(rows, field_name, z_target):
    if not rows:
        return math.nan

    z_field = (
        "z_ip_mm"
        if "z_ip_mm" in rows[0]
        else "z_element_centroid_mm"
    )

    pairs = []
    for row in rows:
        z = as_float(row, z_field)
        value = as_float(row, field_name)

        if math.isfinite(z) and math.isfinite(value):
            pairs.append((z, value))

    if not pairs:
        return math.nan

    if len(pairs) == 1:
        return pairs[0][1]

    z_mean = sum(z for z, _ in pairs) / len(pairs)
    y_mean = sum(value for _, value in pairs) / len(pairs)

    denominator = sum((z - z_mean) ** 2 for z, _ in pairs)

    if denominator <= 1.0e-30:
        return y_mean

    slope = (
        sum(
            (z - z_mean) * (value - y_mean)
            for z, value in pairs
        )
        / denominator
    )
    intercept = y_mean - slope * z_mean

    return intercept + slope * float(z_target)


def outer_face_value(profile_rows, geometry, face_name, field_name):
    rows = [
        row
        for row in profile_rows
        if row["station"] == "midspan" and row["region"] == face_name
    ]

    if face_name == "FACE_TOP":
        z_target = float(geometry["z_top"])
    elif face_name == "FACE_BOTTOM":
        z_target = float(geometry["z_bottom"])
    else:
        raise ValueError(
            "Expected FACE_TOP or FACE_BOTTOM, got {!r}.".format(face_name)
        )

    return linear_fit_value_at_z(
        rows,
        field_name,
        z_target,
    )


def read_case(case_path):
    case_path = Path(case_path).resolve()

    manifest = load_json(case_path / "manifest.json")
    summary = load_json(case_path / "post" / "summary.json")
    profiles = load_csv(
        case_path / "post" / "through_thickness_profiles.csv"
    )

    post_schema = str(summary.get("schema", ""))

    if post_schema != "host_4pb_homogenized_post_v2":
        raise ValueError(
            "{} was postprocessed with schema {!r}; rerun the current "
            "postprocess_4pb_homogenized.py so integration-point z coordinates "
            "are available."
            .format(case_path, post_schema)
        )

    geometry = manifest["geometry"]

    return {
        "case_path": str(case_path),
        "case_id": manifest["case"]["case_id"],
        "case_hash": manifest["case"]["case_hash"],
        "element_type": str(manifest["mesh"]["element_type"]),
        "core_ne_cell": int(
            manifest["mesh"]["core_elements_per_cell_edge"]
        ),
        "face_ne_z": int(
            manifest["mesh"]["face_elements_through_thickness"]
        ),
        "actual_element_count": int(
            manifest["mesh"]["derived_counts"]["actual_element_count"]
        ),
        "actual_node_count": int(
            manifest["mesh"]["derived_counts"]["actual_node_count"]
        ),
        "midspan_U3_mm": float(
            summary["midspan"]["U3_midplane_avg_mm"]
        ),
        "SENER_Nmm": float(
            summary["energy"]["SENER_integral_Nmm"]
        ),
        "reaction_relative_error": float(
            summary["equilibrium"]["vertical_relative_error"]
        ),
        "top_outer_surface_S11_MPa": outer_face_value(
            profiles,
            geometry,
            "FACE_TOP",
            "S11_MPa",
        ),
        "bottom_outer_surface_S11_MPa": outer_face_value(
            profiles,
            geometry,
            "FACE_BOTTOM",
            "S11_MPa",
        ),
        "top_outer_surface_E11": outer_face_value(
            profiles,
            geometry,
            "FACE_TOP",
            "E11",
        ),
        "bottom_outer_surface_E11": outer_face_value(
            profiles,
            geometry,
            "FACE_BOTTOM",
            "E11",
        ),
        "post_status": summary["overall_status"],
    }


def validate_family(records):
    element_types = {record["element_type"] for record in records}
    core_ne_values = {record["core_ne_cell"] for record in records}

    if len(element_types) != 1:
        raise ValueError(
            "Face convergence requires one element formulation; got {}."
            .format(sorted(element_types))
        )

    if len(core_ne_values) != 1:
        raise ValueError(
            "Face convergence requires fixed core/in-plane ne; got {}."
            .format(sorted(core_ne_values))
        )

    face_levels = [record["face_ne_z"] for record in records]

    if len(set(face_levels)) != len(face_levels):
        raise ValueError(
            "Duplicate face-elements-through-thickness levels detected."
        )


def add_reference_errors(records, reference):
    metrics = [
        "midspan_U3_mm",
        "SENER_Nmm",
        "top_outer_surface_S11_MPa",
        "bottom_outer_surface_S11_MPa",
        "top_outer_surface_E11",
        "bottom_outer_surface_E11",
    ]

    for record in records:
        record["reference_face_ne_z"] = reference["face_ne_z"]
        record["reference_case_hash"] = reference["case_hash"]

        for metric in metrics:
            value = record[metric]
            ref = reference[metric]

            if math.isnan(value) or math.isnan(ref):
                error = math.nan
            else:
                error = relative_difference(value, ref)

            record[metric + "_rel_to_reference"] = error


def write_csv(path, records):
    fieldnames = []

    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_values(records, output_path):
    records = sorted(records, key=lambda item: item["face_ne_z"])
    x = [record["face_ne_z"] for record in records]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

    axes[0].plot(
        x,
        [record["midspan_U3_mm"] for record in records],
        marker="o",
    )
    axes[0].set_ylabel("Midspan U3 [mm]")
    axes[0].grid(True, linewidth=0.4)

    axes[1].plot(
        x,
        [record["SENER_Nmm"] for record in records],
        marker="o",
    )
    axes[1].set_xlabel("Face elements through thickness")
    axes[1].set_ylabel("Elastic strain energy [N mm]")
    axes[1].set_xticks(x)
    axes[1].grid(True, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def plot_local_face_values(records, output_path):
    records = sorted(records, key=lambda item: item["face_ne_z"])
    x = [record["face_ne_z"] for record in records]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

    axes[0].plot(
        x,
        [record["top_outer_surface_S11_MPa"] for record in records],
        marker="o",
        label="Top",
    )
    axes[0].plot(
        x,
        [record["bottom_outer_surface_S11_MPa"] for record in records],
        marker="o",
        label="Bottom",
    )
    axes[0].set_ylabel("Outer-surface S11 [MPa]")
    axes[0].grid(True, linewidth=0.4)
    axes[0].legend()

    axes[1].plot(
        x,
        [record["top_outer_surface_E11"] for record in records],
        marker="o",
        label="Top",
    )
    axes[1].plot(
        x,
        [record["bottom_outer_surface_E11"] for record in records],
        marker="o",
        label="Bottom",
    )
    axes[1].set_xlabel("Face elements through thickness")
    axes[1].set_ylabel("Outer-surface E11 [-]")
    axes[1].set_xticks(x)
    axes[1].grid(True, linewidth=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def plot_relative_errors(records, output_path):
    records = sorted(records, key=lambda item: item["face_ne_z"])
    x = [record["face_ne_z"] for record in records]

    metrics = [
        ("midspan_U3_mm_rel_to_reference", "|w_mid|"),
        ("SENER_Nmm_rel_to_reference", "U"),
        ("top_outer_surface_S11_MPa_rel_to_reference", "top S11(z_top)"),
        ("top_outer_surface_E11_rel_to_reference", "top E11(z_top)"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for metric, label in metrics:
        y = [record[metric] for record in records]
        ax.plot(x, y, marker="o", label=label)

    ax.set_xlabel("Face elements through thickness")
    ax.set_ylabel("Relative difference to reference")
    ax.set_xticks(x)
    ax.set_yscale("log")
    ax.grid(True, linewidth=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def main():
    args = parse_args()

    records = [read_case(path) for path in args.case]

    if len(records) < 2:
        raise ValueError("At least two cases are required.")

    validate_family(records)
    records = sorted(records, key=lambda item: item["face_ne_z"])

    if args.reference_face_ne is None:
        reference = max(records, key=lambda item: item["face_ne_z"])
    else:
        matches = [
            record
            for record in records
            if record["face_ne_z"] == args.reference_face_ne
        ]

        if len(matches) != 1:
            raise ValueError(
                "Reference face ne={} not found exactly once."
                .format(args.reference_face_ne)
            )

        reference = matches[0]

    add_reference_errors(records, reference)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        common_parent = Path(
            os.path.commonpath(
                [
                    str(Path(record["case_path"]).parent)
                    for record in records
                ]
            )
        )
        output_dir = common_parent / "face_thickness_convergence"

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema": SCHEMA,
        "element_type": records[0]["element_type"],
        "core_ne_cell": records[0]["core_ne_cell"],
        "reference": {
            "face_ne_z": reference["face_ne_z"],
            "case_hash": reference["case_hash"],
            "case_path": reference["case_path"],
        },
        "cases": records,
        "note": (
            "Local face stress/strain values are evaluated at the same physical "
            "outer-surface z by linear least-squares extrapolation from the "
            "width-averaged integration-point profile."
        ),
    }

    json_path = output_dir / "face_thickness_convergence.json"
    csv_path = output_dir / "face_thickness_convergence.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    write_csv(csv_path, records)

    figures = [
        plot_values(
            records,
            output_dir / "global_response_convergence.png",
        ),
        plot_local_face_values(
            records,
            output_dir / "outer_surface_face_response_convergence.png",
        ),
        plot_relative_errors(
            records,
            output_dir / "relative_error_convergence.png",
        ),
    ]

    print("=" * 86)
    print("HOST 4PB FACE-SHEET THROUGH-THICKNESS CONVERGENCE")
    print("=" * 86)
    print("Element type : {}".format(records[0]["element_type"]))
    print("Core ne/cell : {}".format(records[0]["core_ne_cell"]))
    print(
        "Reference    : face ne={} ({})".format(
            reference["face_ne_z"],
            reference["case_hash"],
        )
    )
    print()

    for record in records:
        print(
            "face_ne={:<2d}  U3_mid={: .10g} mm  dU3={:.3e}  "
            "SENER={:.10g} Nmm  dU={:.3e}  "
            "top S11(z_top)={: .8g} MPa  dS11={:.3e}".format(
                record["face_ne_z"],
                record["midspan_U3_mm"],
                record["midspan_U3_mm_rel_to_reference"],
                record["SENER_Nmm"],
                record["SENER_Nmm_rel_to_reference"],
                record["top_outer_surface_S11_MPa"],
                record["top_outer_surface_S11_MPa_rel_to_reference"],
            )
        )

    print()
    print("Wrote:")
    for path in (
        json_path,
        csv_path,
        output_dir / "global_response_convergence.png",
        output_dir / "outer_surface_face_response_convergence.png",
        output_dir / "relative_error_convergence.png",
    ):
        print("  {}".format(path))

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
