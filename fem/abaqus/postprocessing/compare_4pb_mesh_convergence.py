# -*- coding: utf-8 -*-
"""
Compare a set of standardized HOST 4PB homogenized FE cases for mesh convergence.

Run with the normal project Python environment:

    python fem/abaqus/postprocessing/compare_4pb_mesh_convergence.py ^
        --case "<ne=1 case>" ^
        --case "<ne=2 case>" ^
        --case "<ne=4 case>"

Each case must already have been processed with:
    postprocess_4pb_homogenized.py

The finest core mesh (largest core_elements_per_cell_edge) is used as the
reference unless --reference-ne is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCHEMA = "host_4pb_homogenized_mesh_convergence_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HOST homogenized 4PB FE cases for mesh convergence."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case directory. Pass once per mesh level.",
    )
    parser.add_argument(
        "--reference-ne",
        type=int,
        default=None,
        help="Optional reference core elements per cell edge.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <common parent>/mesh_convergence.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving.",
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
    """
    Least-squares linear fit of a width-averaged face profile versus z.

    C3D20R provides multiple through-thickness integration-point levels even
    with one element through a face. Evaluating the fitted profile at the same
    physical outer-surface coordinate makes face stress/strain comparisons
    independent of the layer-dependent Gauss-point locations.
    """
    if not rows:
        return math.nan

    z_field = (
        "z_ip_mm"
        if "z_ip_mm" in rows[0]
        else "z_element_centroid_mm"
    )

    pairs = [
        (
            as_float(row, z_field),
            as_float(row, field_name),
        )
        for row in rows
    ]
    pairs = [
        (z, value)
        for z, value in pairs
        if math.isfinite(z) and math.isfinite(value)
    ]

    if not pairs:
        return math.nan

    if len(pairs) == 1:
        return pairs[0][1]

    z_mean = sum(z for z, _ in pairs) / len(pairs)
    y_mean = sum(value for _, value in pairs) / len(pairs)

    denominator = sum(
        (z - z_mean) ** 2
        for z, _ in pairs
    )

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


def select_midspan_outer_face_metric(
    profile_rows,
    face_name,
    field_name,
    geometry,
):
    rows = [
        row
        for row in profile_rows
        if row["station"] == "midspan" and row["region"] == face_name
    ]

    if not rows:
        return math.nan

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


def select_left_shear_core_average(profile_rows, field_name):
    rows = [
        row
        for row in profile_rows
        if row["station"] == "left_shear_mid" and row["region"] == "CORE"
    ]

    if not rows:
        return math.nan

    total_volume = sum(as_float(row, "volume_mm3", 0.0) for row in rows)

    if total_volume <= 0.0:
        return math.nan

    return (
        sum(
            as_float(row, field_name) * as_float(row, "volume_mm3", 0.0)
            for row in rows
        )
        / total_volume
    )


def read_case(case_path):
    case_path = Path(case_path).resolve()
    manifest = load_json(case_path / "manifest.json")
    summary = load_json(case_path / "post" / "summary.json")
    profiles = load_csv(case_path / "post" / "through_thickness_profiles.csv")

    ne = int(manifest["mesh"]["core_elements_per_cell_edge"])
    element_type = str(manifest["mesh"]["element_type"])

    return {
        "case_path": str(case_path),
        "case_id": manifest["case"]["case_id"],
        "case_hash": manifest["case"]["case_hash"],
        "element_type": element_type,
        "ne_cell": ne,
        "actual_element_count": int(
            manifest["mesh"]["derived_counts"]["actual_element_count"]
        ),
        "actual_node_count": int(
            manifest["mesh"]["derived_counts"]["actual_node_count"]
        ),
        "midspan_U3_mm": float(summary["midspan"]["U3_midplane_avg_mm"]),
        "SENER_Nmm": float(summary["energy"]["SENER_integral_Nmm"]),
        "reaction_relative_error": float(
            summary["equilibrium"]["vertical_relative_error"]
        ),
        "top_outer_midspan_S11_MPa": select_midspan_outer_face_metric(
            profiles,
            "FACE_TOP",
            "S11_MPa",
            manifest["geometry"],
        ),
        "bottom_outer_midspan_S11_MPa": select_midspan_outer_face_metric(
            profiles,
            "FACE_BOTTOM",
            "S11_MPa",
            manifest["geometry"],
        ),
        "top_outer_midspan_E11": select_midspan_outer_face_metric(
            profiles,
            "FACE_TOP",
            "E11",
            manifest["geometry"],
        ),
        "bottom_outer_midspan_E11": select_midspan_outer_face_metric(
            profiles,
            "FACE_BOTTOM",
            "E11",
            manifest["geometry"],
        ),
        "left_shear_core_avg_S13_MPa": select_left_shear_core_average(
            profiles,
            "S13_MPa",
        ),
        "left_shear_core_avg_gamma13": select_left_shear_core_average(
            profiles,
            "gamma13_engineering",
        ),
        "post_status": summary["overall_status"],
    }


def add_reference_errors(records, reference):
    metric_names = [
        "midspan_U3_mm",
        "SENER_Nmm",
        "top_outer_midspan_S11_MPa",
        "bottom_outer_midspan_S11_MPa",
        "top_outer_midspan_E11",
        "bottom_outer_midspan_E11",
        "left_shear_core_avg_S13_MPa",
        "left_shear_core_avg_gamma13",
    ]

    for record in records:
        record["reference_ne_cell"] = reference["ne_cell"]
        record["reference_case_hash"] = reference["case_hash"]

        for metric in metric_names:
            value = record[metric]
            ref = reference[metric]

            if math.isnan(value) or math.isnan(ref):
                error = math.nan
            else:
                error = relative_difference(value, ref)

            record[metric + "_rel_to_reference"] = error


def apparent_order(records, metric):
    """
    Estimate p from three successive h, h/2, h/4 results if possible.

    The result is diagnostic only; non-monotone values return None.
    """
    by_ne = {record["ne_cell"]: record for record in records}

    nes = sorted(by_ne)
    if len(nes) < 3:
        return None

    # Use the three finest levels if there are more than three.
    n1, n2, n3 = nes[-3:]

    if not (n2 == 2 * n1 and n3 == 2 * n2):
        return None

    f1 = by_ne[n1][metric]
    f2 = by_ne[n2][metric]
    f3 = by_ne[n3][metric]

    if any(math.isnan(value) for value in (f1, f2, f3)):
        return None

    d12 = f1 - f2
    d23 = f2 - f3

    if abs(d23) <= 1.0e-30:
        return None

    ratio = d12 / d23

    if ratio <= 0.0:
        return None

    p = math.log(ratio, 2.0)

    if not math.isfinite(p):
        return None

    return {
        "coarse_ne": n1,
        "medium_ne": n2,
        "fine_ne": n3,
        "p": p,
    }


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


def plot_metric(records, metric, ylabel, output_path):
    records = sorted(records, key=lambda item: item["ne_cell"])
    x = [record["ne_cell"] for record in records]
    y = [record[metric] for record in records]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Core elements per nominal cell edge")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def plot_relative_errors(records, output_path):
    records = sorted(records, key=lambda item: item["ne_cell"])
    x = [record["ne_cell"] for record in records]

    metrics = [
        ("midspan_U3_mm_rel_to_reference", "|w_mid|"),
        ("SENER_Nmm_rel_to_reference", "U"),
        ("top_outer_midspan_S11_MPa_rel_to_reference", "top S11"),
        ("left_shear_core_avg_gamma13_rel_to_reference", "core gamma13"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for metric, label in metrics:
        y = [record[metric] for record in records]
        ax.plot(x, y, marker="o", label=label)

    ax.set_xlabel("Core elements per nominal cell edge")
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

    element_types = sorted({record["element_type"] for record in records})
    if len(element_types) != 1:
        raise ValueError(
            "Mesh-convergence comparison requires one element formulation; got {}."
            .format(element_types)
        )

    ne_values = [record["ne_cell"] for record in records]
    if len(set(ne_values)) != len(ne_values):
        raise ValueError("Duplicate core elements-per-cell-edge levels detected.")

    records = sorted(records, key=lambda item: item["ne_cell"])

    if args.reference_ne is None:
        reference = max(records, key=lambda item: item["ne_cell"])
    else:
        matches = [
            record
            for record in records
            if record["ne_cell"] == args.reference_ne
        ]
        if len(matches) != 1:
            raise ValueError(
                "Reference ne={} not found exactly once.".format(args.reference_ne)
            )
        reference = matches[0]

    add_reference_errors(records, reference)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        parents = [Path(record["case_path"]).parent for record in records]
        common_parent = Path(
            __import__("os").path.commonpath([str(path) for path in parents])
        )
        output_dir = common_parent / "mesh_convergence"

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "midspan_U3_mm",
        "SENER_Nmm",
        "top_outer_midspan_S11_MPa",
        "bottom_outer_midspan_S11_MPa",
        "top_outer_midspan_E11",
        "bottom_outer_midspan_E11",
        "left_shear_core_avg_S13_MPa",
        "left_shear_core_avg_gamma13",
    ]

    orders = {
        metric: apparent_order(records, metric)
        for metric in metric_names
    }

    summary = {
        "schema": SCHEMA,
        "element_type": element_types[0],
        "reference": {
            "ne_cell": reference["ne_cell"],
            "case_hash": reference["case_hash"],
            "case_path": reference["case_path"],
        },
        "cases": records,
        "apparent_orders": orders,
    }

    summary_path = output_dir / "mesh_convergence.json"
    csv_path = output_dir / "mesh_convergence.csv"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    write_csv(csv_path, records)

    figures = [
        plot_metric(
            records,
            "midspan_U3_mm",
            "Midspan U3 [mm]",
            output_dir / "midspan_deflection_convergence.png",
        ),
        plot_metric(
            records,
            "SENER_Nmm",
            "Elastic strain energy [N mm]",
            output_dir / "strain_energy_convergence.png",
        ),
        plot_relative_errors(
            records,
            output_dir / "relative_error_convergence.png",
        ),
    ]

    print("=" * 80)
    print("HOST 4PB MESH CONVERGENCE")
    print("=" * 80)
    print("Element type : {}".format(element_types[0]))
    print(
        "Reference    : ne={} ({})".format(
            reference["ne_cell"],
            reference["case_hash"],
        )
    )
    print()

    for record in records:
        print(
            "ne={:<2d}  elems={:<7d}  U3_mid={: .10g} mm  "
            "dU3={:.3e}  SENER={:.10g} Nmm  dU={:.3e}".format(
                record["ne_cell"],
                record["actual_element_count"],
                record["midspan_U3_mm"],
                record["midspan_U3_mm_rel_to_reference"],
                record["SENER_Nmm"],
                record["SENER_Nmm_rel_to_reference"],
            )
        )

    print()
    print("Apparent order estimates (diagnostic):")
    for metric in (
        "midspan_U3_mm",
        "SENER_Nmm",
        "top_outer_midspan_S11_MPa",
        "left_shear_core_avg_gamma13",
    ):
        result = orders[metric]
        if result is None:
            print("  {:36s}: n/a".format(metric))
        else:
            print("  {:36s}: p={:.4f}".format(metric, result["p"]))

    print()
    print("Wrote:")
    for path in (
        summary_path,
        csv_path,
        output_dir / "midspan_deflection_convergence.png",
        output_dir / "strain_energy_convergence.png",
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
