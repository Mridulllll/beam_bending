# -*- coding: utf-8 -*-
"""
Validate and visualize standardized HOST 4PB homogenized Abaqus postprocessing.

Run with the normal project Python environment, not Abaqus Python:

    python fem/abaqus/postprocessing/validate_4pb_post.py ^
        --case "C:/AbaqusScratch/HOST/4pb_homogenized/<case-directory>"

Inputs:
    <case>/manifest.json
    <case>/post/summary.json
    <case>/post/global_response.csv
    <case>/post/section_resultants.csv
    <case>/post/face_response.csv
    <case>/post/through_thickness_profiles.csv
    <case>/post/energy_summary.csv

Outputs:
    <case>/post/validation/
        validation_metrics.json
        deflection.png
        section_resultants.png
        face_axial_response.png
        through_thickness_profiles.png

The script compares FE section resultants against the exact statics of the
idealized four-point-bending fixture. The target diagrams therefore validate
the FE reduction/postprocessing independently of any sandwich theory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCHEMA = "host_4pb_homogenized_validation_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and plot HOST homogenized 4PB postprocessed results."
        )
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Solved 4PB case directory containing manifest.json and post/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <case>/post/validation.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
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
    if value is None or value == "":
        return default
    return float(value)


def rel_error(value, reference):
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def rms(values):
    values = list(values)
    if not values:
        return math.nan
    return math.sqrt(sum(v * v for v in values) / len(values))


def ideal_4pb_section_resultants(x, geometry, total_load):
    """
    Exact statics for the current idealized symmetric four-point bending case.

    Sign convention follows postprocess_4pb_homogenized.py:
      Qz_raw_from_S13_N:
          negative in the left shear span, positive in the right shear span.
      My_beam_convention_Nmm:
          positive sagging.
    """
    support_left = float(geometry["support_left"])
    support_right = float(geometry["support_right"])
    load_left = float(geometry["load_left"])
    load_right = float(geometry["load_right"])

    reaction = 0.5 * float(total_load)

    if x < support_left:
        qz = 0.0
        moment = 0.0
    elif x < load_left:
        qz = -reaction
        moment = reaction * (x - support_left)
    elif x <= load_right:
        qz = 0.0
        moment = reaction * (load_left - support_left)
    elif x <= support_right:
        qz = reaction
        moment = reaction * (support_right - x)
    else:
        qz = 0.0
        moment = 0.0

    return {
        "Nx_N": 0.0,
        "Qz_N": qz,
        "My_Nmm": moment,
    }


def section_validation(rows, manifest):
    geometry = manifest["geometry"]
    total_load = float(manifest["fixture"]["total_applied_load_N"])

    domain = manifest.get("domain", {})
    resultant_factor = float(
        domain.get(
            "section_resultant_reconstruction_factor",
            1.0,
        )
    )

    records = []

    for row in rows:
        x = as_float(row, "x_center_mm")
        fe_nx_raw = as_float(row, "Nx_raw_N")
        fe_qz_raw = as_float(row, "Qz_raw_from_S13_N")
        fe_my_raw = as_float(row, "My_beam_convention_Nmm")

        fe_nx = resultant_factor * fe_nx_raw
        fe_qz = resultant_factor * fe_qz_raw
        fe_my = resultant_factor * fe_my_raw

        target = ideal_4pb_section_resultants(
            x,
            geometry,
            total_load,
        )

        records.append(
            {
                "x_mm": x,
                "resultant_reconstruction_factor": resultant_factor,
                "fe_Nx_raw_domain_N": fe_nx_raw,
                "fe_Qz_raw_domain_N": fe_qz_raw,
                "fe_My_raw_domain_Nmm": fe_my_raw,
                "fe_Nx_N": fe_nx,
                "fe_Qz_N": fe_qz,
                "fe_My_Nmm": fe_my,
                "target_Nx_N": target["Nx_N"],
                "target_Qz_N": target["Qz_N"],
                "target_My_Nmm": target["My_Nmm"],
            }
        )

    load_left = float(geometry["load_left"])
    load_right = float(geometry["load_right"])
    support_left = float(geometry["support_left"])
    support_right = float(geometry["support_right"])
    dx = as_float(rows[0], "dx_mm")

    # Exclude one slab adjacent to each ideal line load/support when evaluating
    # plateau errors. This avoids grading the expected local line-load boundary
    # layer as if it were a postprocessing error.
    shear_records = [
        item
        for item in records
        if (
            support_left + dx <= item["x_mm"] <= load_left - dx
            or load_right + dx <= item["x_mm"] <= support_right - dx
        )
    ]

    moment_plateau_records = [
        item
        for item in records
        if load_left + dx <= item["x_mm"] <= load_right - dx
    ]

    q_errors = [
        item["fe_Qz_N"] - item["target_Qz_N"]
        for item in shear_records
    ]
    m_errors = [
        item["fe_My_Nmm"] - item["target_My_Nmm"]
        for item in moment_plateau_records
    ]

    max_abs_nx = max(abs(item["fe_Nx_N"]) for item in records)
    max_abs_q_error = max(abs(value) for value in q_errors)
    max_abs_m_error = max(abs(value) for value in m_errors)

    reaction = 0.5 * total_load
    target_mmax = reaction * (load_left - support_left)

    metrics = {
        "target": {
            "reaction_each_N": reaction,
            "shear_plateau_abs_N": reaction,
            "constant_moment_Nmm": target_mmax,
        },
        "axial_resultant": {
            "max_abs_Nx_N": max_abs_nx,
            "normalized_by_total_load": max_abs_nx / total_load,
        },
        "shear_resultant": {
            "plateau_sample_count": len(shear_records),
            "max_abs_error_N": max_abs_q_error,
            "rms_error_N": rms(q_errors),
            "max_relative_error": max_abs_q_error / reaction,
        },
        "bending_moment": {
            "plateau_sample_count": len(moment_plateau_records),
            "max_abs_error_Nmm": max_abs_m_error,
            "rms_error_Nmm": rms(m_errors),
            "max_relative_error": max_abs_m_error / target_mmax,
        },
    }

    return records, metrics


def symmetry_validation(global_rows, manifest):
    domain_mode = str(
        manifest.get("domain", {}).get("mode", "full")
    ).strip().lower()

    if domain_mode == "quarter_xy":
        return {
            "status": "IMPOSED",
            "sample_count": 0,
            "max_abs_U3_mirror_difference_mm": math.nan,
            "rms_U3_mirror_difference_mm": math.nan,
            "normalized_max_by_midspan_deflection": math.nan,
            "note": (
                "Mirror symmetry is imposed by the quarter_xy boundary "
                "conditions and cannot be independently measured in the "
                "reduced domain."
            ),
        }

    length = float(manifest["geometry"]["length"])

    by_x = {
        round(as_float(row, "x_mm"), 9): row
        for row in global_rows
    }

    residuals = []

    for x_key, row in by_x.items():
        mirror_key = round(length - x_key, 9)
        mirror = by_x.get(mirror_key)

        if mirror is None:
            continue

        for field in (
            "top_U3_mm",
            "midplane_U3_mm",
            "bottom_U3_mm",
        ):
            a = as_float(row, field)
            b = as_float(mirror, field)

            if math.isnan(a) or math.isnan(b):
                continue

            residuals.append(a - b)

    max_abs = max(abs(value) for value in residuals) if residuals else math.nan
    rms_value = rms(residuals)

    midspan = abs(
        float(
            load_json(
                Path(manifest["_case_path"])
                / "post"
                / "summary.json"
            )["midspan"]["U3_midplane_avg_mm"]
        )
    )

    return {
        "status": "MEASURED",
        "sample_count": len(residuals),
        "max_abs_U3_mirror_difference_mm": max_abs,
        "rms_U3_mirror_difference_mm": rms_value,
        "normalized_max_by_midspan_deflection": (
            max_abs / max(midspan, 1.0e-30)
            if not math.isnan(max_abs)
            else math.nan
        ),
    }


def plot_deflection(global_rows, manifest, output_path):
    x = [as_float(row, "x_mm") for row in global_rows]
    top = [as_float(row, "top_U3_mm") for row in global_rows]
    mid = [as_float(row, "midplane_U3_mm") for row in global_rows]
    bottom = [as_float(row, "bottom_U3_mm") for row in global_rows]

    geometry = manifest["geometry"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, top, label="Top outer surface")
    ax.plot(x, mid, label="Core midplane")
    ax.plot(x, bottom, label="Bottom outer surface")

    for xpos in (
        float(geometry["support_left"]),
        float(geometry["load_left"]),
        float(geometry["load_right"]),
        float(geometry["support_right"]),
    ):
        ax.axvline(xpos, linewidth=0.8, linestyle="--")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("U3 [mm]")
    ax.set_title("4PB transverse displacement")
    ax.grid(True, linewidth=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def plot_section_resultants(records, output_path):
    x = [item["x_mm"] for item in records]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)

    axes[0].plot(x, [item["fe_Nx_N"] for item in records], label="FE")
    axes[0].plot(x, [item["target_Nx_N"] for item in records], linestyle="--", label="Statics")
    axes[0].set_ylabel("Nx [N]")
    axes[0].grid(True, linewidth=0.4)
    axes[0].legend()

    axes[1].plot(x, [item["fe_Qz_N"] for item in records], label="FE")
    axes[1].plot(x, [item["target_Qz_N"] for item in records], linestyle="--", label="Statics")
    axes[1].set_ylabel("Qz [N]")
    axes[1].grid(True, linewidth=0.4)
    axes[1].legend()

    axes[2].plot(x, [item["fe_My_Nmm"] for item in records], label="FE")
    axes[2].plot(x, [item["target_My_Nmm"] for item in records], linestyle="--", label="Statics")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("My [N mm]")
    axes[2].grid(True, linewidth=0.4)
    axes[2].legend()

    fig.suptitle("4PB section-resultant validation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def plot_face_axial_response(face_rows, output_path):
    grouped = defaultdict(list)

    for row in face_rows:
        key = (
            row["face"],
            as_float(row, "z_element_centroid_mm"),
        )
        grouped[key].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)

    for (face, z_value), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: as_float(item, "x_center_mm"))
        x = [as_float(item, "x_center_mm") for item in rows]
        s11 = [as_float(item, "S11_MPa") for item in rows]
        e11 = [as_float(item, "E11") for item in rows]
        label = "{} z={:.3f} mm".format(face, z_value)

        axes[0].plot(x, s11, label=label)
        axes[1].plot(x, e11, label=label)

    axes[0].set_ylabel("S11 [MPa]")
    axes[0].grid(True, linewidth=0.4)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("E11 [-]")
    axes[1].grid(True, linewidth=0.4)

    fig.suptitle("Face-sheet axial response")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def plot_through_thickness_profiles(profile_rows, output_path):
    preferred = [
        "left_shear_mid",
        "midspan",
        "right_shear_mid",
    ]

    available = []
    for name in preferred:
        if any(row["station"] == name for row in profile_rows):
            available.append(name)

    if not available:
        available = sorted({row["station"] for row in profile_rows})[:3]

    fig, axes = plt.subplots(
        len(available),
        3,
        figsize=(10.5, 3.4 * len(available)),
        squeeze=False,
    )

    for row_index, station in enumerate(available):
        rows = [
            row
            for row in profile_rows
            if row["station"] == station
        ]
        z_field = (
            "z_ip_mm"
            if rows and "z_ip_mm" in rows[0]
            else "z_element_centroid_mm"
        )

        rows = sorted(
            rows,
            key=lambda item: as_float(item, z_field),
        )

        z = [as_float(item, z_field) for item in rows]

        quantities = [
            ("S11_MPa", "S11 [MPa]"),
            ("S13_MPa", "S13 [MPa]"),
            ("S33_MPa", "S33 [MPa]"),
        ]

        for col_index, (field, label) in enumerate(quantities):
            ax = axes[row_index][col_index]
            ax.plot([as_float(item, field) for item in rows], z)
            ax.set_xlabel(label)
            ax.set_ylabel("z [mm]")
            ax.grid(True, linewidth=0.4)

            if col_index == 1:
                ax.set_title(station)

    fig.suptitle("Width-averaged through-thickness stress profiles")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    return fig


def main():
    args = parse_args()

    case_path = Path(args.case).resolve()
    post_path = case_path / "post"

    manifest_path = case_path / "manifest.json"
    summary_path = post_path / "summary.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    manifest = load_json(manifest_path)
    manifest["_case_path"] = str(case_path)
    summary = load_json(summary_path)

    global_rows = load_csv(post_path / "global_response.csv")
    section_rows = load_csv(post_path / "section_resultants.csv")
    face_rows = load_csv(post_path / "face_response.csv")
    profile_rows = load_csv(post_path / "through_thickness_profiles.csv")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else post_path / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    section_records, section_metrics = section_validation(
        section_rows,
        manifest,
    )
    symmetry_metrics = symmetry_validation(
        global_rows,
        manifest,
    )

    metrics = {
        "schema": SCHEMA,
        "case_id": manifest["case"]["case_id"],
        "case_hash": manifest["case"]["case_hash"],
        "postprocessor_status": summary["overall_status"],
        "midspan_U3_mm": summary["midspan"]["U3_midplane_avg_mm"],
        "reaction_equilibrium_relative_error": (
            summary["equilibrium"]["vertical_relative_error"]
        ),
        "section_resultants": section_metrics,
        "displacement_mirror_symmetry": symmetry_metrics,
    }

    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figures = [
        plot_deflection(
            global_rows,
            manifest,
            output_dir / "deflection.png",
        ),
        plot_section_resultants(
            section_records,
            output_dir / "section_resultants.png",
        ),
        plot_face_axial_response(
            face_rows,
            output_dir / "face_axial_response.png",
        ),
        plot_through_thickness_profiles(
            profile_rows,
            output_dir / "through_thickness_profiles.png",
        ),
    ]

    print("=" * 78)
    print("HOST 4PB POSTPROCESSING VALIDATION")
    print("=" * 78)
    print("Case      : {}".format(case_path))
    print("Post QA   : {}".format(summary["overall_status"]))
    print("Midspan U3: {:.10g} mm".format(float(metrics["midspan_U3_mm"])))
    print()
    print("Statics comparison:")
    print(
        "  max |Nx| / P        = {:.3e}".format(
            section_metrics["axial_resultant"]["normalized_by_total_load"]
        )
    )
    print(
        "  max Q plateau error = {:.6g} N ({:.3e} relative)".format(
            section_metrics["shear_resultant"]["max_abs_error_N"],
            section_metrics["shear_resultant"]["max_relative_error"],
        )
    )
    print(
        "  max M plateau error = {:.6g} N mm ({:.3e} relative)".format(
            section_metrics["bending_moment"]["max_abs_error_Nmm"],
            section_metrics["bending_moment"]["max_relative_error"],
        )
    )
    if symmetry_metrics.get("status") == "IMPOSED":
        print(
            "  U3 mirror symmetry = imposed by quarter_xy BCs"
        )
    else:
        print(
            "  max U3 mirror diff  = {:.6g} mm ({:.3e} of |w_mid|)".format(
                symmetry_metrics["max_abs_U3_mirror_difference_mm"],
                symmetry_metrics["normalized_max_by_midspan_deflection"],
            )
        )
    print()
    print("Wrote:")
    for path in (
        metrics_path,
        output_dir / "deflection.png",
        output_dir / "section_resultants.png",
        output_dir / "face_axial_response.png",
        output_dir / "through_thickness_profiles.png",
    ):
        print("  {}".format(path))

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
