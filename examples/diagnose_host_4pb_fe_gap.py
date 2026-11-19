#!/usr/bin/env python3
"""
Diagnose the gap between the current 1-D HOST 4PB solution and the committed
homogenized 3-D Abaqus FE reference.

The diagnostic intentionally does not modify either model. It checks whether
the comparison is contaminated by fixture/reference-surface effects and
quantifies transverse-normal/3-D indicators already present in the exported FE
data.

Usage
-----

python examples/diagnose_host_4pb_fe_gap.py \
    reference_results/4pb/Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1

Outputs
-------

<reference>/host_comparison/host_fe_gap_diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCHEMA = "host_4pb_fe_gap_diagnostics_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose HOST-vs-3D-FE 4PB response differences."
    )
    parser.add_argument(
        "reference_dir",
        help="Committed FE reference package directory.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key, default=math.nan):
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def nearest_x_row(rows, x_target):
    return min(
        rows,
        key=lambda row: abs(as_float(row, "x_mm") - float(x_target)),
    )


def station_displacement(row):
    top = as_float(row, "top_U3_mm")
    mid = as_float(row, "midplane_U3_mm")
    bottom = as_float(row, "bottom_U3_mm")

    return {
        "x_mm": as_float(row, "x_mm"),
        "top_U3_mm": top,
        "midplane_U3_mm": mid,
        "bottom_U3_mm": bottom,
        "top_minus_bottom_mm": top - bottom,
        "mid_minus_face_average_mm": mid - 0.5 * (top + bottom),
    }


def relative_error(value, reference):
    return (value - reference) / max(abs(reference), 1.0e-30)


def main():
    args = parse_args()
    reference_dir = Path(args.reference_dir).resolve()

    reference_manifest = load_json(
        reference_dir / "reference_manifest.json"
    )
    source_manifest = load_json(
        reference_dir / "source_case_manifest.json"
    )
    fe_summary = load_json(
        reference_dir / "summary.json"
    )
    global_rows = load_csv(
        reference_dir / "global_response.csv"
    )
    energy_rows = load_csv(
        reference_dir / "energy_summary.csv"
    )
    profile_rows = load_csv(
        reference_dir / "through_thickness_profiles.csv"
    )

    host_path = (
        reference_dir
        / "host_comparison"
        / "host_vs_fe_summary.json"
    )
    if not host_path.exists():
        raise FileNotFoundError(
            "{} is missing. Run run_host_4pb_from_fe_reference.py first."
            .format(host_path)
        )

    host_summary = load_json(host_path)

    geometry = source_manifest["geometry"]

    stations = {
        "support_left": float(geometry["support_left"]),
        "load_left": float(geometry["load_left"]),
        "midspan": float(geometry["midspan"]),
        "load_right": float(geometry["load_right"]),
        "support_right": float(geometry["support_right"]),
    }

    displacement = {}
    for name, x_value in stations.items():
        displacement[name] = station_displacement(
            nearest_x_row(global_rows, x_value)
        )

    support_mid_left = displacement["support_left"]["midplane_U3_mm"]
    support_mid_right = displacement["support_right"]["midplane_U3_mm"]
    support_mid_mean = 0.5 * (support_mid_left + support_mid_right)

    support_top_mean = 0.5 * (
        displacement["support_left"]["top_U3_mm"]
        + displacement["support_right"]["top_U3_mm"]
    )
    support_bottom_mean = 0.5 * (
        displacement["support_left"]["bottom_U3_mm"]
        + displacement["support_right"]["bottom_U3_mm"]
    )

    mid = displacement["midspan"]

    # A 1-D simply supported beam constrains the reference-surface displacement
    # w(sA)=w(sB)=0. The physical FE fixture instead constrains the bottom
    # surface at the supports. Subtracting the FE midplane support offset does
    # not "correct" the FE result; it reports the part directly comparable to a
    # support-zeroed reference-surface displacement.
    fe_mid_raw_signed = mid["midplane_U3_mm"]
    fe_mid_support_zeroed_signed = (
        fe_mid_raw_signed - support_mid_mean
    )

    fe_top_support_zeroed_signed = (
        mid["top_U3_mm"] - support_top_mean
    )
    fe_bottom_support_zeroed_signed = (
        mid["bottom_U3_mm"] - support_bottom_mean
    )

    host_w = float(
        host_summary["host"][
            "HOST_MI_midspan_deflection_magnitude_mm"
        ]
    )

    energy_by_region = {
        row["region"]: row
        for row in energy_rows
    }

    energy_diagnostics = {}
    for region_name, row in energy_by_region.items():
        total = as_float(row, "SENER_integral_Nmm")
        component_names = (
            "Ux_work_Nmm",
            "Uy_work_Nmm",
            "Uz_work_Nmm",
            "Uxy_work_Nmm",
            "Uxz_work_Nmm",
            "Uyz_work_Nmm",
        )

        components = {
            name: as_float(row, name)
            for name in component_names
        }

        energy_diagnostics[region_name] = {
            "SENER_Nmm": total,
            "half_sigma_colon_epsilon_Nmm": as_float(
                row,
                "half_sigma_colon_epsilon_Nmm",
            ),
            "signed_component_work_Nmm": components,
            "signed_component_fraction_of_SENER": {
                name: value / max(abs(total), 1.0e-30)
                for name, value in components.items()
            },
            "note": (
                "These are signed 0.5*sigma_ij*epsilon_ij work "
                "contributions. For anisotropic coupled materials they are "
                "diagnostics, not unique positive modal energies."
            ),
        }

    profile_diagnostics = {}
    for station in (
        "support_left",
        "left_shear_mid",
        "load_left",
        "midspan",
        "load_right",
        "right_shear_mid",
        "support_right",
    ):
        rows = [
            row
            for row in profile_rows
            if row.get("station") == station
        ]
        if not rows:
            continue

        def max_abs(field):
            values = [
                as_float(row, field)
                for row in rows
                if math.isfinite(as_float(row, field))
            ]
            return max((abs(value) for value in values), default=math.nan)

        profile_diagnostics[station] = {
            "max_abs_S33_MPa": max_abs("S33_MPa"),
            "max_abs_E33": max_abs("E33"),
            "max_abs_S13_MPa": max_abs("S13_MPa"),
            "max_abs_gamma13": max_abs("gamma13_engineering"),
        }

    fe_raw_mag = abs(fe_mid_raw_signed)
    fe_support_zeroed_mag = abs(fe_mid_support_zeroed_signed)

    output = {
        "schema": SCHEMA,
        "reference_name": reference_manifest["reference_name"],
        "fe_case_hash": reference_manifest["source"]["case_hash"],
        "geometry": {
            "support_span_over_height": (
                float(geometry["support_span"])
                / float(geometry["total_height"])
            ),
            "shear_span_over_height": (
                float(geometry["shear_span"])
                / float(geometry["total_height"])
            ),
            "width_over_height": (
                float(geometry["width"])
                / float(geometry["total_height"])
            ),
        },
        "host": {
            "HOST_MI_midspan_deflection_magnitude_mm": host_w,
        },
        "fe_displacement_stations": displacement,
        "reference_surface_fixture_diagnostic": {
            "mean_support_midplane_U3_mm": support_mid_mean,
            "mean_support_top_U3_mm": support_top_mean,
            "mean_support_bottom_U3_mm": support_bottom_mean,
            "FE_raw_midplane_midspan_magnitude_mm": fe_raw_mag,
            "FE_support_zeroed_midplane_midspan_magnitude_mm": (
                fe_support_zeroed_mag
            ),
            "FE_support_zeroed_top_midspan_magnitude_mm": abs(
                fe_top_support_zeroed_signed
            ),
            "FE_support_zeroed_bottom_midspan_magnitude_mm": abs(
                fe_bottom_support_zeroed_signed
            ),
            "HOST_vs_FE_raw_relative_error": relative_error(
                host_w,
                fe_raw_mag,
            ),
            "HOST_vs_FE_support_zeroed_midplane_relative_error": (
                relative_error(
                    host_w,
                    fe_support_zeroed_mag,
                )
            ),
            "interpretation": (
                "The FE supports constrain the bottom outer surface while the "
                "current 1-D HOST beam imposes w=0 on its reference surface. "
                "A nonzero FE midplane displacement at the supports is therefore "
                "a fixture/reference-surface mismatch and local transverse "
                "deformation indicator."
            ),
        },
        "energy_diagnostics": energy_diagnostics,
        "through_thickness_diagnostics": profile_diagnostics,
        "current_host_kinematic_limit": {
            "w_through_thickness": "w(s,z)=w0(s)",
            "epsilon_zz": 0.0,
            "transverse_normal_field_eta_active": False,
            "note": (
                "The current five-field beam cannot represent thickness stretch "
                "or local top/mid/bottom U3 separation."
            ),
        },
    }

    output_dir = reference_dir / "host_comparison"
    output_path = output_dir / "host_fe_gap_diagnostics.json"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            output,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("=" * 88)
    print("HOST <-> 3D FE 4PB GAP DIAGNOSTICS")
    print("=" * 88)
    print(
        "S/h = {:.6g}, a/h = {:.6g}, b/h = {:.6g}".format(
            output["geometry"]["support_span_over_height"],
            output["geometry"]["shear_span_over_height"],
            output["geometry"]["width_over_height"],
        )
    )
    print()
    print("Width-averaged FE U3 at fixture/reference stations [mm]:")
    print(
        "{:<15s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
            "station",
            "top",
            "midplane",
            "bottom",
            "top-bottom",
        )
    )

    for name in (
        "support_left",
        "load_left",
        "midspan",
        "load_right",
        "support_right",
    ):
        item = displacement[name]
        print(
            "{:<15s} {:>14.8g} {:>14.8g} {:>14.8g} {:>14.8g}".format(
                name,
                item["top_U3_mm"],
                item["midplane_U3_mm"],
                item["bottom_U3_mm"],
                item["top_minus_bottom_mm"],
            )
        )

    print()
    print(
        "FE raw |w_mid,midplane|       : {:.12g} mm".format(
            fe_raw_mag
        )
    )
    print(
        "FE support-zeroed |w_mid|     : {:.12g} mm".format(
            fe_support_zeroed_mag
        )
    )
    print(
        "HOST-MI |w_mid|               : {:.12g} mm".format(
            host_w
        )
    )
    print(
        "HOST error vs raw FE          : {:+.6f} %".format(
            100.0
            * output[
                "reference_surface_fixture_diagnostic"
            ]["HOST_vs_FE_raw_relative_error"]
        )
    )
    print(
        "HOST error vs support-zeroed  : {:+.6f} %".format(
            100.0
            * output[
                "reference_surface_fixture_diagnostic"
            ][
                "HOST_vs_FE_support_zeroed_midplane_relative_error"
            ]
        )
    )

    print()
    print("Signed FE component-work diagnostics:")
    for region in ("CORE", "FACE_TOP", "FACE_BOTTOM", "ALL"):
        item = energy_diagnostics.get(region)
        if item is None:
            continue

        fractions = item["signed_component_fraction_of_SENER"]
        print(
            "  {:<11s} U={:>12.6g} Nmm  "
            "Ux={:+.3%}  Uz={:+.3%}  Uxz={:+.3%}  "
            "Uy={:+.3%}  Uyz={:+.3%}".format(
                region,
                item["SENER_Nmm"],
                fractions["Ux_work_Nmm"],
                fractions["Uz_work_Nmm"],
                fractions["Uxz_work_Nmm"],
                fractions["Uy_work_Nmm"],
                fractions["Uyz_work_Nmm"],
            )
        )

    print()
    print("Through-thickness S33/E33 indicators:")
    for station, item in profile_diagnostics.items():
        print(
            "  {:<17s} max|S33|={:>10.6g} MPa  max|E33|={:>10.4e}"
            .format(
                station,
                item["max_abs_S33_MPa"],
                item["max_abs_E33"],
            )
        )

    print()
    print("Wrote:")
    print("  {}".format(output_path))


if __name__ == "__main__":
    main()
