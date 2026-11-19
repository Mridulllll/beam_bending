#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract component-wise stress-strain work and 3-D field measures from one
postprocessed HOST four-point-bending Abaqus case.

Run with Abaqus Python, for example

    abaqus python fem/abaqus/postprocessing/extract_4pb_component_work.py ^
        --case "C:/AbaqusScratch/HOST/4pb_homogenized/<case>"

No FE solve is performed.  The existing ODB is read only.

Energy convention
-----------------
The production 4PB postprocessor already verifies

    U = 0.5 * integral sigma : epsilon dV

using Abaqus E with engineering shear components.  The same convention is
used here.  Therefore the six signed component work terms are

    U11 = 0.5 integral S11 E11 dV
    U22 = 0.5 integral S22 E22 dV
    U33 = 0.5 integral S33 E33 dV
    U12 = 0.5 integral S12 E12 dV
    U13 = 0.5 integral S13 E13 dV
    U23 = 0.5 integral S23 E23 dV

and their sum closes to 0.5*integral(S:E)dV.

These terms are diagnostic signed work partitions, NOT independent positive
modal energies for an anisotropic material.  In particular, constitutive
couplings are already embedded in each stress component.

The script also reports volume-weighted RMS/mean-absolute and maximum-absolute
stress/strain measures.  To distinguish local load/support boundary layers
from the beam interior, it additionally separates axial fixture bands from
bulk material.  By default each fixture band extends one unit-cell size to
either side of every support/load section contained in the computational
domain.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
from collections import defaultdict

from odbAccess import openOdb


COMPONENTS = (
    ("11", 0, "S11", "E11"),
    ("22", 1, "S22", "E22"),
    ("33", 2, "S33", "E33"),
    ("12", 3, "S12", "E12"),
    ("13", 4, "S13", "E13"),
    ("23", 5, "S23", "E23"),
)

REGION_ORDER = (
    "ALL",
    "CORE",
    "FACE_BOTTOM",
    "FACE_TOP",
    "FIXTURE_BANDS",
    "BULK",
    "CORE_FIXTURE_BANDS",
    "CORE_BULK",
)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def scalar_data(value):
    data = value.data

    if isinstance(data, (tuple, list)):
        if len(data) != 1:
            raise ValueError(
                "Expected scalar field data, got {} components.".format(
                    len(data)
                )
            )
        return float(data[0])

    try:
        # Abaqus scalar field data are usually Python floats.
        return float(data)
    except TypeError:
        return float(data[0])


def vector_data(value):
    return tuple(float(v) for v in value.data)


def value_key(value):
    return (
        str(value.instance.name),
        int(value.elementLabel),
        int(getattr(value, "integrationPoint", 0) or 0),
    )


def geometry_value(geometry, *names):
    for name in names:
        if name in geometry:
            return float(geometry[name])
    raise KeyError(
        "None of the geometry keys {} are present.".format(names)
    )


def optional_geometry_value(geometry, default, *names):
    for name in names:
        if name in geometry:
            return float(geometry[name])
    return float(default)


def reconstruction_factor(manifest):
    domain = manifest.get("domain", {})

    for key in (
        "energy_reconstruction_factor",
        "full_model_energy_reconstruction_factor",
        "reconstruction_factor",
    ):
        if key in domain:
            return float(domain[key])

    mode = str(domain.get("mode", "full")).lower()
    if mode == "quarter_xy":
        return 4.0

    return 1.0


def select_step_frame(odb, requested_step):
    if requested_step:
        if requested_step not in odb.steps:
            raise KeyError(
                "Step {!r} not found. Available: {}".format(
                    requested_step,
                    list(odb.steps.keys()),
                )
            )
        step_name = requested_step
    elif "BENDING" in odb.steps:
        step_name = "BENDING"
    else:
        step_names = list(odb.steps.keys())
        if not step_names:
            raise RuntimeError("ODB contains no steps.")
        step_name = step_names[-1]

    step = odb.steps[step_name]
    if not step.frames:
        raise RuntimeError("Step {!r} contains no frames.".format(step_name))

    return step_name, step.frames[-1]


def find_odb(case_dir, manifest):
    job = manifest.get("job", {})
    candidates = []

    for key in ("odb_path", "odb"):
        value = job.get(key)
        if value:
            candidates.append(value)

    job_name = job.get("job_name") or manifest.get("case", {}).get("job_name")
    if job_name:
        candidates.append(os.path.join(case_dir, str(job_name) + ".odb"))

    for name in os.listdir(case_dir):
        if name.lower().endswith(".odb"):
            candidates.append(os.path.join(case_dir, name))

    seen = set()
    for candidate in candidates:
        path = os.path.abspath(str(candidate))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    raise IOError("Could not locate an ODB in {!r}.".format(case_dir))


def build_element_centroids(odb, stress_values):
    """
    Return {(instance_name, element_label): (x,y,z)} for elements represented
    in the stress field.

    The mean of the element nodal coordinates is sufficient here because the
    centroids are used only for material-layer and axial-zone classification.
    """
    needed = defaultdict(set)

    for value in stress_values:
        needed[str(value.instance.name)].add(int(value.elementLabel))

    centroids = {}

    for instance_name, labels in needed.items():
        instance = odb.rootAssembly.instances[instance_name]

        node_coordinates = {}
        for node in instance.nodes:
            node_coordinates[int(node.label)] = tuple(
                float(v) for v in node.coordinates
            )

        for element in instance.elements:
            label = int(element.label)
            if label not in labels:
                continue

            coords = [
                node_coordinates[int(node_label)]
                for node_label in element.connectivity
            ]

            count = float(len(coords))
            centroids[(instance_name, label)] = (
                sum(p[0] for p in coords) / count,
                sum(p[1] for p in coords) / count,
                sum(p[2] for p in coords) / count,
            )

    return centroids


def new_accumulator():
    return {
        "volume": 0.0,
        "sener": 0.0,
        "component_work": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "stress_abs_volume": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "stress_sq_volume": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "stress_max_abs": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "strain_abs_volume": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "strain_sq_volume": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "strain_max_abs": dict((name, 0.0) for name, _, _, _ in COMPONENTS),
        "record_count": 0,
    }


def add_record(acc, stress, strain, ivol, sener_density):
    acc["volume"] += ivol
    acc["record_count"] += 1

    if sener_density is not None:
        acc["sener"] += float(sener_density) * ivol

    for name, index, _, _ in COMPONENTS:
        s = float(stress[index])
        e = float(strain[index])

        acc["component_work"][name] += 0.5 * s * e * ivol

        acc["stress_abs_volume"][name] += abs(s) * ivol
        acc["stress_sq_volume"][name] += s * s * ivol
        acc["stress_max_abs"][name] = max(
            acc["stress_max_abs"][name],
            abs(s),
        )

        acc["strain_abs_volume"][name] += abs(e) * ivol
        acc["strain_sq_volume"][name] += e * e * ivol
        acc["strain_max_abs"][name] = max(
            acc["strain_max_abs"][name],
            abs(e),
        )


def finalize_region(acc, factor):
    volume = float(acc["volume"])
    total_work = sum(acc["component_work"].values())

    result = {
        "record_count": int(acc["record_count"]),
        "computational_volume_mm3": volume,
        "full_model_reconstructed_volume_mm3": volume * factor,
        "computational_SENER_Nmm": float(acc["sener"]),
        "full_model_reconstructed_SENER_Nmm": float(acc["sener"]) * factor,
        "computational_component_work_sum_Nmm": total_work,
        "full_model_reconstructed_component_work_sum_Nmm": total_work * factor,
        "components": {},
    }

    for name, _, stress_label, strain_label in COMPONENTS:
        work = float(acc["component_work"][name])

        if volume > 0.0:
            stress_mean_abs = acc["stress_abs_volume"][name] / volume
            stress_rms = math.sqrt(acc["stress_sq_volume"][name] / volume)
            strain_mean_abs = acc["strain_abs_volume"][name] / volume
            strain_rms = math.sqrt(acc["strain_sq_volume"][name] / volume)
        else:
            stress_mean_abs = float("nan")
            stress_rms = float("nan")
            strain_mean_abs = float("nan")
            strain_rms = float("nan")

        if abs(total_work) > 0.0:
            signed_fraction = work / total_work
        else:
            signed_fraction = float("nan")

        result["components"][name] = {
            "stress_label": stress_label,
            "strain_label": strain_label,
            "computational_work_Nmm": work,
            "full_model_reconstructed_work_Nmm": work * factor,
            "signed_fraction_of_component_work_sum": signed_fraction,
            "stress_mean_abs_MPa": stress_mean_abs,
            "stress_rms_MPa": stress_rms,
            "stress_max_abs_MPa": float(acc["stress_max_abs"][name]),
            "strain_mean_abs": strain_mean_abs,
            "strain_rms": strain_rms,
            "strain_max_abs": float(acc["strain_max_abs"][name]),
        }

    if abs(total_work) > 0.0:
        result["absolute_component_work_fraction_denominator_Nmm"] = sum(
            abs(float(acc["component_work"][name]))
            for name, _, _, _ in COMPONENTS
        )

        denominator = result[
            "absolute_component_work_fraction_denominator_Nmm"
        ]

        for name, _, _, _ in COMPONENTS:
            if denominator > 0.0:
                fraction = abs(float(acc["component_work"][name])) / denominator
            else:
                fraction = float("nan")

            result["components"][name][
                "absolute_fraction_of_absolute_component_work"
            ] = fraction

    return result


def classify_material_region(z, z_core_bottom, z_core_top, tolerance):
    if z < z_core_bottom - tolerance:
        return "FACE_BOTTOM"
    if z > z_core_top + tolerance:
        return "FACE_TOP"
    return "CORE"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        required=True,
        help="4PB case directory containing manifest.json and the ODB.",
    )

    parser.add_argument(
        "--step",
        default=None,
        help="ODB step name. Defaults to BENDING or the last available step.",
    )

    parser.add_argument(
        "--fixture-band-half-width",
        type=float,
        default=None,
        help=(
            "Half-width [mm] of axial support/load boundary-layer bands. "
            "Default: one unit-cell size."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    case_dir = os.path.abspath(args.case)
    manifest_path = os.path.join(case_dir, "manifest.json")
    summary_path = os.path.join(case_dir, "post", "summary.json")

    if not os.path.isfile(manifest_path):
        raise IOError("manifest.json not found in {!r}".format(case_dir))

    manifest = load_json(manifest_path)
    geometry = manifest["geometry"]

    odb_path = find_odb(case_dir, manifest)

    cell_size = geometry_value(
        geometry,
        "cell_size",
        "cell_size_mm",
    )

    core_height = geometry_value(
        geometry,
        "core_height",
        "core_height_mm",
    )

    face_bottom = geometry_value(
        geometry,
        "face_bottom",
        "face_bottom_mm",
    )

    face_top = geometry_value(
        geometry,
        "face_top",
        "face_top_mm",
    )

    total_height = face_bottom + core_height + face_top

    z_core_bottom = optional_geometry_value(
        geometry,
        -0.5 * total_height + face_bottom,
        "z_core_bottom",
        "z_core_bottom_mm",
    )

    z_core_top = optional_geometry_value(
        geometry,
        +0.5 * total_height - face_top,
        "z_core_top",
        "z_core_top_mm",
    )

    fixture_positions = []

    for names in (
        ("support_left", "support_left_mm"),
        ("support_right", "support_right_mm"),
        ("load_left", "load_left_mm"),
        ("load_right", "load_right_mm"),
    ):
        try:
            fixture_positions.append(geometry_value(geometry, *names))
        except KeyError:
            pass

    band_half_width = (
        float(args.fixture_band_half_width)
        if args.fixture_band_half_width is not None
        else cell_size
    )

    factor = reconstruction_factor(manifest)

    print("=" * 88)
    print("HOST / ABAQUS 4PB COMPONENT-WORK EXTRACTOR")
    print("=" * 88)
    print("Case                 :", case_dir)
    print("ODB                  :", odb_path)
    print("Energy reconstruction:", factor)
    print("Fixture band halfwidth: {:.6g} mm".format(band_half_width))

    odb = openOdb(path=odb_path, readOnly=True)

    try:
        step_name, frame = select_step_frame(odb, args.step)

        required = ("S", "E", "IVOL")
        for name in required:
            if name not in frame.fieldOutputs:
                raise KeyError(
                    "Required field {!r} missing in final frame.".format(name)
                )

        stress_values = frame.fieldOutputs["S"].values
        strain_values = frame.fieldOutputs["E"].values
        ivol_values = frame.fieldOutputs["IVOL"].values

        sener_values = (
            frame.fieldOutputs["SENER"].values
            if "SENER" in frame.fieldOutputs
            else []
        )

        strain_map = dict(
            (value_key(v), vector_data(v))
            for v in strain_values
        )

        ivol_map = dict(
            (value_key(v), scalar_data(v))
            for v in ivol_values
        )

        sener_map = dict(
            (value_key(v), scalar_data(v))
            for v in sener_values
        )

        print("Step / frame          : {} / {}".format(step_name, len(odb.steps[step_name].frames) - 1))
        print("Stress IP records     :", len(stress_values))
        print("Building element centroids...")

        centroids = build_element_centroids(odb, stress_values)

        if not centroids:
            raise RuntimeError("Could not build any element centroids.")

        x_values = [coord[0] for coord in centroids.values()]
        x_min = min(x_values)
        x_max = max(x_values)

        active_fixture_positions = [
            x
            for x in fixture_positions
            if x_min - band_half_width <= x <= x_max + band_half_width
        ]

        print(
            "Computational x range : {:.6g} .. {:.6g} mm".format(
                x_min,
                x_max,
            )
        )
        print(
            "Active fixture x      : {}".format(
                ", ".join(
                    "{:.6g}".format(x)
                    for x in active_fixture_positions
                )
            )
        )

        accumulators = dict(
            (name, new_accumulator())
            for name in REGION_ORDER
        )

        missing = 0
        layer_tolerance = max(1.0e-8, 1.0e-9 * total_height)

        for stress_value in stress_values:
            key = value_key(stress_value)

            if key not in strain_map or key not in ivol_map:
                missing += 1
                continue

            stress = vector_data(stress_value)
            strain = strain_map[key]

            if len(stress) != 6 or len(strain) != 6:
                raise ValueError(
                    "Expected 6-component solid S/E data at key {!r}.".format(key)
                )

            ivol = float(ivol_map[key])
            if ivol <= 0.0:
                continue

            element_key = (key[0], key[1])
            if element_key not in centroids:
                missing += 1
                continue

            x, _, z = centroids[element_key]

            material_region = classify_material_region(
                z,
                z_core_bottom,
                z_core_top,
                layer_tolerance,
            )

            near_fixture = any(
                abs(x - x_fixture) <= band_half_width
                for x_fixture in active_fixture_positions
            )

            axial_region = "FIXTURE_BANDS" if near_fixture else "BULK"

            regions = [
                "ALL",
                material_region,
                axial_region,
            ]

            if material_region == "CORE":
                regions.append(
                    "CORE_FIXTURE_BANDS"
                    if near_fixture
                    else "CORE_BULK"
                )

            sener_density = sener_map.get(key)

            for region in regions:
                add_record(
                    accumulators[region],
                    stress,
                    strain,
                    ivol,
                    sener_density,
                )

        finalized = dict(
            (name, finalize_region(accumulators[name], factor))
            for name in REGION_ORDER
        )

        all_region = finalized["ALL"]
        component_sum = float(
            all_region["computational_component_work_sum_Nmm"]
        )
        sener = float(all_region["computational_SENER_Nmm"])

        if sener != 0.0:
            closure_rel = abs(component_sum - sener) / abs(sener)
        else:
            closure_rel = float("nan")

        summary_midspan = None
        if os.path.isfile(summary_path):
            try:
                post_summary = load_json(summary_path)
                summary_midspan = abs(
                    float(
                        post_summary["midspan"]["U3_midplane_avg_mm"]
                    )
                )
            except Exception:
                summary_midspan = None

        n_shear = int(geometry.get("n_shear", -1))
        shear_span = geometry_value(
            geometry,
            "shear_span",
            "shear_span_mm",
        )
        support_span = geometry_value(
            geometry,
            "support_span",
            "support_span_mm",
        )

        report = {
            "schema": "host_4pb_component_work_v1",
            "case": {
                "path": case_dir,
                "case_hash": manifest.get("case", {}).get("case_hash"),
                "odb": odb_path,
                "step": step_name,
                "frame_index": len(odb.steps[step_name].frames) - 1,
            },
            "geometry": {
                "n_shear": n_shear,
                "cell_size_mm": cell_size,
                "core_height_mm": core_height,
                "face_bottom_mm": face_bottom,
                "face_top_mm": face_top,
                "total_height_mm": total_height,
                "shear_span_mm": shear_span,
                "support_span_mm": support_span,
                "a_over_h": shear_span / total_height,
                "S_over_h": support_span / total_height,
                "computational_x_min_mm": x_min,
                "computational_x_max_mm": x_max,
            },
            "domain": {
                "mode": manifest.get("domain", {}).get("mode", "full"),
                "energy_reconstruction_factor": factor,
            },
            "diagnostic_definition": {
                "component_work": "0.5 * integral S_ij * E_ij dV",
                "shear_strain_convention": "Abaqus E engineering shear; no factor 2",
                "interpretation": (
                    "signed stress-strain work terms; not independent positive "
                    "modal energies for anisotropic material"
                ),
                "fixture_band_half_width_mm": band_half_width,
                "active_fixture_positions_mm": active_fixture_positions,
            },
            "midspan": {
                "abs_U3_midplane_avg_mm": summary_midspan,
            },
            "qa": {
                "stress_record_count": len(stress_values),
                "missing_or_unmatched_record_count": missing,
                "component_work_vs_SENER_relative_error": closure_rel,
            },
            "regions": finalized,
        }

        post_dir = os.path.join(case_dir, "post")
        if not os.path.isdir(post_dir):
            os.makedirs(post_dir)

        json_path = os.path.join(post_dir, "component_work.json")
        csv_path = os.path.join(post_dir, "component_work.csv")

        write_json(json_path, report)

        fields = (
            "region",
            "component",
            "stress_label",
            "strain_label",
            "computational_work_Nmm",
            "full_model_reconstructed_work_Nmm",
            "signed_fraction_of_component_work_sum",
            "absolute_fraction_of_absolute_component_work",
            "stress_mean_abs_MPa",
            "stress_rms_MPa",
            "stress_max_abs_MPa",
            "strain_mean_abs",
            "strain_rms",
            "strain_max_abs",
            "computational_volume_mm3",
            "full_model_reconstructed_volume_mm3",
        )

        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()

            for region_name in REGION_ORDER:
                region = finalized[region_name]

                for component_name, _, _, _ in COMPONENTS:
                    component = region["components"][component_name]

                    row = {
                        "region": region_name,
                        "component": component_name,
                        "computational_volume_mm3": region[
                            "computational_volume_mm3"
                        ],
                        "full_model_reconstructed_volume_mm3": region[
                            "full_model_reconstructed_volume_mm3"
                        ],
                    }

                    for field in fields:
                        if field in component:
                            row[field] = component[field]

                    writer.writerow(row)

        print()
        print("Closure 0.5*S:E vs SENER : {:.3e}".format(closure_rel))

        def print_region(name):
            region = finalized[name]
            total = region["computational_component_work_sum_Nmm"]
            c33 = region["components"]["33"]
            c13 = region["components"]["13"]

            print(
                "{:<18s} U={:>12.6g} Nmm  "
                "U33={:>9.4%}  U13={:>9.4%}  "
                "E33_rms={:>10.4e}  S33_rms={:>10.4e} MPa".format(
                    name,
                    total,
                    c33["signed_fraction_of_component_work_sum"],
                    c13["signed_fraction_of_component_work_sum"],
                    c33["strain_rms"],
                    c33["stress_rms_MPa"],
                )
            )

        print_region("ALL")
        print_region("CORE")
        print_region("CORE_BULK")
        print_region("CORE_FIXTURE_BANDS")

        print()
        print("Wrote:")
        print(" ", json_path)
        print(" ", csv_path)
        print("=" * 88)

    finally:
        odb.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
