#!/usr/bin/env python3
"""
Run the existing HOST four-point-bending benchmark directly from a committed
homogenized Abaqus FE reference package.

This adapter deliberately reuses examples/run_tpms_beam_bending_from_json.py
rather than duplicating the verified HOST beam implementation. It converts the
exact structural-frame stiffness tensor stored in source_case_manifest.json to
the TPMS JSON schema expected by HOST, forwards the FE geometry/load definition,
captures the benchmark output, and writes a compact first HOST-vs-FE comparison.

Usage
-----

python examples/run_host_4pb_from_fe_reference.py \
    reference_results/4pb/Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1

Outputs are written into:

    <reference>/host_comparison/

including:

    host_core_input.json
    host_run_definition.json
    host_benchmark_stdout.txt
    host_vs_fe_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


HOST_VOIGT_ORDER = [
    "xx",
    "yy",
    "zz",
    "xy",
    "xz",
    "yz",
]

SCHEMA = "host_4pb_from_fe_reference_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the verified HOST 4PB beam benchmark from a committed "
            "homogenized FE reference package."
        )
    )
    parser.add_argument(
        "reference_dir",
        help=(
            "Reference package directory containing "
            "reference_manifest.json and source_case_manifest.json."
        ),
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y"),
        default="x",
        help=(
            "HOST beam axis. The current FE benchmark uses structural X, "
            "therefore the default is x."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <reference_dir>/host_comparison.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create HOST input/run-definition files but do not launch the solver.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def nested_first(mapping, paths):
    for path in paths:
        current = mapping
        ok = True

        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]

        if ok and current is not None:
            return current

    return None


def extract_face_elastic_constants(source_manifest):
    """
    Recover the isotropic face constants from common manifest locations.

    The current FE benchmark uses E=70000 MPa, nu=0.33. We still try to read
    them from the manifest first so future cases do not silently inherit these
    defaults.
    """
    constitutive = source_manifest.get("constitutive", {})

    E = nested_first(
        constitutive,
        (
            ("face", "E_MPa"),
            ("face", "E"),
            ("face", "youngs_modulus_MPa"),
            ("face_material", "E_MPa"),
            ("face_material", "E"),
            ("face_material", "youngs_modulus_MPa"),
            ("face_E_MPa",),
            ("face_E",),
        ),
    )

    nu = nested_first(
        constitutive,
        (
            ("face", "nu"),
            ("face", "poisson_ratio"),
            ("face_material", "nu"),
            ("face_material", "poisson_ratio"),
            ("face_nu",),
        ),
    )

    source = "manifest"

    if E is None:
        E = 70000.0
        source = "fallback_current_4pb_builder_default"

    if nu is None:
        nu = 0.33
        source = "fallback_current_4pb_builder_default"

    return float(E), float(nu), source


def require_matrix_6x6(value, label):
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("{} must contain 6 rows.".format(label))

    matrix = []

    for row in value:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("{} must be 6x6.".format(label))
        matrix.append([float(item) for item in row])

    return matrix


def extract_float(pattern, text, label, required=True):
    match = re.search(
        pattern,
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        if required:
            raise RuntimeError(
                "Could not parse {!r} from HOST benchmark output.".format(label)
            )
        return None

    return float(match.group(1))


def repo_root_from_script():
    return Path(__file__).resolve().parents[1]


def main():
    args = parse_args()

    reference_dir = Path(args.reference_dir).resolve()
    reference_manifest_path = reference_dir / "reference_manifest.json"
    source_manifest_path = reference_dir / "source_case_manifest.json"
    fe_summary_path = reference_dir / "summary.json"

    for path in (
        reference_manifest_path,
        source_manifest_path,
        fe_summary_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    reference_manifest = load_json(reference_manifest_path)
    source_manifest = load_json(source_manifest_path)
    fe_summary = load_json(fe_summary_path)

    geometry = source_manifest["geometry"]
    fixture = source_manifest["fixture"]
    query = source_manifest["database_query"]
    constitutive = source_manifest["constitutive"]

    if "C_hom_model_MPa" not in constitutive:
        raise KeyError(
            "source_case_manifest.json does not contain "
            "constitutive.C_hom_model_MPa."
        )

    C_model = require_matrix_6x6(
        constitutive["C_hom_model_MPa"],
        "constitutive.C_hom_model_MPa",
    )

    face_E, face_nu, face_property_source = extract_face_elastic_constants(
        source_manifest
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else reference_dir / "host_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    orientation = source_manifest.get("orientation", {})

    host_core_input = {
        "topology": str(query["topology"]),
        "relative_density_target": float(
            query["relative_density_target"]
        ),
        "relative_density_actual": float(
            query["relative_density_actual"]
        ),
        "units": "MPa",
        "orientation": (
            "Structural global X/Y/Z; tensor already rotated from RVE frame. "
            "FE orientation preset: {}."
            .format(
                orientation.get("preset", "unspecified")
            )
        ),
        "shear_convention": "engineering",
        "voigt_order": HOST_VOIGT_ORDER,
        "C_hom": C_model,
        "source": {
            "reference_name": reference_manifest["reference_name"],
            "fe_case_hash": reference_manifest["source"]["case_hash"],
            "database_eval_id": query.get("eval_id"),
            "campaign_name": query.get("campaign_name"),
            "optimization_load_case": query.get(
                "optimization_load_case"
            ),
            "requested_state": query.get("requested_state"),
            "iteration": query.get("iteration"),
            "tensor_field": "constitutive.C_hom_model_MPa",
            "important": (
                "No additional tensor rotation is to be applied by HOST for "
                "the structural-X beam comparison."
            ),
        },
    }

    host_core_path = output_dir / "host_core_input.json"
    write_json(
        host_core_path,
        host_core_input,
    )

    total_load = float(
        fixture["total_applied_load_N"]
    )

    run_definition = {
        "schema": SCHEMA,
        "reference_name": reference_manifest["reference_name"],
        "fe_case_hash": reference_manifest["source"]["case_hash"],
        "axis": args.axis,
        "geometry": {
            "length_mm": float(geometry["length"]),
            "width_mm": float(geometry["width"]),
            "core_height_mm": float(geometry["core_height"]),
            "face_bottom_mm": float(geometry["face_bottom"]),
            "face_top_mm": float(geometry["face_top"]),
            "support_span_mm": float(geometry["support_span"]),
            "load_span_mm": float(geometry["load_span"]),
        },
        "load": {
            "total_transverse_load_N": total_load,
            "host_sign_convention": (
                "Existing HOST benchmark reports positive downward "
                "deflection magnitude."
            ),
            "fe_sign_convention": "FE U3 is negative for downward deflection.",
        },
        "face_material": {
            "E_MPa": face_E,
            "nu": face_nu,
            "source": face_property_source,
        },
        "core_input": str(host_core_path),
        "fe_targets": {
            "midspan_U3_mm_signed": float(
                fe_summary["midspan"]["U3_midplane_avg_mm"]
            ),
            "midspan_deflection_magnitude_mm": abs(
                float(fe_summary["midspan"]["U3_midplane_avg_mm"])
            ),
            "strain_energy_Nmm": float(
                fe_summary["energy"]["SENER_integral_Nmm"]
            ),
        },
    }

    run_definition_path = output_dir / "host_run_definition.json"
    write_json(
        run_definition_path,
        run_definition,
    )

    print("=" * 88)
    print("HOST 4PB FROM HOMOGENIZED FE REFERENCE")
    print("=" * 88)
    print("Reference       : {}".format(reference_manifest["reference_name"]))
    print("FE case hash    : {}".format(reference_manifest["source"]["case_hash"]))
    print("Beam axis       : {}".format(args.axis))
    print(
        "Geometry        : L={:.6g}, b={:.6g}, hc={:.6g}, hf={:.6g}/{:.6g} mm"
        .format(
            float(geometry["length"]),
            float(geometry["width"]),
            float(geometry["core_height"]),
            float(geometry["face_bottom"]),
            float(geometry["face_top"]),
        )
    )
    print(
        "Fixture         : support span={:.6g}, load span={:.6g}, P={:.6g} N"
        .format(
            float(geometry["support_span"]),
            float(geometry["load_span"]),
            total_load,
        )
    )
    print(
        "Face material   : E={:.6g} MPa, nu={:.6g} ({})"
        .format(
            face_E,
            face_nu,
            face_property_source,
        )
    )
    print("Core tensor     : constitutive.C_hom_model_MPa")
    print(
        "FE |w_mid|      : {:.12g} mm".format(
            run_definition["fe_targets"][
                "midspan_deflection_magnitude_mm"
            ]
        )
    )
    print()

    if args.prepare_only:
        print("Prepared only; HOST solver was not launched.")
        print("Wrote:")
        print("  {}".format(host_core_path))
        print("  {}".format(run_definition_path))
        return

    repo_root = repo_root_from_script()
    benchmark_script = (
        repo_root
        / "examples"
        / "run_tpms_beam_bending_from_json.py"
    )

    if not benchmark_script.exists():
        raise FileNotFoundError(benchmark_script)

    command = [
        sys.executable,
        str(benchmark_script),
        str(host_core_path),
        "--axis",
        str(args.axis),
        "--test",
        "four-point",
        "--length",
        str(float(geometry["length"])),
        "--width",
        str(float(geometry["width"])),
        "--core-height",
        str(float(geometry["core_height"])),
        "--face-bottom",
        str(float(geometry["face_bottom"])),
        "--face-top",
        str(float(geometry["face_top"])),
        "--support-span",
        str(float(geometry["support_span"])),
        "--load-span",
        str(float(geometry["load_span"])),
        "--load",
        str(total_load),
        "--face-E",
        str(face_E),
        "--face-nu",
        str(face_nu),
    ]

    print("Launching existing verified HOST benchmark:")
    print("  {}".format(" ".join(command)))
    print()

    result = subprocess.run(
        command,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    stdout_path = output_dir / "host_benchmark_stdout.txt"
    stdout_path.write_text(
        stdout,
        encoding="utf-8",
    )

    if stderr:
        stderr_path = output_dir / "host_benchmark_stderr.txt"
        stderr_path.write_text(
            stderr,
            encoding="utf-8",
        )
    else:
        stderr_path = None

    # Echo captured output so the command behaves like the underlying benchmark.
    if stdout:
        print(stdout, end="")

    if stderr:
        print(stderr, file=sys.stderr, end="")

    if result.returncode != 0:
        raise RuntimeError(
            "HOST benchmark failed with return code {}. See {}{}."
            .format(
                result.returncode,
                stdout_path,
                (
                    " and {}".format(stderr_path)
                    if stderr_path is not None
                    else ""
                ),
            )
        )

    # Existing benchmark output is intentionally parsed here instead of
    # duplicating its verified beam assembly/solver implementation.
    w_host = extract_float(
        r"^\s*HOST-MI w \[mm\]\s*:\s*([+\-0-9.eE]+)\s*$",
        stdout,
        "HOST-MI w [mm]",
    )

    w_reddy = extract_float(
        r"^\s*Reddy w \[mm\]\s*:\s*([+\-0-9.eE]+)\s*$",
        stdout,
        "Reddy w [mm]",
        required=False,
    )

    w_eb = extract_float(
        r"^\s*EB w \[mm\]\s*:\s*([+\-0-9.eE]+)\s*$",
        stdout,
        "EB w [mm]",
        required=False,
    )

    host_midspan_M = extract_float(
        r"^\s*HOST midspan M \[Nmm\]\s*:\s*([+\-0-9.eE]+)\s*$",
        stdout,
        "HOST midspan M",
        required=False,
    )

    host_midspan_V = extract_float(
        r"^\s*HOST midspan V \[N\]\s*:\s*([+\-0-9.eE]+)\s*$",
        stdout,
        "HOST midspan V",
        required=False,
    )

    fe_w_signed = float(
        fe_summary["midspan"]["U3_midplane_avg_mm"]
    )
    fe_w = abs(fe_w_signed)

    comparison = {
        "schema": SCHEMA,
        "reference_name": reference_manifest["reference_name"],
        "fe_case_hash": reference_manifest["source"]["case_hash"],
        "axis": args.axis,
        "tensor_source": "constitutive.C_hom_model_MPa",
        "tensor_rotation_applied_by_adapter": False,
        "fe": {
            "midspan_U3_mm_signed": fe_w_signed,
            "midspan_deflection_magnitude_mm": fe_w,
            "strain_energy_Nmm": float(
                fe_summary["energy"]["SENER_integral_Nmm"]
            ),
        },
        "host": {
            "HOST_MI_midspan_deflection_magnitude_mm": w_host,
            "Reddy_midspan_deflection_magnitude_mm": w_reddy,
            "EB_midspan_deflection_magnitude_mm": w_eb,
            "midspan_M_Nmm": host_midspan_M,
            "midspan_V_N": host_midspan_V,
        },
        "comparison": {
            "HOST_MI_minus_FE_mm": w_host - fe_w,
            "HOST_MI_relative_error_signed": (
                (w_host - fe_w) / fe_w
            ),
            "HOST_MI_relative_error_abs": abs(
                w_host - fe_w
            ) / fe_w,
            "HOST_MI_over_FE": w_host / fe_w,
        },
        "files": {
            "host_core_input": str(host_core_path),
            "host_run_definition": str(run_definition_path),
            "host_benchmark_stdout": str(stdout_path),
        },
    }

    comparison_path = output_dir / "host_vs_fe_summary.json"
    write_json(
        comparison_path,
        comparison,
    )

    print()
    print("-" * 88)
    print("FIRST HOST <-> HOMOGENIZED 3D FE COMPARISON")
    print("-" * 88)
    print(
        "FE |w_mid|      : {:.12g} mm".format(
            fe_w
        )
    )
    print(
        "HOST-MI |w_mid| : {:.12g} mm".format(
            w_host
        )
    )
    print(
        "HOST / FE       : {:.10f}".format(
            w_host / fe_w
        )
    )
    print(
        "relative error  : {:+.6f} %".format(
            100.0
            * (
                (w_host - fe_w)
                / fe_w
            )
        )
    )
    print()
    print("Wrote:")
    print("  {}".format(host_core_path))
    print("  {}".format(run_definition_path))
    print("  {}".format(stdout_path))
    if stderr_path is not None:
        print("  {}".format(stderr_path))
    print("  {}".format(comparison_path))


if __name__ == "__main__":
    main()
