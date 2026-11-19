# -*- coding: utf-8 -*-
"""
Export a compact, self-contained HOST 4PB FE reference package.

Run with the normal project Python environment:

    python fem/abaqus/postprocessing/export_4pb_fe_reference.py ^
        --case "C:/AbaqusScratch/HOST/4pb_homogenized/<case-directory>"

The large Abaqus ODB remains in scratch. Only the compact manifest and
postprocessed CSV/JSON results required for theory comparison are copied into:

    reference_results/4pb/<reference-name>/

This makes the FE benchmark reproducible and versionable without committing
multi-hundred-megabyte ODB files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REFERENCE_SCHEMA = "host_4pb_fe_reference_v1"

REQUIRED_POST_FILES = (
    "summary.json",
    "equilibrium.csv",
    "global_response.csv",
    "face_response.csv",
    "section_resultants.csv",
    "through_thickness_profiles.csv",
    "energy_summary.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a compact HOST 4PB FE reference package."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Solved/postprocessed 4PB case directory.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional reference directory name.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Defaults to <repo>/reference_results/4pb. "
            "Use this to export elsewhere."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing reference directory.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_root_from_script():
    return Path(__file__).resolve().parents[3]


def make_default_name(manifest):
    mesh = manifest["mesh"]
    query = manifest["database_query"]

    topology = str(query["topology"])
    rd = float(query["relative_density_target"])
    state = str(query["requested_state"])

    return (
        "{}_rd{:02d}_{}_{}_ne{}_fz{}_{}"
        .format(
            topology,
            int(round(rd * 100.0)),
            state,
            mesh["element_type"],
            mesh["core_elements_per_cell_edge"],
            mesh["face_elements_through_thickness"],
            manifest["case"]["case_hash"],
        )
    )


def main():
    args = parse_args()

    case_dir = Path(args.case).resolve()
    manifest_path = case_dir / "manifest.json"
    post_dir = case_dir / "post"
    post_summary_path = post_dir / "summary.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not post_summary_path.exists():
        raise FileNotFoundError(
            "{} is missing. Postprocess the case first.".format(
                post_summary_path
            )
        )

    manifest = load_json(manifest_path)
    post_summary = load_json(post_summary_path)

    if post_summary.get("overall_status") != "PASS":
        raise ValueError(
            "Refusing to export a reference whose postprocessing QA is {!r}."
            .format(post_summary.get("overall_status"))
        )

    for filename in REQUIRED_POST_FILES:
        path = post_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else repo_root_from_script() / "reference_results" / "4pb"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    name = args.name or make_default_name(manifest)
    destination = output_root / name

    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(
                "{} already exists. Use --overwrite to replace it."
                .format(destination)
            )
        shutil.rmtree(destination)

    destination.mkdir(parents=True)

    shutil.copy2(
        manifest_path,
        destination / "source_case_manifest.json",
    )

    copied = {}
    for filename in REQUIRED_POST_FILES:
        source = post_dir / filename
        target = destination / filename
        shutil.copy2(source, target)
        copied[filename] = {
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        }

    reference_manifest = {
        "schema": REFERENCE_SCHEMA,
        "reference_name": name,
        "source": {
            "case_directory": str(case_dir),
            "case_id": manifest["case"]["case_id"],
            "case_hash": manifest["case"]["case_hash"],
            "builder_schema": manifest["schema"],
            "postprocessor_schema": post_summary["schema"],
            "job_name": manifest["job_name"],
        },
        "model": {
            "topology": manifest["database_query"]["topology"],
            "relative_density_target": manifest["database_query"][
                "relative_density_target"
            ],
            "relative_density_actual": manifest["database_query"][
                "relative_density_actual"
            ],
            "state": manifest["database_query"]["requested_state"],
            "load_case": manifest["database_query"][
                "optimization_load_case"
            ],
            "element_type": manifest["mesh"]["element_type"],
            "core_elements_per_cell_edge": manifest["mesh"][
                "core_elements_per_cell_edge"
            ],
            "face_elements_through_thickness": manifest["mesh"][
                "face_elements_through_thickness"
            ],
            "actual_node_count": manifest["mesh"]["derived_counts"][
                "actual_node_count"
            ],
            "actual_element_count": manifest["mesh"]["derived_counts"][
                "actual_element_count"
            ],
        },
        "geometry": manifest["geometry"],
        "fixture": manifest["fixture"],
        "orientation": manifest["orientation"],
        "qa": post_summary["qa"],
        "headline_results": {
            "midspan_U3_mm": post_summary["midspan"][
                "U3_midplane_avg_mm"
            ],
            "SENER_Nmm": post_summary["energy"]["SENER_integral_Nmm"],
            "reaction_relative_error": post_summary["equilibrium"][
                "vertical_relative_error"
            ],
        },
        "files": copied,
        "note": (
            "The Abaqus ODB is intentionally not copied. This package contains "
            "the compact standardized FE quantities required for HOST/theory "
            "comparison; source_case_manifest.json preserves the full model "
            "definition and scratch provenance."
        ),
    }

    reference_manifest_path = destination / "reference_manifest.json"
    with open(reference_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            reference_manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("=" * 84)
    print("HOST 4PB FE REFERENCE EXPORT")
    print("=" * 84)
    print("Source      : {}".format(case_dir))
    print("Destination : {}".format(destination))
    print("Case hash   : {}".format(manifest["case"]["case_hash"]))
    print(
        "Mesh        : {} ne={} face_z={}"
        .format(
            manifest["mesh"]["element_type"],
            manifest["mesh"]["core_elements_per_cell_edge"],
            manifest["mesh"]["face_elements_through_thickness"],
        )
    )
    print(
        "Midspan U3  : {:.10g} mm".format(
            float(
                post_summary["midspan"]["U3_midplane_avg_mm"]
            )
        )
    )
    print(
        "SENER       : {:.10g} N mm".format(
            float(post_summary["energy"]["SENER_integral_Nmm"])
        )
    )
    print()
    print("Wrote:")
    print("  {}".format(reference_manifest_path))
    for filename in REQUIRED_POST_FILES:
        print("  {}".format(destination / filename))


if __name__ == "__main__":
    main()
