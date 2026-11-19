#!/usr/bin/env python3
"""
Compare full-domain and quarter_xy 4PB FE results for selected TPMS topologies.

The script uses the actual HOST manifest/postprocessing schemas:
  manifest.json
  post/summary.json
  post/through_thickness_profiles.csv

It compares:
  - midspan U3;
  - reconstructed full-model strain energy;
  - left_shear_mid S13 and engineering gamma13 through-thickness profiles.

No HOST theory is evaluated here.  This is purely a FE symmetry-reduction audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["Primitive", "Diamond"],
    )

    parser.add_argument(
        "--relative-density",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--n-shear",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--campaign-type",
        default="sizing_shape",
    )

    parser.add_argument(
        "--load-case",
        default="compression",
    )

    parser.add_argument(
        "--state",
        default="initial",
    )

    parser.add_argument(
        "--element-type",
        default="C3D20R",
    )

    parser.add_argument(
        "--ne-cell",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--face-ne-z",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--fixture-mode",
        default="section_distributing",
    )

    parser.add_argument(
        "--model-implementation-revision",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "verification/transfer/results/"
            "topology_rd12_initial/"
            "quarter_full_symmetry_audit.json"
        ),
    )

    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def close(a, b, tol=1.0e-9):
    return abs(float(a) - float(b)) <= float(tol)


def matches_common(manifest, args, topology):
    q = manifest.get("database_query", {})
    g = manifest.get("geometry", {})
    m = manifest.get("mesh", {})
    f = manifest.get("fixture", {})

    try:
        return (
            int(manifest.get("model_implementation_revision"))
            == int(args.model_implementation_revision)
            and str(q.get("topology", "")).lower()
            == str(topology).lower()
            and close(
                q.get("relative_density_target"),
                args.relative_density,
            )
            and str(q.get("campaign_type", "")).lower()
            == str(args.campaign_type).lower()
            and str(q.get("optimization_load_case", "")).lower()
            == str(args.load_case).lower()
            and str(q.get("requested_state", "")).lower()
            == str(args.state).lower()
            and int(g.get("n_shear")) == int(args.n_shear)
            and str(m.get("element_type", "")).upper()
            == str(args.element_type).upper()
            and int(m.get("core_elements_per_cell_edge"))
            == int(args.ne_cell)
            and int(m.get("face_elements_through_thickness"))
            == int(args.face_ne_z)
            and str(f.get("mode", "")).lower()
            == str(args.fixture_mode).lower()
        )
    except (TypeError, ValueError):
        return False


def find_case(run_root, args, topology, domain_mode):
    candidates = []

    for case_dir in run_root.iterdir():
        if not case_dir.is_dir():
            continue

        manifest_path = case_dir / "manifest.json"
        summary_path = case_dir / "post" / "summary.json"

        if not manifest_path.is_file() or not summary_path.is_file():
            continue

        try:
            manifest = load_json(manifest_path)
            summary = load_json(summary_path)
        except Exception:
            continue

        if not matches_common(manifest, args, topology):
            continue

        if str(
            manifest.get("domain", {}).get("mode", "")
        ).lower() != str(domain_mode).lower():
            continue

        if str(summary.get("overall_status", "")).upper() != "PASS":
            continue

        candidates.append(
            (
                case_dir.stat().st_mtime,
                case_dir,
                manifest,
                summary,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No QA-PASS {} case found for {}.".format(
                domain_mode,
                topology,
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, case_dir, manifest, summary = candidates[0]

    return case_dir, manifest, summary


def reconstructed_energy(summary):
    energy = summary["energy"]

    if "full_model_reconstructed_SENER_Nmm" in energy:
        return float(
            energy["full_model_reconstructed_SENER_Nmm"]
        )

    return float(energy["SENER_integral_Nmm"])


def read_profile(case_dir, station="left_shear_mid"):
    path = (
        case_dir
        / "post"
        / "through_thickness_profiles.csv"
    )

    if not path.is_file():
        return {}

    result = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if row.get("station") != station:
                continue

            try:
                key = (
                    str(row["region"]),
                    round(float(row["z_ip_mm"]), 8),
                )

                result[key] = {
                    "z": float(row["z_ip_mm"]),
                    "weight": float(row["volume_mm3"]),
                    "S13": float(row["S13_MPa"]),
                    "gamma13": float(
                        row["gamma13_engineering"]
                    ),
                }
            except (KeyError, TypeError, ValueError):
                continue

    return result


def relative_rmse(full_profile, quarter_profile, field):
    keys = sorted(
        set(full_profile).intersection(quarter_profile),
        key=lambda key: (
            full_profile[key]["z"],
            key[0],
        ),
    )

    if not keys:
        return float("nan"), 0

    full = np.asarray(
        [full_profile[key][field] for key in keys],
        dtype=float,
    )

    quarter = np.asarray(
        [quarter_profile[key][field] for key in keys],
        dtype=float,
    )

    weights = np.asarray(
        [full_profile[key]["weight"] for key in keys],
        dtype=float,
    )

    numerator = math.sqrt(
        float(
            np.sum(
                weights
                * (quarter - full) ** 2
            )
        )
    )

    denominator = math.sqrt(
        float(
            np.sum(
                weights
                * full ** 2
            )
        )
    )

    return (
        100.0 * numerator / denominator
        if denominator > 0.0
        else float("nan"),
        len(keys),
    )


def percent_difference(quarter, full):
    return 100.0 * (float(quarter) - float(full)) / abs(float(full))


def main():
    args = parse_args()
    run_root = args.run_root.resolve()

    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    rows = []

    for topology in args.topologies:
        full_dir, full_manifest, full_summary = find_case(
            run_root,
            args,
            topology,
            "full",
        )

        quarter_dir, quarter_manifest, quarter_summary = find_case(
            run_root,
            args,
            topology,
            "quarter_xy",
        )

        w_full = float(
            full_summary["midspan"]["U3_midplane_avg_mm"]
        )

        w_quarter = float(
            quarter_summary["midspan"]["U3_midplane_avg_mm"]
        )

        e_full = reconstructed_energy(full_summary)
        e_quarter = reconstructed_energy(quarter_summary)

        full_profile = read_profile(full_dir)
        quarter_profile = read_profile(quarter_dir)

        tau_rmse, n_tau = relative_rmse(
            full_profile,
            quarter_profile,
            "S13",
        )

        gamma_rmse, n_gamma = relative_rmse(
            full_profile,
            quarter_profile,
            "gamma13",
        )

        row = {
            "topology": topology,
            "full_case": str(full_dir),
            "quarter_case": str(quarter_dir),
            "reflection_error_x": float(
                quarter_manifest["domain"][
                    "constitutive_reflection_relative_error_x"
                ]
            ),
            "reflection_error_y": float(
                quarter_manifest["domain"][
                    "constitutive_reflection_relative_error_y"
                ]
            ),
            "U3_full_mm": w_full,
            "U3_quarter_mm": w_quarter,
            "U3_quarter_vs_full_percent": percent_difference(
                w_quarter,
                w_full,
            ),
            "SENER_full_reconstructed_Nmm": e_full,
            "SENER_quarter_reconstructed_Nmm": e_quarter,
            "SENER_quarter_vs_full_percent": percent_difference(
                e_quarter,
                e_full,
            ),
            "left_shear_mid_tau13_RMSE_percent": tau_rmse,
            "left_shear_mid_gamma13_RMSE_percent": gamma_rmse,
            "profile_matching_points": min(n_tau, n_gamma),
        }

        rows.append(row)

    output = args.output

    if not output.is_absolute():
        output = Path.cwd() / output

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "host_quarter_full_topology_audit_v1",
        "criteria": {
            "relative_density": args.relative_density,
            "n_shear": args.n_shear,
            "campaign_type": args.campaign_type,
            "load_case": args.load_case,
            "state": args.state,
            "element_type": args.element_type,
            "ne_cell": args.ne_cell,
            "face_ne_z": args.face_ne_z,
            "fixture_mode": args.fixture_mode,
            "model_implementation_revision": (
                args.model_implementation_revision
            ),
        },
        "rows": rows,
    }

    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print()
    print("=" * 126)
    print("4PB QUARTER_XY vs FULL-DOMAIN SYMMETRY AUDIT")
    print("=" * 126)
    print(
        "{:>10s} {:>12s} {:>12s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
            "topology",
            "U3 full",
            "U3 quarter",
            "dU3 [%]",
            "dU [%]",
            "tau RMSE",
            "gam RMSE",
        )
    )

    for row in rows:
        print(
            "{:>10s} {:12.8f} {:12.8f} {:11.4f} {:11.4f} {:11.4f} {:11.4f}".format(
                row["topology"],
                row["U3_full_mm"],
                row["U3_quarter_mm"],
                row["U3_quarter_vs_full_percent"],
                row["SENER_quarter_vs_full_percent"],
                row["left_shear_mid_tau13_RMSE_percent"],
                row["left_shear_mid_gamma13_RMSE_percent"],
            )
        )

    print()
    print("Wrote:")
    print(" ", output)
    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
