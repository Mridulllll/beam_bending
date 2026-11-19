# -*- coding: utf-8 -*-
"""
Compare a full HOST 4PB FE case with a quarter_xy symmetry counterpart.

Quarter reconstruction:
    midspan displacement: direct
    total strain energy: 4 x quarter
    section resultants: 2 x quarter (half-width section)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCHEMA = "host_4pb_domain_symmetry_comparison_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-case", required=True)
    parser.add_argument("--quarter-case", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rel_error(value, reference):
    return abs(float(value) - float(reference)) / max(
        abs(float(reference)),
        1.0e-30,
    )


def rows_by_x(rows):
    return {
        round(float(row["x_center_mm"]), 9): row
        for row in rows
    }


def main():
    args = parse_args()

    full_case = Path(args.full_case).resolve()
    quarter_case = Path(args.quarter_case).resolve()

    fm = load_json(full_case / "manifest.json")
    qm = load_json(quarter_case / "manifest.json")
    fs = load_json(full_case / "post" / "summary.json")
    qs = load_json(quarter_case / "post" / "summary.json")

    domain = qm.get("domain", {})

    if str(domain.get("mode", "")).lower() != "quarter_xy":
        raise RuntimeError(
            "The supplied quarter case is not domain.mode='quarter_xy'."
        )

    energy_factor = float(
        domain.get("energy_reconstruction_factor", 4.0)
    )
    resultant_factor = float(
        domain.get("section_resultant_reconstruction_factor", 2.0)
    )

    w_full = float(fs["midspan"]["U3_midplane_avg_mm"])
    w_quarter = float(qs["midspan"]["U3_midplane_avg_mm"])

    e_full = float(fs["energy"]["SENER_integral_Nmm"])
    e_quarter_raw = float(qs["energy"]["SENER_integral_Nmm"])
    e_quarter_full = energy_factor * e_quarter_raw

    full_rows = rows_by_x(
        load_csv(full_case / "post" / "section_resultants.csv")
    )
    quarter_rows = rows_by_x(
        load_csv(quarter_case / "post" / "section_resultants.csv")
    )

    P = float(fm["fixture"]["total_applied_load_N"])
    g = fm["geometry"]
    Mmax = (
        0.5 * P
        * (
            float(g["load_left"])
            - float(g["support_left"])
        )
    )

    differences = []

    for x_key, qrow in sorted(quarter_rows.items()):
        frow = full_rows.get(x_key)
        if frow is None:
            continue

        differences.append(
            {
                "x_mm": float(qrow["x_center_mm"]),
                "dNx_N": (
                    resultant_factor * float(qrow["Nx_raw_N"])
                    - float(frow["Nx_raw_N"])
                ),
                "dQz_N": (
                    resultant_factor * float(qrow["Qz_raw_from_S13_N"])
                    - float(frow["Qz_raw_from_S13_N"])
                ),
                "dMy_Nmm": (
                    resultant_factor * float(qrow["My_beam_convention_Nmm"])
                    - float(frow["My_beam_convention_Nmm"])
                ),
            }
        )

    max_dnx = max(abs(row["dNx_N"]) for row in differences)
    max_dq = max(abs(row["dQz_N"]) for row in differences)
    max_dm = max(abs(row["dMy_Nmm"]) for row in differences)

    result = {
        "schema": SCHEMA,
        "full_case": str(full_case),
        "quarter_case": str(quarter_case),
        "full_case_hash": fm["case"]["case_hash"],
        "quarter_case_hash": qm["case"]["case_hash"],
        "material_reflection_relative_error_x": domain.get(
            "constitutive_reflection_relative_error_x"
        ),
        "material_reflection_relative_error_y": domain.get(
            "constitutive_reflection_relative_error_y"
        ),
        "midspan": {
            "full_U3_mm": w_full,
            "quarter_U3_mm": w_quarter,
            "relative_error": rel_error(w_quarter, w_full),
        },
        "energy": {
            "full_SENER_Nmm": e_full,
            "quarter_raw_SENER_Nmm": e_quarter_raw,
            "quarter_reconstruction_factor": energy_factor,
            "quarter_reconstructed_SENER_Nmm": e_quarter_full,
            "relative_error": rel_error(e_quarter_full, e_full),
        },
        "section_resultants": {
            "quarter_reconstruction_factor": resultant_factor,
            "matched_slab_count": len(differences),
            "max_abs_dNx_over_P": max_dnx / max(P, 1.0e-30),
            "max_abs_dQz_over_P": max_dq / max(P, 1.0e-30),
            "max_abs_dMy_over_Mmax": max_dm / max(Mmax, 1.0e-30),
        },
    }

    output = (
        Path(args.output).resolve()
        if args.output
        else quarter_case
        / "post"
        / "validation"
        / "quarter_vs_full.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("=" * 78)
    print("HOST 4PB FULL vs QUARTER_XY")
    print("=" * 78)
    print("Full case    : {}".format(full_case))
    print("Quarter case : {}".format(quarter_case))
    print()
    print(
        "Midspan U3 : full={:.12g} mm  quarter={:.12g} mm  rel.err={:.6%}"
        .format(
            w_full,
            w_quarter,
            result["midspan"]["relative_error"],
        )
    )
    print(
        "SENER      : full={:.12g} Nmm  reconstructed={:.12g} Nmm  rel.err={:.6%}"
        .format(
            e_full,
            e_quarter_full,
            result["energy"]["relative_error"],
        )
    )
    print(
        "Sections   : |dQ|/P={:.3e}, |dM|/Mmax={:.3e}, |dN|/P={:.3e}"
        .format(
            result["section_resultants"]["max_abs_dQz_over_P"],
            result["section_resultants"]["max_abs_dMy_over_Mmax"],
            result["section_resultants"]["max_abs_dNx_over_P"],
        )
    )
    print(
        "Material reflection residuals: x={:.3e}, y={:.3e}".format(
            float(result["material_reflection_relative_error_x"]),
            float(result["material_reflection_relative_error_y"]),
        )
    )
    print()
    print("Wrote: {}".format(output))


if __name__ == "__main__":
    main()
