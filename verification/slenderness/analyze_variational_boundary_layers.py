#!/usr/bin/env python3
"""
Analyze the longitudinal boundary-layer spectrum of the variational two-mode
HOST beam.

This script does not derive a new section mode and does not run Abaqus.
It consumes one or more variational_second_shear_mode.json files written by

    derive_variational_second_shear_mode.py

and reconstructs the exact eight-state first-order matrix

    y' = A y,

with

    y = [w, theta, beta1, beta2, M, V, H1, H2]^T.

The nonzero eigenvalues of A are the actual longitudinal higher-order
characteristic exponents.  For a stable eigenvalue

    r = -k + i omega,

the decay length is

    ell_decay = 1/k,

and, when omega != 0, the oscillation wavelength is

    Lambda = 2*pi/|omega|.

These are the longitudinal boundary-layer scales of the enriched beam.  They
must be distinguished from the section eigenproblem scale

    ell_section = 1/sqrt(lambda_section),

which is a through-thickness modal scale.

The script can also compare the boundary-layer lengths across the section-mode
discretization convergence folders and relate the slowest decay length to the
4PB shear span a.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from run_fe_informed_two_mode_beam import state_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--convergence-root",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/variational_mode_convergence"
        ),
        help=(
            "Directory containing cXXX_fXXX/variational_second_shear_mode.json."
        ),
    )

    parser.add_argument(
        "--variational-json",
        type=Path,
        action="append",
        default=None,
        help=(
            "Analyze an explicit variational_second_shear_mode.json. "
            "May be supplied multiple times. If omitted, convergence-root "
            "is scanned."
        ),
    )

    parser.add_argument(
        "--slenderness",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/slenderness_sweep.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/variational_boundary_layers"
        ),
    )

    parser.add_argument(
        "--zero-tol",
        type=float,
        default=1.0e-8,
        help=(
            "Relative tolerance used to classify state-matrix eigenvalues "
            "as zero."
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_variational_jsons(
    convergence_root: Path,
) -> List[Tuple[str, Path]]:
    if not convergence_root.is_dir():
        raise FileNotFoundError(convergence_root)

    found: List[Tuple[str, Path]] = []

    for child in convergence_root.iterdir():
        if not child.is_dir():
            continue

        candidate = (
            child
            / "variational_second_shear_mode.json"
        )

        if candidate.is_file():
            found.append(
                (
                    child.name,
                    candidate,
                )
            )

    if not found:
        raise RuntimeError(
            "No variational_second_shear_mode.json files found below {}."
            .format(convergence_root)
        )

    def sort_key(item: Tuple[str, Path]) -> Tuple[int, int, str]:
        tag = item[0]

        match = re.search(
            r"c(\d+)_f(\d+)",
            tag,
        )

        if match:
            return (
                int(match.group(1)),
                int(match.group(2)),
                tag,
            )

        return (
            10**9,
            10**9,
            tag,
        )

    found.sort(key=sort_key)

    return found


def parse_resolution(
    tag: str,
    report: dict,
) -> Tuple[int, int]:
    match = re.search(
        r"c(\d+)_f(\d+)",
        tag,
    )

    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
        )

    section = report.get(
        "section",
        {},
    )

    return (
        int(
            section.get(
                "core_elements_half",
                -1,
            )
        ),
        int(
            section.get(
                "face_elements_half",
                -1,
            )
        ),
    )


def spectral_analysis(
    K5: np.ndarray,
    *,
    zero_tol: float,
) -> dict:
    K5 = np.asarray(
        K5,
        dtype=float,
    )

    if K5.shape != (5, 5):
        raise ValueError(
            "Expected K5 shape (5,5), got {}.".format(
                K5.shape
            )
        )

    A = state_matrix(
        K5
    )

    eigenvalues, eigenvectors = np.linalg.eig(
        A
    )

    scale = max(
        1.0,
        float(
            np.max(
                np.abs(
                    eigenvalues
                )
            )
        ),
    )

    threshold = (
        float(zero_tol)
        * scale
    )

    zero_indices = [
        i
        for i, value in enumerate(eigenvalues)
        if abs(value) <= threshold
    ]

    nonzero_indices = [
        i
        for i, value in enumerate(eigenvalues)
        if abs(value) > threshold
    ]

    modes = []

    for i in nonzero_indices:
        value = complex(
            eigenvalues[i]
        )

        vector = np.asarray(
            eigenvectors[:, i],
            dtype=complex,
        )

        residual = float(
            np.linalg.norm(
                A @ vector
                - value * vector
            )
            / max(
                np.linalg.norm(
                    A @ vector
                ),
                abs(value)
                * np.linalg.norm(vector),
                1.0e-30,
            )
        )

        real = float(value.real)
        imag = float(value.imag)

        if abs(real) > threshold:
            decay_length = (
                1.0 / abs(real)
            )
        else:
            decay_length = float("inf")

        if abs(imag) > threshold:
            wavelength = (
                2.0
                * math.pi
                / abs(imag)
            )
        else:
            wavelength = float("inf")

        beta1 = abs(
            vector[2]
        )
        beta2 = abs(
            vector[3]
        )

        beta_norm = math.sqrt(
            float(
                beta1 * beta1
                + beta2 * beta2
            )
        )

        if beta_norm > 0.0:
            beta2_fraction = float(
                beta2 / beta_norm
            )
        else:
            beta2_fraction = float("nan")

        modes.append(
            {
                "real_per_mm": real,
                "imag_per_mm": imag,
                "abs_per_mm": float(abs(value)),
                "stable": bool(real < -threshold),
                "unstable": bool(real > threshold),
                "purely_imaginary": bool(
                    abs(real) <= threshold
                    and abs(imag) > threshold
                ),
                "decay_length_mm": float(
                    decay_length
                ),
                "oscillation_wavelength_mm": float(
                    wavelength
                ),
                "beta2_fraction_within_beta_subspace": (
                    beta2_fraction
                ),
                "eigen_residual_relative": residual,
            }
        )

    stable = [
        row
        for row in modes
        if row["stable"]
    ]

    stable.sort(
        key=lambda row: row["decay_length_mm"],
        reverse=True,
    )

    positive = [
        row
        for row in modes
        if row["unstable"]
    ]

    positive.sort(
        key=lambda row: row["decay_length_mm"],
        reverse=True,
    )

    if stable:
        slowest = float(
            stable[0]["decay_length_mm"]
        )
        fastest = float(
            stable[-1]["decay_length_mm"]
        )
    else:
        slowest = float("nan")
        fastest = float("nan")

    return {
        "A": A.tolist(),
        "zero_threshold_per_mm": threshold,
        "zero_eigenvalue_count": len(
            zero_indices
        ),
        "nonzero_eigenvalue_count": len(
            nonzero_indices
        ),
        "modes": modes,
        "stable_modes": stable,
        "unstable_modes": positive,
        "slowest_decay_length_mm": slowest,
        "fastest_decay_length_mm": fastest,
    }


def finite_relative_difference(
    value: float,
    reference: float,
) -> float:
    if not (
        math.isfinite(value)
        and math.isfinite(reference)
    ):
        return float("nan")

    if reference == 0.0:
        return float("nan")

    return (
        value - reference
    ) / reference


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    slenderness_path = args.slenderness.resolve()

    if args.variational_json:
        inputs = [
            (
                path.stem,
                path.resolve(),
            )
            for path in args.variational_json
        ]
    else:
        inputs = discover_variational_jsons(
            args.convergence_root.resolve()
        )

    rows = []
    detailed = {}

    for tag, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

        report = load_json(
            path
        )

        core_elements, face_elements = parse_resolution(
            tag,
            report,
        )

        section = report["section"]
        mode = report["mode"]
        summary = report["summary"]

        spectrum = spectral_analysis(
            np.asarray(
                section["K5_N_zero"],
                dtype=float,
            ),
            zero_tol=float(args.zero_tol),
        )

        h = float(
            section["h_mm"]
        )

        slowest = float(
            spectrum["slowest_decay_length_mm"]
        )

        fastest = float(
            spectrum["fastest_decay_length_mm"]
        )

        row = {
            "tag": tag,
            "core_elements": core_elements,
            "face_elements": face_elements,
            "lambda_section_per_mm2": float(
                mode["lambda_per_mm2"]
            ),
            "ell_section_mm": float(
                mode["characteristic_length_mm"]
            ),
            "ell_section_over_h": float(
                mode["characteristic_length_over_h"]
            ),
            "alpha_local": float(
                section[
                    "local_beta_condensed_alpha"
                ]
            ),
            "beam_RMS_error_percent": (
                100.0
                * float(
                    summary[
                        "RMS_two_mode_relative_error"
                    ]
                )
            ),
            "beam_max_error_percent": (
                100.0
                * float(
                    summary[
                        "max_two_mode_relative_error"
                    ]
                )
            ),
            "zero_eigenvalue_count": int(
                spectrum[
                    "zero_eigenvalue_count"
                ]
            ),
            "nonzero_eigenvalue_count": int(
                spectrum[
                    "nonzero_eigenvalue_count"
                ]
            ),
            "slowest_longitudinal_decay_mm": (
                slowest
            ),
            "slowest_decay_over_h": (
                slowest / h
            ),
            "fastest_longitudinal_decay_mm": (
                fastest
            ),
            "fastest_decay_over_h": (
                fastest / h
            ),
        }

        rows.append(
            row
        )

        detailed[tag] = {
            "source": str(path),
            "spectrum": spectrum,
        }

    if not rows:
        raise RuntimeError(
            "No variational reports analyzed."
        )

    # Last/discretely finest row is the convergence reference.
    reference = rows[-1]

    convergence_rows = []

    for row in rows:
        convergence_rows.append(
            {
                **row,
                "lambda_rel_to_finest": finite_relative_difference(
                    row["lambda_section_per_mm2"],
                    reference["lambda_section_per_mm2"],
                ),
                "ell_section_rel_to_finest": finite_relative_difference(
                    row["ell_section_mm"],
                    reference["ell_section_mm"],
                ),
                "alpha_rel_to_finest": finite_relative_difference(
                    row["alpha_local"],
                    reference["alpha_local"],
                ),
                "slow_decay_rel_to_finest": finite_relative_difference(
                    row["slowest_longitudinal_decay_mm"],
                    reference["slowest_longitudinal_decay_mm"],
                ),
                "fast_decay_rel_to_finest": finite_relative_difference(
                    row["fastest_longitudinal_decay_mm"],
                    reference["fastest_longitudinal_decay_mm"],
                ),
            }
        )

    slenderness_relationship = []

    if slenderness_path.is_file():
        slenderness = load_json(
            slenderness_path
        )

        ell_slow = float(
            reference[
                "slowest_longitudinal_decay_mm"
            ]
        )

        for row in slenderness.get(
            "rows",
            [],
        ):
            fe = row.get(
                "FE_3D_abs_w_mid_mm"
            )

            if fe in ("", None):
                continue

            a = float(
                row["shear_span_mm"]
            )

            slenderness_relationship.append(
                {
                    "n_shear": int(
                        row["n_shear"]
                    ),
                    "a_over_h": float(
                        row["a_over_h"]
                    ),
                    "shear_span_mm": a,
                    "a_over_slowest_decay": (
                        a / ell_slow
                    ),
                    "exp_minus_a_over_slowest_decay": (
                        math.exp(
                            -a / ell_slow
                        )
                    ),
                }
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "variational_boundary_layer_spectrum.json"
    )

    csv_path = (
        output_dir
        / "variational_boundary_layer_convergence.csv"
    )

    span_csv_path = (
        output_dir
        / "shear_span_vs_boundary_layer.csv"
    )

    report = {
        "schema": "host_variational_boundary_layer_spectrum_v1",
        "warning": (
            "The longitudinal decay lengths come from the eigenvalues of the "
            "complete eight-state beam state matrix. They are distinct from "
            "the through-thickness section scale 1/sqrt(lambda_section)."
        ),
        "convergence": convergence_rows,
        "finest": reference,
        "details": detailed,
        "shear_span_relationship": slenderness_relationship,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )
        handle.write(
            "\n"
        )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                convergence_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            convergence_rows
        )

    if slenderness_relationship:
        with span_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    slenderness_relationship[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                slenderness_relationship
            )

    print()
    print("=" * 128)
    print("HOST VARIATIONAL TWO-MODE LONGITUDINAL BOUNDARY-LAYER SPECTRUM")
    print("=" * 128)
    print(
        "{:>12s} {:>8s} {:>8s} {:>13s} {:>13s} {:>13s} {:>13s}".format(
            "tag",
            "core",
            "face",
            "ell_section",
            "ell_slow",
            "ell_fast",
            "alpha",
        )
    )

    for row in convergence_rows:
        print(
            "{:>12s} {:8d} {:8d} {:13.6f} {:13.6f} {:13.6f} {:13.9f}".format(
                str(
                    row["tag"]
                ),
                int(
                    row["core_elements"]
                ),
                int(
                    row["face_elements"]
                ),
                float(
                    row["ell_section_mm"]
                ),
                float(
                    row[
                        "slowest_longitudinal_decay_mm"
                    ]
                ),
                float(
                    row[
                        "fastest_longitudinal_decay_mm"
                    ]
                ),
                float(
                    row["alpha_local"]
                ),
            )
        )

    print()
    print(
        "Finest state spectrum:"
    )

    finest_detail = detailed[
        reference["tag"]
    ]["spectrum"]

    print(
        "  zero eigenvalues     : {}".format(
            finest_detail[
                "zero_eigenvalue_count"
            ]
        )
    )
    print(
        "  nonzero eigenvalues  : {}".format(
            finest_detail[
                "nonzero_eigenvalue_count"
            ]
        )
    )

    for i, mode in enumerate(
        finest_detail[
            "stable_modes"
        ],
        start=1,
    ):
        print(
            "  stable mode {:d}: r={:+.9e}{:+.9e}j  "
            "ell={:.9f} mm  beta2_fraction={:.6f}  residual={:.3e}".format(
                i,
                float(
                    mode["real_per_mm"]
                ),
                float(
                    mode["imag_per_mm"]
                ),
                float(
                    mode["decay_length_mm"]
                ),
                float(
                    mode[
                        "beta2_fraction_within_beta_subspace"
                    ]
                ),
                float(
                    mode[
                        "eigen_residual_relative"
                    ]
                ),
            )
        )

    if slenderness_relationship:
        print()
        print(
            "{:>6s} {:>8s} {:>12s} {:>14s} {:>14s}".format(
                "N_A",
                "a/h",
                "a [mm]",
                "a/ell_slow",
                "exp(-a/ell)",
            )
        )

        for row in slenderness_relationship:
            print(
                "{:6d} {:8.3f} {:12.3f} {:14.6f} {:14.6e}".format(
                    int(
                        row["n_shear"]
                    ),
                    float(
                        row["a_over_h"]
                    ),
                    float(
                        row["shear_span_mm"]
                    ),
                    float(
                        row[
                            "a_over_slowest_decay"
                        ]
                    ),
                    float(
                        row[
                            "exp_minus_a_over_slowest_decay"
                        ]
                    ),
                )
            )

    print()
    print("Wrote:")
    print(" ", json_path)
    print(" ", csv_path)

    if slenderness_relationship:
        print(
            " ",
            span_csv_path,
        )

    print("=" * 128)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
