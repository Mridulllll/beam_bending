#!/usr/bin/env python3
"""
Consolidate HOST 4PB validation/transfer result folders into one reproducible
summary and a compact set of publication/thesis-ready diagnostic plots.

Input:
    verification/transfer/results/**/topology_transfer.json

The script deduplicates repeated FE cases (for example the same Gyroid RD12
baseline appearing in topology, face-thickness, and core-height result folders)
and prefers the most information-rich row for each case.

Outputs:
    validation_summary.json
    validation_summary.csv
    validation_statistics.csv
    error_vs_a_over_h.png
    profile_rmse_vs_a_over_h.png
    gyroid_density_transfer.png
    gyroid_face_thickness_transfer.png
    gyroid_core_height_transfer.png
    gyroid_boundary_length_scaling.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_ROOT = Path(
    "verification/transfer/results"
)

DEFAULT_OUTPUT_DIR = Path(
    "verification/transfer/results/consolidated"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def numeric(value: Any) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def hydrate_row_from_case_manifest(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Fill geometry/provenance fields that may be absent from older transfer
    result JSON files.

    Earlier transfer campaigns were generated before face/core geometry fields
    were added to analyze_4pb_topology_transfer.py.  The FE case manifest is
    authoritative and already contains those values, so use it to make
    historical and newer transfer rows comparable without rerunning HOST.
    """
    enriched = dict(row)

    case_path_text = str(
        enriched.get(
            "case_path",
            "",
        )
    ).strip()

    if not case_path_text:
        return enriched

    manifest_path = (
        Path(case_path_text)
        / "manifest.json"
    )

    if not manifest_path.is_file():
        return enriched

    try:
        manifest = load_json(
            manifest_path
        )
    except Exception:
        return enriched

    geometry = manifest.get(
        "geometry",
        {},
    )

    def put_if_missing(
        key: str,
        value: Any,
    ) -> None:
        existing = enriched.get(
            key
        )

        if existing is None:
            enriched[key] = value
            return

        if (
            isinstance(
                existing,
                float,
            )
            and not math.isfinite(
                existing
            )
        ):
            enriched[key] = value

    if geometry:
        try:
            put_if_missing(
                "n_core_height",
                int(
                    geometry[
                        "n_height"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

        try:
            put_if_missing(
                "core_height_mm",
                float(
                    geometry[
                        "core_height"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

        try:
            put_if_missing(
                "face_bottom_mm",
                float(
                    geometry[
                        "face_bottom"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

        try:
            put_if_missing(
                "face_top_mm",
                float(
                    geometry[
                        "face_top"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

        try:
            put_if_missing(
                "total_height_mm",
                float(
                    geometry[
                        "total_height"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

        try:
            core_height = float(
                geometry[
                    "core_height"
                ]
            )

            face_mean = 0.5 * (
                float(
                    geometry[
                        "face_bottom"
                    ]
                )
                + float(
                    geometry[
                        "face_top"
                    ]
                )
            )

            put_if_missing(
                "face_to_core_thickness_ratio",
                face_mean
                / core_height,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            pass

        try:
            put_if_missing(
                "a_over_h",
                float(
                    geometry[
                        "shear_span"
                    ]
                )
                / float(
                    geometry[
                        "total_height"
                    ]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ):
            pass

    return enriched


def row_information_score(row: dict[str, Any]) -> int:
    score = 0

    for value in row.values():
        if value is None:
            continue

        if isinstance(value, float) and not math.isfinite(value):
            continue

        score += 1

    return score


def row_identity(row: dict[str, Any]) -> str:
    case_id = str(
        row.get(
            "case_id",
            "",
        )
    ).strip()

    if case_id:
        return "case_id:" + case_id

    case_path = str(
        row.get(
            "case_path",
            "",
        )
    ).strip()

    if case_path:
        return "case_path:" + case_path.lower()

    keys = (
        row.get("topology"),
        row.get("rd_target"),
        row.get("n_shear"),
        row.get("n_core_height"),
        row.get("face_bottom_mm"),
        row.get("face_top_mm"),
    )

    return "fallback:" + repr(keys)


def discover_rows(
    results_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    discovered_files = sorted(
        results_root.rglob(
            "topology_transfer.json"
        )
    )

    selected: dict[
        str,
        dict[str, Any],
    ] = {}

    sources: dict[
        str,
        list[str],
    ] = {}

    for path in discovered_files:
        try:
            payload = load_json(path)
        except Exception:
            continue

        for raw_row in payload.get(
            "rows",
            [],
        ):
            if not isinstance(
                raw_row,
                dict,
            ):
                continue

            row = hydrate_row_from_case_manifest(
                dict(raw_row)
            )

            identity = row_identity(
                row
            )

            relative_source = str(
                path.relative_to(
                    results_root
                )
            )

            sources.setdefault(
                identity,
                [],
            ).append(
                relative_source
            )

            previous = selected.get(
                identity
            )

            if previous is None:
                selected[identity] = row
                continue

            if (
                row_information_score(row)
                > row_information_score(
                    previous
                )
            ):
                selected[identity] = row

    rows = []

    for identity, row in selected.items():
        enriched = dict(row)

        enriched[
            "result_sources"
        ] = ";".join(
            sorted(
                set(
                    sources[
                        identity
                    ]
                )
            )
        )

        rows.append(
            enriched
        )

    rows.sort(
        key=lambda row: (
            str(
                row.get(
                    "topology",
                    "",
                )
            ),
            numeric(
                row.get(
                    "rd_target"
                )
            )
            or 0.0,
            numeric(
                row.get(
                    "core_height_mm"
                )
            )
            or 0.0,
            numeric(
                row.get(
                    "face_bottom_mm"
                )
            )
            or 0.0,
            int(
                row.get(
                    "n_shear",
                    0,
                )
            ),
        )
    )

    return (
        rows,
        [
            str(
                path.relative_to(
                    results_root
                )
            )
            for path in discovered_files
        ],
    )


def rms(
    values: list[float],
) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                array ** 2
            )
        )
    )


def max_abs(
    values: list[float],
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(
                    values,
                    dtype=float,
                )
            )
        )
    )


def collect_errors(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values = []

    for row in rows:
        value = numeric(
            row.get(
                field
            )
        )

        if value is not None:
            values.append(
                value
            )

    return values


def subset_statistics(
    label: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    one = collect_errors(
        rows,
        "one_mode_error_percent",
    )

    two = collect_errors(
        rows,
        "two_mode_error_percent",
    )

    return {
        "subset": label,
        "cases": len(rows),
        "rms_one_mode_percent": (
            rms(one)
            if one
            else None
        ),
        "rms_two_mode_percent": (
            rms(two)
            if two
            else None
        ),
        "max_abs_one_mode_percent": (
            max_abs(one)
            if one
            else None
        ),
        "max_abs_two_mode_percent": (
            max_abs(two)
            if two
            else None
        ),
    }


def is_close(
    value: Any,
    target: float,
    tol: float = 1.0e-9,
) -> bool:
    parsed = numeric(value)

    return (
        parsed is not None
        and abs(
            parsed - target
        )
        <= tol
    )


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    topology: str | None = None,
    rd: float | None = None,
    n_shear: int | None = None,
    face_mm: float | None = None,
    core_height_mm: float | None = None,
) -> list[dict[str, Any]]:
    output = []

    for row in rows:
        if (
            topology is not None
            and str(
                row.get(
                    "topology",
                    "",
                )
            ).lower()
            != topology.lower()
        ):
            continue

        if (
            rd is not None
            and not is_close(
                row.get(
                    "rd_target"
                ),
                rd,
            )
        ):
            continue

        if (
            n_shear is not None
            and int(
                row.get(
                    "n_shear",
                    -1,
                )
            )
            != int(n_shear)
        ):
            continue

        if face_mm is not None:
            if not (
                is_close(
                    row.get(
                        "face_bottom_mm"
                    ),
                    face_mm,
                )
                and is_close(
                    row.get(
                        "face_top_mm"
                    ),
                    face_mm,
                )
            ):
                continue

        if (
            core_height_mm is not None
            and not is_close(
                row.get(
                    "core_height_mm"
                ),
                core_height_mm,
            )
        ):
            continue

        output.append(
            row
        )

    return output


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_statistics_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def plot_error_vs_a_over_h(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    usable = [
        row
        for row in rows
        if (
            numeric(
                row.get(
                    "a_over_h"
                )
            )
            is not None
            and numeric(
                row.get(
                    "one_mode_error_percent"
                )
            )
            is not None
            and numeric(
                row.get(
                    "two_mode_error_percent"
                )
            )
            is not None
        )
    ]

    usable.sort(
        key=lambda row: float(
            row[
                "a_over_h"
            ]
        )
    )

    x = np.asarray(
        [
            float(
                row[
                    "a_over_h"
                ]
            )
            for row in usable
        ]
    )

    y1 = np.asarray(
        [
            abs(
                float(
                    row[
                        "one_mode_error_percent"
                    ]
                )
            )
            for row in usable
        ]
    )

    y2 = np.asarray(
        [
            abs(
                float(
                    row[
                        "two_mode_error_percent"
                    ]
                )
            )
            for row in usable
        ]
    )

    fig, ax = plt.subplots()

    ax.scatter(
        x,
        y1,
        marker="o",
        label="Equilibrium primary",
    )

    ax.scatter(
        x,
        y2,
        marker="x",
        label="Primary + variational secondary",
    )

    ax.set_xlabel(
        r"$a/h$"
    )

    ax.set_ylabel(
        "Absolute midspan deflection error [%]"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
    )
    plt.close(fig)


def plot_profile_rmse_vs_a_over_h(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    usable = [
        row
        for row in rows
        if (
            numeric(
                row.get(
                    "a_over_h"
                )
            )
            is not None
            and numeric(
                row.get(
                    "tau_profile_RMSE_percent"
                )
            )
            is not None
        )
    ]

    usable.sort(
        key=lambda row: float(
            row[
                "a_over_h"
            ]
        )
    )

    x = np.asarray(
        [
            float(
                row[
                    "a_over_h"
                ]
            )
            for row in usable
        ]
    )

    tau = np.asarray(
        [
            float(
                row[
                    "tau_profile_RMSE_percent"
                ]
            )
            for row in usable
        ]
    )

    gamma = np.asarray(
        [
            float(
                row[
                    "gamma_profile_RMSE_percent"
                ]
            )
            for row in usable
        ]
    )

    fig, ax = plt.subplots()

    ax.scatter(
        x,
        tau,
        marker="o",
        label=r"$\tau_{xz}$",
    )

    ax.scatter(
        x,
        gamma,
        marker="x",
        label=r"$\gamma_{xz}$",
    )

    ax.set_xlabel(
        r"$a/h$"
    )

    ax.set_ylabel(
        "Equilibrium-primary profile RMSE [%]"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
    )
    plt.close(fig)


def gyroid_baseline_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Geometry fields are available in the latest repeated baseline result.
    # When older rows lack them, case identity deduplication prefers the newer,
    # more information-rich row.
    return filter_rows(
        rows,
        topology="Gyroid",
        face_mm=1.0,
        core_height_mm=60.0,
    )


def plot_density_transfer(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    usable = gyroid_baseline_rows(
        rows
    )

    rd_values = sorted(
        {
            float(
                row[
                    "rd_target"
                ]
            )
            for row in usable
            if numeric(
                row.get(
                    "rd_target"
                )
            )
            is not None
        }
    )

    if not rd_values:
        return

    fig, ax = plt.subplots()

    for n_shear in (
        6,
        24,
    ):
        points = []

        for rd in rd_values:
            matches = [
                row
                for row in usable
                if (
                    int(
                        row.get(
                            "n_shear",
                            -1,
                        )
                    )
                    == n_shear
                    and is_close(
                        row.get(
                            "rd_target"
                        ),
                        rd,
                    )
                )
            ]

            if not matches:
                continue

            row = matches[0]

            points.append(
                (
                    rd,
                    abs(
                        float(
                            row[
                                "two_mode_error_percent"
                            ]
                        )
                    ),
                )
            )

        if not points:
            continue

        ax.plot(
            [
                100.0 * item[0]
                for item in points
            ],
            [
                item[1]
                for item in points
            ],
            marker="o",
            label=(
                r"$N_A={}$".format(
                    n_shear
                )
            ),
        )

    ax.set_xlabel(
        "Core relative density [%]"
    )

    ax.set_ylabel(
        "Two-mode absolute deflection error [%]"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
    )
    plt.close(fig)


def plot_face_thickness_transfer(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    usable = filter_rows(
        rows,
        topology="Gyroid",
        rd=0.12,
        core_height_mm=60.0,
    )

    fig, ax = plt.subplots()

    for n_shear in (
        6,
        24,
    ):
        points = []

        for row in usable:
            if int(
                row.get(
                    "n_shear",
                    -1,
                )
            ) != n_shear:
                continue

            bottom = numeric(
                row.get(
                    "face_bottom_mm"
                )
            )

            top = numeric(
                row.get(
                    "face_top_mm"
                )
            )

            error = numeric(
                row.get(
                    "two_mode_error_percent"
                )
            )

            if (
                bottom is None
                or top is None
                or error is None
                or abs(
                    bottom - top
                )
                > 1.0e-9
            ):
                continue

            points.append(
                (
                    bottom,
                    abs(error),
                )
            )

        points.sort()

        if points:
            ax.plot(
                [
                    item[0]
                    for item in points
                ],
                [
                    item[1]
                    for item in points
                ],
                marker="o",
                label=(
                    r"$N_A={}$".format(
                        n_shear
                    )
                ),
            )

    ax.set_xlabel(
        "Symmetric face thickness [mm]"
    )

    ax.set_ylabel(
        "Two-mode absolute deflection error [%]"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
    )
    plt.close(fig)


def plot_core_height_transfer(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    usable = filter_rows(
        rows,
        topology="Gyroid",
        rd=0.12,
        face_mm=1.0,
    )

    fig, ax = plt.subplots()

    for n_shear in (
        6,
        24,
    ):
        points = []

        for row in usable:
            if int(
                row.get(
                    "n_shear",
                    -1,
                )
            ) != n_shear:
                continue

            height = numeric(
                row.get(
                    "core_height_mm"
                )
            )

            error = numeric(
                row.get(
                    "two_mode_error_percent"
                )
            )

            if (
                height is None
                or error is None
            ):
                continue

            points.append(
                (
                    height,
                    abs(error),
                )
            )

        points.sort()

        if points:
            ax.plot(
                [
                    item[0]
                    for item in points
                ],
                [
                    item[1]
                    for item in points
                ],
                marker="o",
                label=(
                    r"$N_A={}$".format(
                        n_shear
                    )
                ),
            )

    ax.set_xlabel(
        "Core height [mm]"
    )

    ax.set_ylabel(
        "Two-mode absolute deflection error [%]"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
    )
    plt.close(fig)


def plot_boundary_length_scaling(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    usable = filter_rows(
        rows,
        topology="Gyroid",
        rd=0.12,
        face_mm=1.0,
        n_shear=6,
    )

    points = []

    for row in usable:
        height = numeric(
            row.get(
                "core_height_mm"
            )
        )

        ell = numeric(
            row.get(
                "ell_slow_BL_mm"
            )
        )

        if (
            height is None
            or ell is None
        ):
            continue

        points.append(
            (
                height,
                ell,
            )
        )

    points.sort()

    if not points:
        return

    heights = np.asarray(
        [
            item[0]
            for item in points
        ]
    )

    lengths = np.asarray(
        [
            item[1]
            for item in points
        ]
    )

    fig, ax = plt.subplots()

    ax.plot(
        heights,
        lengths,
        marker="o",
        label=r"$\ell_{\mathrm{BL}}$",
    )

    fit = np.polyfit(
        heights,
        lengths,
        deg=1,
    )

    fitted = np.polyval(
        fit,
        heights,
    )

    ax.plot(
        heights,
        fitted,
        linestyle="--",
        label=(
            "Linear fit: "
            + r"$\ell_{\mathrm{BL}}="
            + "{:.4f}h_c{:+.3f}$".format(
                fit[0],
                fit[1],
            )
        ),
    )

    ax.set_xlabel(
        "Core height [mm]"
    )

    ax.set_ylabel(
        "Slow boundary-layer length [mm]"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()

    results_root = (
        args.results_root
    )

    if not results_root.is_absolute():
        results_root = (
            Path.cwd()
            / results_root
        )

    results_root = (
        results_root.resolve()
    )

    output_dir = (
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = (
            Path.cwd()
            / output_dir
        )

    output_dir = (
        output_dir.resolve()
    )

    if not results_root.is_dir():
        raise FileNotFoundError(
            results_root
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows, source_files = (
        discover_rows(
            results_root
        )
    )

    if not rows:
        raise RuntimeError(
            "No topology_transfer.json rows were found."
        )

    topology_rows = [
        row
        for row in rows
        if (
            is_close(
                row.get(
                    "rd_target"
                ),
                0.12,
            )
            and is_close(
                row.get(
                    "core_height_mm"
                ),
                60.0,
            )
            and is_close(
                row.get(
                    "face_bottom_mm"
                ),
                1.0,
            )
            and is_close(
                row.get(
                    "face_top_mm"
                ),
                1.0,
            )
            and str(
                row.get(
                    "topology",
                    "",
                )
            )
            in {
                "Primitive",
                "Gyroid",
                "Diamond",
            }
            and int(
                row.get(
                    "n_shear",
                    -1,
                )
            )
            in {
                6,
                24,
            }
        )
    ]

    density_rows = [
        row
        for row in rows
        if (
            str(
                row.get(
                    "topology",
                    "",
                )
            )
            == "Gyroid"
            and is_close(
                row.get(
                    "core_height_mm"
                ),
                60.0,
            )
            and is_close(
                row.get(
                    "face_bottom_mm"
                ),
                1.0,
            )
            and is_close(
                row.get(
                    "face_top_mm"
                ),
                1.0,
            )
            and (
                is_close(
                    row.get(
                        "rd_target"
                    ),
                    0.10,
                )
                or is_close(
                    row.get(
                        "rd_target"
                    ),
                    0.12,
                )
                or is_close(
                    row.get(
                        "rd_target"
                    ),
                    0.16,
                )
            )
            and int(
                row.get(
                    "n_shear",
                    -1,
                )
            )
            in {
                6,
                24,
            }
        )
    ]

    face_rows = [
        row
        for row in rows
        if (
            str(
                row.get(
                    "topology",
                    "",
                )
            )
            == "Gyroid"
            and is_close(
                row.get(
                    "rd_target"
                ),
                0.12,
            )
            and is_close(
                row.get(
                    "core_height_mm"
                ),
                60.0,
            )
            and (
                is_close(
                    row.get(
                        "face_bottom_mm"
                    ),
                    0.5,
                )
                or is_close(
                    row.get(
                        "face_bottom_mm"
                    ),
                    1.0,
                )
                or is_close(
                    row.get(
                        "face_bottom_mm"
                    ),
                    2.0,
                )
            )
            and is_close(
                row.get(
                    "face_bottom_mm"
                ),
                numeric(
                    row.get(
                        "face_top_mm"
                    )
                )
                or -999.0,
            )
            and int(
                row.get(
                    "n_shear",
                    -1,
                )
            )
            in {
                6,
                24,
            }
        )
    ]

    height_rows = [
        row
        for row in rows
        if (
            str(
                row.get(
                    "topology",
                    "",
                )
            )
            == "Gyroid"
            and is_close(
                row.get(
                    "rd_target"
                ),
                0.12,
            )
            and is_close(
                row.get(
                    "face_bottom_mm"
                ),
                1.0,
            )
            and is_close(
                row.get(
                    "face_top_mm"
                ),
                1.0,
            )
            and (
                is_close(
                    row.get(
                        "core_height_mm"
                    ),
                    30.0,
                )
                or is_close(
                    row.get(
                        "core_height_mm"
                    ),
                    60.0,
                )
                or is_close(
                    row.get(
                        "core_height_mm"
                    ),
                    90.0,
                )
            )
            and int(
                row.get(
                    "n_shear",
                    -1,
                )
            )
            in {
                6,
                24,
            }
        )
    ]

    statistics = [
        subset_statistics(
            "all_unique_cases",
            rows,
        ),
        subset_statistics(
            "topology_transfer_rd12",
            topology_rows,
        ),
        subset_statistics(
            "gyroid_density_transfer",
            density_rows,
        ),
        subset_statistics(
            "gyroid_face_thickness_transfer",
            face_rows,
        ),
        subset_statistics(
            "gyroid_core_height_transfer",
            height_rows,
        ),
    ]

    summary_path = (
        output_dir
        / "validation_summary.json"
    )

    csv_path = (
        output_dir
        / "validation_summary.csv"
    )

    stats_path = (
        output_dir
        / "validation_statistics.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema": "host_4pb_consolidated_validation_v1",
                "results_root": str(
                    results_root
                ),
                "source_files": source_files,
                "statistics": statistics,
                "rows": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    write_csv(
        csv_path,
        rows,
    )

    write_statistics_csv(
        stats_path,
        statistics,
    )

    plot_error_vs_a_over_h(
        rows,
        output_dir
        / "error_vs_a_over_h.png",
    )

    plot_profile_rmse_vs_a_over_h(
        rows,
        output_dir
        / "profile_rmse_vs_a_over_h.png",
    )

    plot_density_transfer(
        rows,
        output_dir
        / "gyroid_density_transfer.png",
    )

    plot_face_thickness_transfer(
        rows,
        output_dir
        / "gyroid_face_thickness_transfer.png",
    )

    plot_core_height_transfer(
        rows,
        output_dir
        / "gyroid_core_height_transfer.png",
    )

    plot_boundary_length_scaling(
        rows,
        output_dir
        / "gyroid_boundary_length_scaling.png",
    )

    print()
    print("=" * 112)
    print("HOST 4PB CONSOLIDATED VALIDATION")
    print("=" * 112)
    print(
        "Unique FE/HOST cases: {}".format(
            len(rows)
        )
    )
    print(
        "Campaign rows: topology={}  density={}  face={}  core-height={}".format(
            len(topology_rows),
            len(density_rows),
            len(face_rows),
            len(height_rows),
        )
    )
    print()

    print(
        "{:<34s} {:>6s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
            "subset",
            "cases",
            "RMS 1 [%]",
            "RMS 2 [%]",
            "max 1 [%]",
            "max 2 [%]",
        )
    )

    for item in statistics:
        print(
            "{:<34s} {:6d} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(
                item[
                    "subset"
                ],
                item[
                    "cases"
                ],
                item[
                    "rms_one_mode_percent"
                ],
                item[
                    "rms_two_mode_percent"
                ],
                item[
                    "max_abs_one_mode_percent"
                ],
                item[
                    "max_abs_two_mode_percent"
                ],
            )
        )

    height_points = sorted(
        {
            (
                numeric(
                    row.get(
                        "core_height_mm"
                    )
                ),
                numeric(
                    row.get(
                        "ell_slow_BL_mm"
                    )
                ),
            )
            for row in height_rows
            if (
                numeric(
                    row.get(
                        "core_height_mm"
                    )
                )
                is not None
                and numeric(
                    row.get(
                        "ell_slow_BL_mm"
                    )
                )
                is not None
            )
        }
    )

    if height_points:
        print()
        print(
            "Gyroid RD12, tf=1 mm boundary-layer scaling:"
        )

        for height, ell in height_points:
            print(
                "  h_c={:6.1f} mm  ell_BL={:9.4f} mm  ell_BL/h_c={:.6f}".format(
                    height,
                    ell,
                    ell / height,
                )
            )

    print()
    print("Wrote:")
    for path in (
        summary_path,
        csv_path,
        stats_path,
    ):
        print(
            "  {}".format(
                path
            )
        )

    print()
    print(
        "Plots: {}".format(
            output_dir
        )
    )
    print("=" * 112)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
