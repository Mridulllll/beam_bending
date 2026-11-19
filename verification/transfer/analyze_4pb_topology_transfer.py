#!/usr/bin/env python3
"""
Analyze topology transfer of the equilibrium-enriched HOST 4PB beam theory.

The script scans already solved/postprocessed Abaqus 4PB case directories,
selects the requested topology/slenderness cases by their manifests, rebuilds
the HOST section from each case's own C_hom_model_MPa tensor, and compares:

  * equilibrium-derived primary mode only;
  * equilibrium-derived primary + variational secondary mode;

against the corresponding 3-D FE midspan deflection.

FE through-thickness gamma13/tau13 profiles are used only for validation, never
to construct the production theory modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from host.beam_point_loads import (
    PointForce,
    solve_simply_supported_point_loads,
)

from host.beam_two_mode import (
    WarpingMode,
    boundary_layer_spectrum,
    build_two_mode_beam_section,
    local_shear_condensation_ratio,
)

from host.beam_two_mode_point_loads import (
    solve_simply_supported_two_mode_point_loads,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.equilibrium_shear_modes import (
    derive_equilibrium_shear_mode,
)

from host.variational_shear_modes import (
    BeamLayer2D,
    derive_variational_second_shear_mode,
    free_stress_beam_stiffness,
)


DEFAULT_OUTPUT = Path(
    "verification/transfer/results/"
    "topology_rd12_initial"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        default=[
            "Primitive",
            "Gyroid",
            "Diamond",
        ],
    )

    parser.add_argument(
        "--n-shear",
        nargs="+",
        type=int,
        default=[
            6,
            24,
        ],
    )

    parser.add_argument(
        "--relative-density",
        type=float,
        default=0.12,
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
        "--n-core-height",
        type=int,
        default=None,
        help=(
            "Optional number of nominal unit cells through the core height. "
            "If omitted, core height is not used for case selection."
        ),
    )

    parser.add_argument(
        "--face-bottom-mm",
        type=float,
        default=None,
        help=(
            "Optional bottom face-sheet thickness filter. "
            "If omitted, face thickness is not used for case selection."
        ),
    )

    parser.add_argument(
        "--face-top-mm",
        type=float,
        default=None,
        help=(
            "Optional top face-sheet thickness filter. "
            "If omitted, face thickness is not used for case selection."
        ),
    )

    parser.add_argument(
        "--fixture-mode",
        default="section_distributing",
    )

    parser.add_argument(
        "--domain-mode",
        default="quarter_xy",
    )

    parser.add_argument(
        "--model-implementation-revision",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--core-mode-elements",
        type=int,
        default=240,
    )

    parser.add_argument(
        "--face-mode-elements",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-segment-mm",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def close(
    a: float,
    b: float,
    tol: float = 1.0e-9,
) -> bool:
    return abs(
        float(a) - float(b)
    ) <= float(tol)


def case_matches(
    manifest: dict,
    *,
    topology: str,
    n_shear: int,
    relative_density: float,
    campaign_type: str,
    load_case: str,
    state: str,
    element_type: str,
    ne_cell: int,
    face_ne_z: int,
    n_core_height: int | None,
    face_bottom_mm: float | None,
    face_top_mm: float | None,
    fixture_mode: str,
    domain_mode: str,
    model_implementation_revision: int,
) -> bool:
    query = manifest.get(
        "database_query",
        {},
    )

    geometry = manifest.get(
        "geometry",
        {},
    )

    mesh = manifest.get(
        "mesh",
        {},
    )

    fixture = manifest.get(
        "fixture",
        {},
    )

    domain = manifest.get(
        "domain",
        {},
    )

    try:
        return (
            int(
                manifest.get(
                    "model_implementation_revision"
                )
            )
            == int(
                model_implementation_revision
            )
            and str(
                query.get(
                    "topology",
                    "",
                )
            ).lower()
            == str(topology).lower()
            and close(
                float(
                    query.get(
                        "relative_density_target"
                    )
                ),
                float(relative_density),
            )
            and str(
                query.get(
                    "campaign_type",
                    "",
                )
            ).lower()
            == str(
                campaign_type
            ).lower()
            and str(
                query.get(
                    "optimization_load_case",
                    "",
                )
            ).lower()
            == str(load_case).lower()
            and str(
                query.get(
                    "requested_state",
                    "",
                )
            ).lower()
            == str(state).lower()
            and int(
                geometry.get(
                    "n_shear"
                )
            )
            == int(n_shear)
            and (
                n_core_height is None
                or int(
                    geometry.get(
                        "n_height"
                    )
                )
                == int(n_core_height)
            )
            and str(
                mesh.get(
                    "element_type",
                    "",
                )
            ).upper()
            == str(element_type).upper()
            and int(
                mesh.get(
                    "core_elements_per_cell_edge"
                )
            )
            == int(ne_cell)
            and int(
                mesh.get(
                    "face_elements_through_thickness"
                )
            )
            == int(face_ne_z)
            and (
                face_bottom_mm is None
                or close(
                    float(
                        geometry.get(
                            "face_bottom"
                        )
                    ),
                    float(face_bottom_mm),
                )
            )
            and (
                face_top_mm is None
                or close(
                    float(
                        geometry.get(
                            "face_top"
                        )
                    ),
                    float(face_top_mm),
                )
            )
            and str(
                fixture.get(
                    "mode",
                    "",
                )
            ).lower()
            == str(
                fixture_mode
            ).lower()
            and str(
                domain.get(
                    "mode",
                    "",
                )
            ).lower()
            == str(
                domain_mode
            ).lower()
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def discover_case(
    run_root: Path,
    **criteria,
):
    candidates = []

    for case_dir in run_root.iterdir():
        if not case_dir.is_dir():
            continue

        manifest_path = (
            case_dir
            / "manifest.json"
        )

        summary_path = (
            case_dir
            / "post"
            / "summary.json"
        )

        if (
            not manifest_path.is_file()
            or not summary_path.is_file()
        ):
            continue

        try:
            manifest = load_json(
                manifest_path
            )

            summary = load_json(
                summary_path
            )

        except Exception:
            continue

        if not case_matches(
            manifest,
            **criteria,
        ):
            continue

        if str(
            summary.get(
                "overall_status",
                "",
            )
        ).upper() != "PASS":
            continue

        candidates.append(
            (
                float(
                    case_dir.stat().st_mtime
                ),
                case_dir,
                manifest,
                summary,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No QA-PASS case found for topology={!r}, N_A={}.".format(
                criteria["topology"],
                criteria["n_shear"],
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, case_dir, manifest, summary = (
        candidates[0]
    )

    return (
        case_dir,
        manifest,
        summary,
    )


def mirrored_breakpoints(
    nodes_half: np.ndarray,
) -> np.ndarray:
    nodes_half = np.asarray(
        nodes_half,
        dtype=float,
    )

    return np.unique(
        np.concatenate(
            (
                -nodes_half[::-1],
                nodes_half,
            )
        )
    )


def selected_one_mode_K3(
    K6: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(
        [0, 1, 2, 4],
        dtype=int,
    )

    K4 = np.asarray(
        K6,
        dtype=float,
    )[
        np.ix_(
            indices,
            indices,
        )
    ]

    K00 = float(
        K4[0, 0]
    )

    K0t = K4[
        0,
        1:,
    ]

    Ktt = K4[
        1:,
        1:,
    ]

    K3 = (
        Ktt
        - np.outer(
            K0t,
            K0t,
        )
        / K00
    )

    return 0.5 * (
        K3 + K3.T
    )


def weighted_shape_fit(
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> dict:
    reference = np.asarray(
        reference,
        dtype=float,
    )

    candidate = np.asarray(
        candidate,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    denominator = float(
        np.sum(
            weights
            * candidate
            * candidate
        )
    )

    if denominator <= 0.0:
        return {
            "relative_rmse": float(
                "nan"
            ),
            "correlation": float(
                "nan"
            ),
        }

    amplitude = float(
        np.sum(
            weights
            * candidate
            * reference
        )
        / denominator
    )

    fitted = (
        amplitude * candidate
    )

    reference_norm = math.sqrt(
        float(
            np.sum(
                weights
                * reference
                * reference
            )
        )
    )

    relative_rmse = (
        math.sqrt(
            float(
                np.sum(
                    weights
                    * (
                        reference
                        - fitted
                    ) ** 2
                )
            )
        )
        / reference_norm
        if reference_norm > 0.0
        else float("nan")
    )

    total_weight = float(
        np.sum(weights)
    )

    reference_mean = float(
        np.sum(
            weights
            * reference
        )
        / total_weight
    )

    fitted_mean = float(
        np.sum(
            weights
            * fitted
        )
        / total_weight
    )

    reference_centered = (
        reference
        - reference_mean
    )

    fitted_centered = (
        fitted
        - fitted_mean
    )

    correlation_denominator = math.sqrt(
        float(
            np.sum(
                weights
                * reference_centered ** 2
            )
            * np.sum(
                weights
                * fitted_centered ** 2
            )
        )
    )

    correlation = (
        float(
            np.sum(
                weights
                * reference_centered
                * fitted_centered
            )
            / correlation_denominator
        )
        if correlation_denominator > 0.0
        else float("nan")
    )

    return {
        "relative_rmse": float(
            relative_rmse
        ),
        "correlation": float(
            correlation
        ),
    }


def read_bulk_shear_profile(
    case_dir: Path,
    equilibrium_primary,
) -> dict:
    path = (
        case_dir
        / "post"
        / "through_thickness_profiles.csv"
    )

    if not path.is_file():
        return {}

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            if (
                row.get("station")
                != "left_shear_mid"
            ):
                continue

            if row.get(
                "region"
            ) not in {
                "FACE_BOTTOM",
                "CORE",
                "FACE_TOP",
            }:
                continue

            try:
                rows.append(
                    {
                        "z": float(
                            row["z_ip_mm"]
                        ),
                        "gamma": float(
                            row[
                                "gamma13_engineering"
                            ]
                        ),
                        "tau": float(
                            row[
                                "S13_MPa"
                            ]
                        ),
                        "weight": float(
                            row[
                                "volume_mm3"
                            ]
                        ),
                    }
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

    if not rows:
        return {}

    rows.sort(
        key=lambda item: item["z"]
    )

    z = np.asarray(
        [
            row["z"]
            for row in rows
        ],
        dtype=float,
    )

    weights = np.asarray(
        [
            row["weight"]
            for row in rows
        ],
        dtype=float,
    )

    gamma_fe = np.asarray(
        [
            row["gamma"]
            for row in rows
        ],
        dtype=float,
    )

    tau_fe = np.asarray(
        [
            row["tau"]
            for row in rows
        ],
        dtype=float,
    )

    gamma_eq = np.asarray(
        [
            equilibrium_primary.df(
                value
            )
            for value in z
        ],
        dtype=float,
    )

    tau_eq = np.asarray(
        [
            equilibrium_primary.tau_per_unit_shear(
                value
            )
            for value in z
        ],
        dtype=float,
    )

    gamma_fit = weighted_shape_fit(
        gamma_fe,
        gamma_eq,
        weights,
    )

    tau_fit = weighted_shape_fit(
        tau_fe,
        tau_eq,
        weights,
    )

    return {
        "gamma_profile_RMSE_percent": (
            100.0
            * gamma_fit[
                "relative_rmse"
            ]
        ),
        "gamma_profile_correlation": (
            gamma_fit[
                "correlation"
            ]
        ),
        "tau_profile_RMSE_percent": (
            100.0
            * tau_fit[
                "relative_rmse"
            ]
        ),
        "tau_profile_correlation": (
            tau_fit[
                "correlation"
            ]
        ),
    }


def build_host_result(
    case_dir: Path,
    manifest: dict,
    summary: dict,
    *,
    core_mode_elements: int,
    face_mode_elements: int,
    max_segment_mm: float,
) -> dict:
    geometry = manifest[
        "geometry"
    ]

    constitutive = manifest[
        "constitutive"
    ]

    Ccore3 = np.asarray(
        constitutive[
            "C_hom_model_MPa"
        ],
        dtype=float,
    )

    face = constitutive[
        "face_material"
    ]

    Cface3 = isotropic_stiffness_3d(
        E=float(
            face["E_MPa"]
        ),
        nu=float(
            face["nu"]
        ),
    )

    Ccore = free_stress_beam_stiffness(
        Ccore3,
        axis="x",
    )

    Cface = free_stress_beam_stiffness(
        Cface3,
        axis="x",
    )

    z_bottom = float(
        geometry["z_bottom"]
    )

    z_core_bottom = float(
        geometry[
            "z_core_bottom"
        ]
    )

    z_core_top = float(
        geometry[
            "z_core_top"
        ]
    )

    z_top = float(
        geometry["z_top"]
    )

    width = float(
        geometry["width"]
    )

    if (
        abs(
            z_core_bottom
            + z_core_top
        )
        > 1.0e-9
        or abs(
            z_bottom
            + z_top
        )
        > 1.0e-9
    ):
        raise RuntimeError(
            "Transfer analyzer currently expects a symmetric sandwich."
        )

    full_layers = (
        BeamLayer2D(
            "face_bottom",
            z_bottom,
            z_core_bottom,
            Cface,
        ),
        BeamLayer2D(
            "core",
            z_core_bottom,
            z_core_top,
            Ccore,
        ),
        BeamLayer2D(
            "face_top",
            z_core_top,
            z_top,
            Cface,
        ),
    )

    half_layers = (
        BeamLayer2D(
            "core+",
            0.0,
            z_core_top,
            Ccore,
        ),
        BeamLayer2D(
            "face+",
            z_core_top,
            z_top,
            Cface,
        ),
    )

    f1 = derive_equilibrium_shear_mode(
        full_layers,
        width=width,
    )

    f2 = derive_variational_second_shear_mode(
        half_layers,
        f1.f,
        elements_per_layer=(
            int(
                core_mode_elements
            ),
            int(
                face_mode_elements
            ),
        ),
    )

    section = build_two_mode_beam_section(
        full_layers,
        width=width,
        mode1=WarpingMode(
            f1.f,
            f1.df,
        ),
        mode2=WarpingMode(
            f2.f,
            f2.df,
        ),
        quadrature_order=8,
        integration_breakpoints=mirrored_breakpoints(
            f2.nodes_half
        ),
    )

    K3 = selected_one_mode_K3(
        section.K6
    )

    load_total = float(
        manifest[
            "fixture"
        ][
            "total_applied_load_N"
        ]
    )

    applied_forces = (
        PointForce(
            float(
                geometry[
                    "load_left"
                ]
            ),
            0.5 * load_total,
        ),
        PointForce(
            float(
                geometry[
                    "load_right"
                ]
            ),
            0.5 * load_total,
        ),
    )

    solution_one = (
        solve_simply_supported_point_loads(
            K3,
            length=float(
                geometry[
                    "length"
                ]
            ),
            support_left=float(
                geometry[
                    "support_left"
                ]
            ),
            support_right=float(
                geometry[
                    "support_right"
                ]
            ),
            applied_forces=applied_forces,
        )
    )

    solution_two = (
        solve_simply_supported_two_mode_point_loads(
            section.K5,
            length=float(
                geometry[
                    "length"
                ]
            ),
            support_left=float(
                geometry[
                    "support_left"
                ]
            ),
            support_right=float(
                geometry[
                    "support_right"
                ]
            ),
            applied_forces=applied_forces,
            max_segment_length=float(
                max_segment_mm
            ),
        )
    )

    x_mid = float(
        geometry["midspan"]
    )

    w_one = abs(
        float(
            solution_one.state(
                x_mid
            )[0]
        )
    )

    w_two = abs(
        float(
            solution_two.state(
                x_mid
            )[0]
        )
    )

    w_fe = abs(
        float(
            summary[
                "midspan"
            ][
                "U3_midplane_avg_mm"
            ]
        )
    )

    spectrum = boundary_layer_spectrum(
        section.K5
    )

    eta15 = float(
        abs(
            Ccore[0, 1]
        )
        / math.sqrt(
            Ccore[0, 0]
            * Ccore[1, 1]
        )
    )

    result = {
        "case_id": str(
            manifest[
                "case"
            ][
                "case_id"
            ]
        ),
        "case_hash": str(
            manifest[
                "case"
            ][
                "case_hash"
            ]
        ),
        "case_path": str(
            case_dir
        ),
        "FE_abs_w_mid_mm": w_fe,
        "HOST_equilibrium_1mode_abs_w_mid_mm": (
            w_one
        ),
        "HOST_equilibrium_2mode_abs_w_mid_mm": (
            w_two
        ),
        "one_mode_error_percent": (
            100.0
            * (
                w_one - w_fe
            )
            / w_fe
        ),
        "two_mode_error_percent": (
            100.0
            * (
                w_two - w_fe
            )
            / w_fe
        ),
        "Cb11_MPa": float(
            Ccore[0, 0]
        ),
        "Cb15_MPa": float(
            Ccore[0, 1]
        ),
        "Cb55_MPa": float(
            Ccore[1, 1]
        ),
        "eta15": eta15,
        "S_eq_N": float(
            f1.equivalent_shear_stiffness
        ),
        "K_eq": float(
            f1.shear_correction_factor
        ),
        "unit_V_check": float(
            f1.unit_shear_resultant_check
        ),
        "lambda2_per_mm2": float(
            f2.eigenvalue
        ),
        "ell_section2_mm": float(
            f2.characteristic_length
        ),
        "alpha_local": float(
            local_shear_condensation_ratio(
                section.K5
            )
        ),
        "ell_slow_BL_mm": float(
            spectrum[
                "stable_modes"
            ][0][
                "decay_length"
            ]
        ),
        "ell_fast_BL_mm": float(
            spectrum[
                "stable_modes"
            ][-1][
                "decay_length"
            ]
        ),
    }

    result.update(
        read_bulk_shear_profile(
            case_dir,
            f1,
        )
    )

    return result


def main() -> int:
    args = parse_args()

    run_root = args.run_root.resolve()

    output_dir = (
        args.output_dir
    )

    if not output_dir.is_absolute():
        output_dir = (
            Path.cwd()
            / output_dir
        )

    output_dir = output_dir.resolve()

    if not run_root.is_dir():
        raise FileNotFoundError(
            run_root
        )

    topology_order = {
        str(topology): index
        for index, topology
        in enumerate(
            args.topologies
        )
    }

    rows = []

    for topology in args.topologies:
        for n_shear in args.n_shear:
            case_dir, manifest, summary = discover_case(
                run_root,
                topology=topology,
                n_shear=n_shear,
                relative_density=float(
                    args.relative_density
                ),
                campaign_type=str(
                    args.campaign_type
                ),
                load_case=str(
                    args.load_case
                ),
                state=str(
                    args.state
                ),
                element_type=str(
                    args.element_type
                ),
                ne_cell=int(
                    args.ne_cell
                ),
                face_ne_z=int(
                    args.face_ne_z
                ),
                n_core_height=(
                    None
                    if args.n_core_height is None
                    else int(
                        args.n_core_height
                    )
                ),
                face_bottom_mm=(
                    None
                    if args.face_bottom_mm is None
                    else float(
                        args.face_bottom_mm
                    )
                ),
                face_top_mm=(
                    None
                    if args.face_top_mm is None
                    else float(
                        args.face_top_mm
                    )
                ),
                fixture_mode=str(
                    args.fixture_mode
                ),
                domain_mode=str(
                    args.domain_mode
                ),
                model_implementation_revision=int(
                    args.model_implementation_revision
                ),
            )

            query = manifest[
                "database_query"
            ]

            geometry = manifest[
                "geometry"
            ]

            result = build_host_result(
                case_dir,
                manifest,
                summary,
                core_mode_elements=int(
                    args.core_mode_elements
                ),
                face_mode_elements=int(
                    args.face_mode_elements
                ),
                max_segment_mm=float(
                    args.max_segment_mm
                ),
            )

            result.update(
                {
                    "topology": str(
                        query[
                            "topology"
                        ]
                    ),
                    "rd_target": float(
                        query[
                            "relative_density_target"
                        ]
                    ),
                    "rd_actual": float(
                        query[
                            "relative_density_actual"
                        ]
                    ),
                    "n_shear": int(
                        geometry[
                            "n_shear"
                        ]
                    ),
                    "n_core_height": int(
                        geometry[
                            "n_height"
                        ]
                    ),
                    "core_height_mm": float(
                        geometry[
                            "core_height"
                        ]
                    ),
                    "face_bottom_mm": float(
                        geometry[
                            "face_bottom"
                        ]
                    ),
                    "face_top_mm": float(
                        geometry[
                            "face_top"
                        ]
                    ),
                    "total_height_mm": float(
                        geometry[
                            "total_height"
                        ]
                    ),
                    "face_to_core_thickness_ratio": float(
                        0.5
                        * (
                            geometry[
                                "face_bottom"
                            ]
                            + geometry[
                                "face_top"
                            ]
                        )
                        / (
                            geometry[
                                "z_core_top"
                            ]
                            - geometry[
                                "z_core_bottom"
                            ]
                        )
                    ),
                    "a_over_h": float(
                        geometry[
                            "shear_span"
                        ]
                        / geometry[
                            "total_height"
                        ]
                    ),
                    "reflection_error_x": float(
                        manifest[
                            "domain"
                        ].get(
                            "constitutive_reflection_relative_error_x",
                            float("nan"),
                        )
                    ),
                    "reflection_error_y": float(
                        manifest[
                            "domain"
                        ].get(
                            "constitutive_reflection_relative_error_y",
                            float("nan"),
                        )
                    ),
                }
            )

            rows.append(
                result
            )

    rows.sort(
        key=lambda row: (
            topology_order.get(
                row["topology"],
                999,
            ),
            row["n_shear"],
        )
    )

    errors_one = np.asarray(
        [
            row[
                "one_mode_error_percent"
            ]
            for row in rows
        ],
        dtype=float,
    )

    errors_two = np.asarray(
        [
            row[
                "two_mode_error_percent"
            ]
            for row in rows
        ],
        dtype=float,
    )

    summary = {
        "number_of_cases": len(rows),
        "RMS_one_mode_percent": float(
            np.sqrt(
                np.mean(
                    errors_one ** 2
                )
            )
        ),
        "RMS_two_mode_percent": float(
            np.sqrt(
                np.mean(
                    errors_two ** 2
                )
            )
        ),
        "max_abs_one_mode_percent": float(
            np.max(
                np.abs(
                    errors_one
                )
            )
        ),
        "max_abs_two_mode_percent": float(
            np.max(
                np.abs(
                    errors_two
                )
            )
        ),
        "max_eta15": float(
            max(
                row["eta15"]
                for row in rows
            )
        ),
        "max_reflection_error_x": float(
            max(
                row[
                    "reflection_error_x"
                ]
                for row in rows
            )
        ),
        "max_reflection_error_y": float(
            max(
                row[
                    "reflection_error_y"
                ]
                for row in rows
            )
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "topology_transfer.json"
    )

    csv_path = (
        output_dir
        / "topology_transfer.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema": "host_topology_transfer_v2",
                "criteria": {
                    "topologies": list(
                        args.topologies
                    ),
                    "n_shear": list(
                        args.n_shear
                    ),
                    "relative_density": float(
                        args.relative_density
                    ),
                    "campaign_type": str(
                        args.campaign_type
                    ),
                    "load_case": str(
                        args.load_case
                    ),
                    "state": str(
                        args.state
                    ),
                    "element_type": str(
                        args.element_type
                    ),
                    "ne_cell": int(
                        args.ne_cell
                    ),
                    "face_ne_z": int(
                        args.face_ne_z
                    ),
                    "n_core_height": (
                        None
                        if args.n_core_height is None
                        else int(
                            args.n_core_height
                        )
                    ),
                    "face_bottom_mm": (
                        None
                        if args.face_bottom_mm is None
                        else float(
                            args.face_bottom_mm
                        )
                    ),
                    "face_top_mm": (
                        None
                        if args.face_top_mm is None
                        else float(
                            args.face_top_mm
                        )
                    ),
                    "fixture_mode": str(
                        args.fixture_mode
                    ),
                    "domain_mode": str(
                        args.domain_mode
                    ),
                    "model_implementation_revision": int(
                        args.model_implementation_revision
                    ),
                },
                "summary": summary,
                "rows": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 158)
    print("HOST EQUILIBRIUM-ENRICHED TOPOLOGY TRANSFER")
    print("=" * 158)
    print(
        "{:>10s} {:>5s} {:>7s} {:>11s} {:>11s} {:>11s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
            "topology",
            "N_A",
            "a/h",
            "FE",
            "HOST-1",
            "HOST-2",
            "err1[%]",
            "err2[%]",
            "K_eq",
            "eta15",
            "ellBL",
        )
    )

    for row in rows:
        print(
            "{:>10s} {:5d} {:7.3f} {:11.8f} {:11.8f} {:11.8f} {:9.3f} {:9.3f} {:9.5f} {:9.3e} {:9.3f}".format(
                row["topology"],
                row["n_shear"],
                row["a_over_h"],
                row["FE_abs_w_mid_mm"],
                row[
                    "HOST_equilibrium_1mode_abs_w_mid_mm"
                ],
                row[
                    "HOST_equilibrium_2mode_abs_w_mid_mm"
                ],
                row[
                    "one_mode_error_percent"
                ],
                row[
                    "two_mode_error_percent"
                ],
                row["K_eq"],
                row["eta15"],
                row[
                    "ell_slow_BL_mm"
                ],
            )
        )

    print()
    print(
        "RMS error 1mode -> 2mode : {:.4f}% -> {:.4f}%".format(
            summary[
                "RMS_one_mode_percent"
            ],
            summary[
                "RMS_two_mode_percent"
            ],
        )
    )

    print(
        "Max error 1mode -> 2mode : {:.4f}% -> {:.4f}%".format(
            summary[
                "max_abs_one_mode_percent"
            ],
            summary[
                "max_abs_two_mode_percent"
            ],
        )
    )

    print(
        "Max normalized C15       : {:.6e}".format(
            summary[
                "max_eta15"
            ]
        )
    )

    print(
        "Max reflection x / y     : {:.6e} / {:.6e}".format(
            summary[
                "max_reflection_error_x"
            ],
            summary[
                "max_reflection_error_y"
            ],
        )
    )

    profile_rows = [
        row
        for row in rows
        if (
            "gamma_profile_RMSE_percent"
            in row
        )
    ]

    if profile_rows:
        print()
        print(
            "Equilibrium-primary FE profile RMSE [%]:"
        )

        for row in profile_rows:
            print(
                "  {:>10s} N_A={:<2d}  gamma={:.4f}  tau={:.4f}".format(
                    row[
                        "topology"
                    ],
                    row[
                        "n_shear"
                    ],
                    row[
                        "gamma_profile_RMSE_percent"
                    ],
                    row[
                        "tau_profile_RMSE_percent"
                    ],
                )
            )

    print()
    print("Wrote:")
    print(" ", json_path)
    print(" ", csv_path)
    print("=" * 158)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
