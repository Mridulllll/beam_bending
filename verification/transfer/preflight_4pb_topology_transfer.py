#!/usr/bin/env python3
"""
Preflight the HOST 4PB topology-transfer campaign without running Abaqus.

The script uses the exact production stiffness-database query/rotation helper
and the current HOST equilibrium-primary + variational-secondary theory.

It verifies that Primitive, Gyroid, and Diamond RD12 initial records can each
be selected uniquely from the configured SQLite database and reports the
result-relevant material/theory quantities before any FE jobs are launched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["Primitive", "Gyroid", "Diamond"],
    )
    parser.add_argument(
        "--relative-density",
        default="0.12",
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
        "--core-height-mm",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--face-thickness-mm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--width-mm",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--face-E-MPa",
        type=float,
        default=70000.0,
    )
    parser.add_argument(
        "--face-nu",
        type=float,
        default=0.33,
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
        "--output",
        type=Path,
        default=Path(
            "verification/transfer/results/"
            "topology_rd12_initial/preflight.json"
        ),
    )

    return parser.parse_args()


def reflection_relative_error(
    C: np.ndarray,
    normal_axis: str,
) -> float:
    axis = str(normal_axis).strip().lower()

    if axis == "x":
        signs = np.asarray(
            [1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            dtype=float,
        )
    elif axis == "y":
        signs = np.asarray(
            [1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
            dtype=float,
        )
    elif axis == "z":
        signs = np.asarray(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            dtype=float,
        )
    else:
        raise ValueError(normal_axis)

    transformed = (
        signs[:, None]
        * C
        * signs[None, :]
    )

    return float(
        np.max(
            np.abs(
                C - transformed
            )
        )
        / max(
            float(np.max(np.abs(C))),
            1.0e-30,
        )
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


def main() -> int:
    args = parse_args()

    repo = args.repo.resolve()

    config_path = (
        repo / "config" / "paths.local.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        paths = json.load(handle)

    database_path = Path(
        paths["tpms_stiffness_database"]
    )

    materials_dir = (
        repo / "fem" / "abaqus" / "materials"
    )

    src_dir = repo / "src"

    for directory in (
        materials_dir,
        src_dir,
    ):
        text = str(directory)

        if text not in sys.path:
            sys.path.insert(0, text)

    from query_tpms_stiffness_database_rotated import (
        get_single_stiffness_record,
        oriented_stiffness_from_record,
        rotation_matrix_from_axis_map,
    )

    from host.beam_two_mode import (
        WarpingMode,
        boundary_layer_spectrum,
        build_two_mode_beam_section,
        local_shear_condensation_ratio,
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

    if not database_path.is_file():
        raise FileNotFoundError(
            "Configured stiffness database does not exist:\n{}".format(
                database_path
            )
        )

    # Exact production builder orientation:
    #   model X <- +UC x
    #   model Y <- -UC z
    #   model Z <- +UC y
    Q = rotation_matrix_from_axis_map(
        "+x",
        "-z",
        "+y",
    )

    h_core = float(
        args.core_height_mm
    )

    tf = float(
        args.face_thickness_mm
    )

    h = h_core + 2.0 * tf
    width = float(
        args.width_mm
    )

    Cface3 = isotropic_stiffness_3d(
        E=float(args.face_E_MPa),
        nu=float(args.face_nu),
    )

    Cface = free_stress_beam_stiffness(
        Cface3,
        axis="x",
    )

    rows = []

    for topology in args.topologies:
        row, convention = get_single_stiffness_record(
            database_path=database_path,
            campaign_type=str(
                args.campaign_type
            ),
            topology=str(topology),
            relative_density=args.relative_density,
            load_case=str(args.load_case),
            state=str(args.state),
        )

        oriented = oriented_stiffness_from_record(
            row=row,
            convention=convention,
            q_source_to_model=Q,
            symmetrize_before_rotation=False,
        )

        Cmodel = np.asarray(
            oriented["C_model"],
            dtype=float,
        )

        symmetry_error = float(
            np.max(
                np.abs(
                    Cmodel - Cmodel.T
                )
            )
            / max(
                float(
                    np.max(
                        np.abs(Cmodel)
                    )
                ),
                1.0,
            )
        )

        # Match the builder: reject material asymmetry first, then enforce exact
        # major symmetry only as a negligible numerical cleanup.
        if symmetry_error > 1.0e-10:
            raise RuntimeError(
                "{}: rotated stiffness symmetry error is {:.6e}.".format(
                    topology,
                    symmetry_error,
                )
            )

        Cmodel = 0.5 * (
            Cmodel + Cmodel.T
        )

        Ccore = free_stress_beam_stiffness(
            Cmodel,
            axis="x",
        )

        eta15 = float(
            abs(Ccore[0, 1])
            / math.sqrt(
                Ccore[0, 0]
                * Ccore[1, 1]
            )
        )

        full_layers = (
            BeamLayer2D(
                "face_bottom",
                -0.5 * h,
                -0.5 * h_core,
                Cface,
            ),
            BeamLayer2D(
                "core",
                -0.5 * h_core,
                +0.5 * h_core,
                Ccore,
            ),
            BeamLayer2D(
                "face_top",
                +0.5 * h_core,
                +0.5 * h,
                Cface,
            ),
        )

        half_layers = (
            BeamLayer2D(
                "core+",
                0.0,
                0.5 * h_core,
                Ccore,
            ),
            BeamLayer2D(
                "face+",
                0.5 * h_core,
                0.5 * h,
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
                    args.core_mode_elements
                ),
                int(
                    args.face_mode_elements
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

        spectrum = boundary_layer_spectrum(
            section.K5
        )

        constants = oriented[
            "engineering_constants_model"
        ]

        result = {
            "topology": str(
                row["topology"]
            ),
            "campaign_type": str(
                row["campaign_type"]
            ),
            "campaign_name": str(
                row["campaign_name"]
            ),
            "eval_id": str(
                row["eval_id"]
            ),
            "rd_target": float(
                row["rd_target_fraction"]
            ),
            "rd_actual": float(
                row["rd_actual_fraction"]
            ),
            "load_case": str(
                row["load_case"]
            ),
            "target_metric": str(
                row["target_metric_name"]
            ),
            "iteration": int(
                row["iteration"]
            ),
            "Ex_MPa": float(
                constants["Ex"]
            ),
            "Ez_MPa": float(
                constants["Ez"]
            ),
            "Gxz_MPa": float(
                constants["Gxz"]
            ),
            "reflection_error_x": (
                reflection_relative_error(
                    Cmodel,
                    "x",
                )
            ),
            "reflection_error_y": (
                reflection_relative_error(
                    Cmodel,
                    "y",
                )
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

        rows.append(result)

    output = args.output

    if not output.is_absolute():
        output = repo / output

    output = output.resolve()
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema": "host_topology_transfer_preflight_v1",
        "database_path": str(
            database_path
        ),
        "orientation": {
            "model_x_from_uc": "+x",
            "model_y_from_uc": "-z",
            "model_z_from_uc": "+y",
        },
        "rows": rows,
    }

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )
        handle.write("\n")

    print()
    print("=" * 154)
    print("HOST 4PB TOPOLOGY-TRANSFER PREFLIGHT")
    print("=" * 154)
    print(
        "Database: {}".format(
            database_path
        )
    )
    print()
    print(
        "{:>10s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            "topology",
            "Ex",
            "Ez",
            "Gxz",
            "refl-x",
            "refl-y",
            "eta15",
            "K_eq",
            "ell2",
            "ellBL",
        )
    )

    for row in rows:
        print(
            "{:>10s} {:9.2f} {:9.2f} {:9.2f} {:10.3e} {:10.3e} {:10.3e} {:10.6f} {:10.3f} {:10.3f}".format(
                row["topology"],
                row["Ex_MPa"],
                row["Ez_MPa"],
                row["Gxz_MPa"],
                row["reflection_error_x"],
                row["reflection_error_y"],
                row["eta15"],
                row["K_eq"],
                row["ell_section2_mm"],
                row["ell_slow_BL_mm"],
            )
        )

    print()
    print("Selected rows:")
    for row in rows:
        print(
            "  {:>10s}: {} | it={} | RD={:.9f} | target={}".format(
                row["topology"],
                row["eval_id"],
                row["iteration"],
                row["rd_actual"],
                row["target_metric"],
            )
        )

    print()
    print("Wrote:")
    print(" ", output)
    print("=" * 154)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
