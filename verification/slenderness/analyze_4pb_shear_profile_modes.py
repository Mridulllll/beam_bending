#!/usr/bin/env python3
"""
Decompose the FE transverse-shear profiles into even/odd thickness parts and
test whether one additional through-thickness shear mode explains the part
that is missing from the current HOST mode.

Why this audit is needed
------------------------
The previous local profile projection found

    K_beta_FE / K_beta_HOST ~= 1.00002

for every slenderness, while the full FE gamma_xz(z) profile differs from
the single HOST mode f'_x(z) by about 26%.

Those two facts do NOT imply that HOST should simply rescale K_beta.  They
suggest that the stiffness of the retained HOST mode is already correct but
that the 3-D continuum may use additional through-thickness shear modes.

Before drawing that conclusion, however, the FE profile must be split into

    even part:  [g(z) + g(-z)] / 2
    odd part :  [g(z) - g(-z)] / 2

because the current HOST transverse-force shear mode is even, while
anisotropic normal/shear coupling under bending can generate odd S13/gamma13
content that integrates to zero section shear force.

After the parity split, the script:

1. projects the EVEN FE gamma13 profile onto the current HOST f'_x(z);
2. reports the remaining even-profile error and the odd-profile fraction;
3. collects the normalized even residual profiles from all six slendernesses;
4. performs an area-weighted SVD of those residuals;
5. reports how much residual energy is captured by ONE additional common
   through-thickness mode.

If one common residual mode collapses the profile error for all slendernesses,
that is evidence for enriching the HOST shear kinematics rather than changing
the constitutive shear modulus or introducing an empirical correction factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_N_SHEAR = (6, 9, 12, 18, 24, 30)
DEFAULT_PLOT_N_SHEAR = (6, 12, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(r"C:\AbaqusScratch\HOST\4pb_homogenized"),
    )

    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "verification/beam_reduction/"
            "plate_to_beam_audit.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_profile_modes"
        ),
    )

    parser.add_argument(
        "--station",
        default="left_shear_mid",
    )

    parser.add_argument(
        "--n-shear",
        default=",".join(str(v) for v in DEFAULT_N_SHEAR),
    )

    parser.add_argument(
        "--plot-n-shear",
        default=",".join(str(v) for v in DEFAULT_PLOT_N_SHEAR),
    )

    parser.add_argument("--element-type", default="C3D20R")
    parser.add_argument("--ne-cell", type=int, default=2)
    parser.add_argument("--face-ne-z", type=int, default=2)
    parser.add_argument("--fixture-mode", default="section_distributing")
    parser.add_argument("--domain-mode", default="quarter_xy")

    return parser.parse_args()


def parse_ints(text: str) -> Tuple[int, ...]:
    return tuple(
        int(token.strip())
        for token in str(text).split(",")
        if token.strip()
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fprime_x(
    z: np.ndarray,
    *,
    h_total: float,
    h_core: float,
    G_core: float,
    G_face: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)

    p = 1.0 - 4.0 * z * z / (float(h_total) ** 2)

    core_mask = np.abs(z) <= 0.5 * float(h_core) + 1.0e-12

    result = np.empty_like(z)
    result[core_mask] = p[core_mask]
    result[~core_mask] = (
        float(G_core)
        / float(G_face)
        * p[~core_mask]
    )

    return result


def physical_sample_width_mm(rows: Sequence[dict]) -> float:
    n_slab = int(float(rows[0]["x_slab_count"]))
    x_min = float(rows[0]["x_sampled_min_mm"])
    x_max = float(rows[0]["x_sampled_max_mm"])

    if n_slab < 2:
        raise ValueError(
            "Need x_slab_count >= 2 to infer the physical sample width."
        )

    return (
        (x_max - x_min)
        * float(n_slab)
        / float(n_slab - 1)
    )


def weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(
        float(
            np.sum(
                np.asarray(weights, dtype=float)
                * np.asarray(values, dtype=float) ** 2
            )
        )
    )


def weighted_relative_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    denominator = weighted_norm(observed, weights)

    if denominator <= 0.0:
        return float("nan")

    return (
        weighted_norm(
            np.asarray(observed) - np.asarray(predicted),
            weights,
        )
        / denominator
    )


def weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    total = float(np.sum(w))
    if total <= 0.0:
        return float("nan")

    mx = float(np.sum(w * x) / total)
    my = float(np.sum(w * y) / total)

    dx = x - mx
    dy = y - my

    vx = float(np.sum(w * dx * dx) / total)
    vy = float(np.sum(w * dy * dy) / total)

    if vx <= 0.0 or vy <= 0.0:
        return float("nan")

    covariance = float(np.sum(w * dx * dy) / total)

    return covariance / math.sqrt(vx * vy)


def parity_decomposition(
    z: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float = 1.0e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return even, odd and pair-averaged weights on the original z ordering.
    """
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n = z.size

    even = np.empty(n, dtype=float)
    odd = np.empty(n, dtype=float)
    paired_weight = np.empty(n, dtype=float)

    used = set()

    for i in range(n):
        if i in used:
            continue

        j = int(
            np.argmin(
                np.abs(
                    z + z[i]
                )
            )
        )

        mismatch = abs(
            z[j] + z[i]
        )

        if mismatch > tolerance:
            raise RuntimeError(
                "Could not find symmetric z partner for z={:.12g}; "
                "nearest mismatch={:.3e}.".format(
                    z[i],
                    mismatch,
                )
            )

        if i == j:
            even[i] = values[i]
            odd[i] = 0.0
            paired_weight[i] = weights[i]
            used.add(i)
            continue

        average = 0.5 * (
            values[i] + values[j]
        )

        antisymmetric_i = 0.5 * (
            values[i] - values[j]
        )

        average_weight = 0.5 * (
            weights[i] + weights[j]
        )

        even[i] = average
        even[j] = average

        odd[i] = antisymmetric_i
        odd[j] = -antisymmetric_i

        paired_weight[i] = average_weight
        paired_weight[j] = average_weight

        used.add(i)
        used.add(j)

    return even, odd, paired_weight


def find_cases(
    run_root: Path,
    n_shear_values: Sequence[int],
    *,
    element_type: str,
    ne_cell: int,
    face_ne_z: int,
    fixture_mode: str,
    domain_mode: str,
) -> Dict[int, dict]:
    wanted = set(int(v) for v in n_shear_values)
    candidates: Dict[int, List[dict]] = {
        value: []
        for value in wanted
    }

    for case_dir in run_root.iterdir():
        if not case_dir.is_dir():
            continue

        manifest_path = case_dir / "manifest.json"
        profile_path = case_dir / "post" / "through_thickness_profiles.csv"

        if not manifest_path.is_file() or not profile_path.is_file():
            continue

        try:
            manifest = load_json(manifest_path)
            n_shear = int(manifest["geometry"]["n_shear"])
        except Exception:
            continue

        if n_shear not in wanted:
            continue

        mesh = manifest.get("mesh", {})
        fixture = manifest.get("fixture", {})
        domain = manifest.get("domain", {})

        if str(mesh.get("element_type", "")).upper() != element_type.upper():
            continue
        if int(mesh.get("core_elements_per_cell_edge", -1)) != int(ne_cell):
            continue
        if int(mesh.get("face_elements_through_thickness", -1)) != int(face_ne_z):
            continue
        if str(fixture.get("mode", "")).lower() != fixture_mode.lower():
            continue
        if str(domain.get("mode", "")).lower() != domain_mode.lower():
            continue

        candidates[n_shear].append(
            {
                "case_dir": case_dir,
                "manifest": manifest,
                "profile_path": profile_path,
                "mtime": profile_path.stat().st_mtime,
            }
        )

    selected = {}

    for n_shear, items in candidates.items():
        if not items:
            continue

        items.sort(key=lambda item: item["mtime"], reverse=True)
        selected[n_shear] = items[0]

    return selected


def station_rows(profile_path: Path, station: str) -> List[dict]:
    rows = [
        row
        for row in read_csv(profile_path)
        if str(row.get("station", "")) == str(station)
    ]

    if not rows:
        raise RuntimeError(
            "Station {!r} not found in {}.".format(
                station,
                profile_path,
            )
        )

    rows.sort(key=lambda row: float(row["z_ip_mm"]))

    return rows


def main() -> int:
    args = parse_args()

    run_root = args.run_root.resolve()
    audit_path = args.audit.resolve()
    output_dir = args.output_dir.resolve()

    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)

    audit = load_json(audit_path)

    n_shear_values = parse_ints(args.n_shear)
    plot_n_shear = set(parse_ints(args.plot_n_shear))

    selected = find_cases(
        run_root,
        n_shear_values,
        element_type=args.element_type,
        ne_cell=args.ne_cell,
        face_ne_z=args.face_ne_z,
        fixture_mode=args.fixture_mode,
        domain_mode=args.domain_mode,
    )

    missing = sorted(set(n_shear_values) - set(selected))

    if missing:
        raise RuntimeError(
            "Missing profile cases for N_SHEAR={}".format(
                missing
            )
        )

    G_core = float(
        audit["shear_moduli"]["G_xz_core_MPa"]
    )

    G_face = float(
        audit["shear_moduli"]["G_xz_face_MPa"]
    )

    result_rows = []
    profile_data = {}

    reference_z = None
    reference_weights = None
    normalized_even_profiles = []
    normalized_even_residuals = []

    for n_shear in n_shear_values:
        item = selected[n_shear]
        manifest = item["manifest"]
        geometry = manifest["geometry"]

        rows = station_rows(
            item["profile_path"],
            args.station,
        )

        z = np.asarray(
            [float(row["z_ip_mm"]) for row in rows],
            dtype=float,
        )

        gamma = np.asarray(
            [float(row["gamma13_engineering"]) for row in rows],
            dtype=float,
        )

        tau = np.asarray(
            [float(row["S13_MPa"]) for row in rows],
            dtype=float,
        )

        volume = np.asarray(
            [float(row["volume_mm3"]) for row in rows],
            dtype=float,
        )

        dx_sample = physical_sample_width_mm(rows)

        area_half = (
            volume / dx_sample
        )

        gamma_even, gamma_odd, weights = parity_decomposition(
            z,
            gamma,
            area_half,
        )

        tau_even, tau_odd, tau_weights = parity_decomposition(
            z,
            tau,
            area_half,
        )

        if not np.allclose(
            weights,
            tau_weights,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Gamma/tau parity weights differ unexpectedly."
            )

        h_core = float(
            geometry.get(
                "core_height",
                geometry.get("core_height_mm"),
            )
        )

        face_bottom = float(
            geometry.get(
                "face_bottom",
                geometry.get("face_bottom_mm"),
            )
        )

        face_top = float(
            geometry.get(
                "face_top",
                geometry.get("face_top_mm"),
            )
        )

        h_total = (
            h_core
            + face_bottom
            + face_top
        )

        fp = fprime_x(
            z,
            h_total=h_total,
            h_core=h_core,
            G_core=G_core,
            G_face=G_face,
        )

        denominator = float(
            np.sum(
                weights
                * fp
                * fp
            )
        )

        beta_even = float(
            np.sum(
                weights
                * fp
                * gamma_even
            )
            / denominator
        )

        gamma_host_even = (
            beta_even
            * fp
        )

        gamma_even_rmse = weighted_relative_rmse(
            gamma_even,
            gamma_host_even,
            weights,
        )

        gamma_even_corr = weighted_correlation(
            gamma_even,
            gamma_host_even,
            weights,
        )

        gamma_total_norm = weighted_norm(
            gamma,
            weights,
        )

        gamma_even_norm = weighted_norm(
            gamma_even,
            weights,
        )

        gamma_odd_norm = weighted_norm(
            gamma_odd,
            weights,
        )

        tau_total_norm = weighted_norm(
            tau,
            weights,
        )

        tau_even_norm = weighted_norm(
            tau_even,
            weights,
        )

        tau_odd_norm = weighted_norm(
            tau_odd,
            weights,
        )

        gamma_odd_fraction = (
            gamma_odd_norm
            / gamma_total_norm
        )

        tau_odd_fraction = (
            tau_odd_norm
            / tau_total_norm
        )

        gamma_even_fraction = (
            gamma_even_norm
            / gamma_total_norm
        )

        tau_even_fraction = (
            tau_even_norm
            / tau_total_norm
        )

        normalized_even = (
            gamma_even / beta_even
        )

        normalized_residual = (
            normalized_even - fp
        )

        if reference_z is None:
            reference_z = z.copy()
            reference_weights = weights.copy()
        else:
            if not np.allclose(
                z,
                reference_z,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise RuntimeError(
                    "Through-thickness z grid differs between sweep cases."
                )

            if not np.allclose(
                weights,
                reference_weights,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "Through-thickness area weights differ between sweep cases."
                )

        normalized_even_profiles.append(
            normalized_even
        )

        normalized_even_residuals.append(
            normalized_residual
        )

        shear_span = float(
            geometry.get(
                "shear_span",
                geometry.get("shear_span_mm"),
            )
        )

        row_out = {
            "n_shear": int(n_shear),
            "a_over_h": shear_span / h_total,
            "beta_even_projection": beta_even,
            "gamma_even_shape_relative_rmse": gamma_even_rmse,
            "gamma_even_shape_weighted_correlation": gamma_even_corr,
            "gamma_even_norm_fraction_of_total": gamma_even_fraction,
            "gamma_odd_norm_fraction_of_total": gamma_odd_fraction,
            "tau_even_norm_fraction_of_total": tau_even_fraction,
            "tau_odd_norm_fraction_of_total": tau_odd_fraction,
            "case_hash": manifest.get("case", {}).get("case_hash"),
            "case_path": str(item["case_dir"]),
        }

        result_rows.append(row_out)

        profile_data[str(n_shear)] = {
            "z_mm": z.tolist(),
            "z_over_h": (z / h_total).tolist(),
            "area_half_mm2": weights.tolist(),
            "gamma_total": gamma.tolist(),
            "gamma_even": gamma_even.tolist(),
            "gamma_odd": gamma_odd.tolist(),
            "gamma_even_normalized": normalized_even.tolist(),
            "HOST_fprime": fp.tolist(),
            "even_residual_normalized": normalized_residual.tolist(),
            "tau_total_MPa": tau.tolist(),
            "tau_even_MPa": tau_even.tolist(),
            "tau_odd_MPa": tau_odd.tolist(),
        }

    result_rows.sort(key=lambda row: row["a_over_h"])

    G = np.vstack(
        normalized_even_profiles
    )

    R = np.vstack(
        normalized_even_residuals
    )

    weights = np.asarray(
        reference_weights,
        dtype=float,
    )

    sqrt_w = np.sqrt(weights)

    # Weighted SVD of the residual left after the current HOST shear mode.
    Rw = (
        R
        * sqrt_w[np.newaxis, :]
    )

    U, singular_values, Vh = np.linalg.svd(
        Rw,
        full_matrices=False,
    )

    singular_energy = singular_values ** 2
    total_residual_energy = float(
        np.sum(
            singular_energy
        )
    )

    if total_residual_energy > 0.0:
        explained = (
            singular_energy
            / total_residual_energy
        )
    else:
        explained = np.zeros_like(
            singular_energy
        )

    first_reconstruction_weighted = (
        np.outer(
            U[:, 0] * singular_values[0],
            Vh[0, :],
        )
    )

    first_reconstruction = (
        first_reconstruction_weighted
        / sqrt_w[np.newaxis, :]
    )

    first_mode = (
        Vh[0, :]
        / sqrt_w
    )

    # Normalize the additional mode to unit weighted L2 norm and choose a
    # deterministic sign.
    first_mode_norm = weighted_norm(
        first_mode,
        weights,
    )

    first_mode = (
        first_mode
        / first_mode_norm
    )

    central_index = int(
        np.argmin(
            np.abs(
                reference_z
            )
        )
    )

    if first_mode[central_index] < 0.0:
        first_mode *= -1.0
        first_reconstruction *= -1.0
        U[:, 0] *= -1.0

    two_mode_rmse = []
    residual_mode_coefficients = []

    for i in range(R.shape[0]):
        # Weighted projection coefficient on the normalized common residual mode.
        coefficient = float(
            np.sum(
                weights
                * R[i, :]
                * first_mode
            )
        )

        residual_mode_coefficients.append(
            coefficient
        )

        prediction = (
            audit["shear_moduli"]  # harmless access to retain explicit audit dependency
        )

        # Current HOST mode plus one additional common residual mode.
        host_plus_one = (
            G[i, :]
            - R[i, :]
            + coefficient * first_mode
        )

        rmse = weighted_relative_rmse(
            G[i, :],
            host_plus_one,
            weights,
        )

        two_mode_rmse.append(
            rmse
        )

    # Width-independent profile collapse metric.
    profile_mean = np.mean(
        G,
        axis=0,
    )

    profile_scatter = []

    for i in range(G.shape[0]):
        profile_scatter.append(
            weighted_relative_rmse(
                G[i, :],
                profile_mean,
                weights,
            )
        )

    for row, coefficient, rmse2, scatter in zip(
        result_rows,
        residual_mode_coefficients,
        two_mode_rmse,
        profile_scatter,
    ):
        row["common_residual_mode_coefficient"] = float(coefficient)
        row["HOST_plus_one_mode_relative_rmse"] = float(rmse2)
        row["normalized_even_profile_vs_mean_relative_rmse"] = float(scatter)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "shear_profile_modes.csv"
    )

    json_path = (
        output_dir
        / "shear_profile_modes.json"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            result_rows
        )

    report = {
        "schema": "host_4pb_shear_profile_modes_v1",
        "station": args.station,
        "interpretation_warning": (
            "The SVD mode is an FE-informed diagnostic residual mode, not a "
            "proposed final analytical warping function."
        ),
        "rows": result_rows,
        "residual_svd": {
            "singular_values": singular_values.tolist(),
            "explained_residual_energy_fraction": explained.tolist(),
            "first_mode_explained_fraction": float(explained[0]),
            "first_two_modes_explained_fraction": float(
                np.sum(explained[:2])
            ),
            "common_residual_mode_z_mm": reference_z.tolist(),
            "common_residual_mode": first_mode.tolist(),
            "weighted_mean_two_mode_relative_rmse": float(
                np.mean(two_mode_rmse)
            ),
            "weighted_max_two_mode_relative_rmse": float(
                np.max(two_mode_rmse)
            ),
            "mean_normalized_profile_collapse_rmse": float(
                np.mean(profile_scatter)
            ),
            "max_normalized_profile_collapse_rmse": float(
                np.max(profile_scatter)
            ),
        },
        "profiles": profile_data,
    }

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for n_shear in n_shear_values:
            if n_shear not in plot_n_shear:
                continue

            profile = profile_data[str(n_shear)]

            ax.plot(
                profile["gamma_even_normalized"],
                profile["z_over_h"],
                marker="o",
                markersize=3,
                label="FE even N_A={}".format(n_shear),
            )

        reference_n = next(
            value
            for value in n_shear_values
            if value in plot_n_shear
        )

        reference_profile = profile_data[str(reference_n)]

        ax.plot(
            reference_profile["HOST_fprime"],
            reference_profile["z_over_h"],
            linewidth=2.0,
            label="HOST f'_x",
        )

        ax.set_xlabel(r"normalized even $\gamma_{xz}$")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "gamma13_even_vs_HOST.png"
        )
        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(fig)
        plot_paths.append(
            str(path)
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        h_total_reference = (
            float(
                selected[n_shear_values[0]]["manifest"]["geometry"].get(
                    "core_height",
                    selected[n_shear_values[0]]["manifest"]["geometry"].get(
                        "core_height_mm"
                    ),
                )
            )
            + float(
                selected[n_shear_values[0]]["manifest"]["geometry"].get(
                    "face_bottom",
                    selected[n_shear_values[0]]["manifest"]["geometry"].get(
                        "face_bottom_mm"
                    ),
                )
            )
            + float(
                selected[n_shear_values[0]]["manifest"]["geometry"].get(
                    "face_top",
                    selected[n_shear_values[0]]["manifest"]["geometry"].get(
                        "face_top_mm"
                    ),
                )
            )
        )

        ax.plot(
            first_mode,
            reference_z / h_total_reference,
            marker="o",
            markersize=3,
        )

        ax.set_xlabel("first common residual shear mode")
        ax.set_ylabel("z/h")
        ax.grid(True)
        fig.tight_layout()

        path = (
            output_dir
            / "first_common_residual_mode.png"
        )
        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(fig)
        plot_paths.append(
            str(path)
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        a_over_h = np.asarray(
            [
                row["a_over_h"]
                for row in result_rows
            ],
            dtype=float,
        )

        odd_gamma = np.asarray(
            [
                row["gamma_odd_norm_fraction_of_total"]
                for row in result_rows
            ],
            dtype=float,
        )

        one_mode_error = np.asarray(
            [
                row["gamma_even_shape_relative_rmse"]
                for row in result_rows
            ],
            dtype=float,
        )

        two_mode_error_array = np.asarray(
            [
                row["HOST_plus_one_mode_relative_rmse"]
                for row in result_rows
            ],
            dtype=float,
        )

        ax.plot(
            a_over_h,
            100.0 * odd_gamma,
            marker="o",
            label="odd gamma fraction",
        )

        ax.plot(
            a_over_h,
            100.0 * one_mode_error,
            marker="s",
            label="HOST one-mode error",
        )

        ax.plot(
            a_over_h,
            100.0 * two_mode_error_array,
            marker="^",
            label="HOST + one residual mode",
        )

        ax.set_xlabel("a/h")
        ax.set_ylabel("[%]")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "parity_and_mode_errors.png"
        )
        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(fig)
        plot_paths.append(
            str(path)
        )

    except Exception as exc:
        plot_paths.append(
            "Plotting skipped: {}".format(exc)
        )

    report["plots"] = plot_paths

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

    print()
    print("=" * 126)
    print("HOST 4PB TRANSVERSE-SHEAR PARITY / RESIDUAL-MODE AUDIT")
    print("=" * 126)
    print("Station :", args.station)
    print()
    print(
        "{:>6s} {:>8s} {:>11s} {:>11s} {:>11s} {:>11s} {:>12s}".format(
            "N_A",
            "a/h",
            "odd gamma",
            "odd tau",
            "even RMSE",
            "even corr",
            "+1mode RMSE",
        )
    )

    for row in result_rows:
        print(
            "{:6d} {:8.3f} {:10.3%} {:10.3%} {:10.3%} {:11.6f} {:11.3%}".format(
                int(row["n_shear"]),
                float(row["a_over_h"]),
                float(row["gamma_odd_norm_fraction_of_total"]),
                float(row["tau_odd_norm_fraction_of_total"]),
                float(row["gamma_even_shape_relative_rmse"]),
                float(row["gamma_even_shape_weighted_correlation"]),
                float(row["HOST_plus_one_mode_relative_rmse"]),
            )
        )

    print()
    print("Residual SVD:")
    print(
        "  first mode explained fraction      : {:.6%}".format(
            float(explained[0])
        )
    )
    print(
        "  first two modes explained fraction : {:.6%}".format(
            float(np.sum(explained[:2]))
        )
    )
    print(
        "  mean HOST+1mode profile RMSE       : {:.6%}".format(
            float(np.mean(two_mode_rmse))
        )
    )
    print(
        "  max HOST+1mode profile RMSE        : {:.6%}".format(
            float(np.max(two_mode_rmse))
        )
    )
    print(
        "  mean normalized-profile scatter    : {:.6%}".format(
            float(np.mean(profile_scatter))
        )
    )
    print(
        "  max normalized-profile scatter     : {:.6%}".format(
            float(np.max(profile_scatter))
        )
    )
    print()
    print("Wrote:")
    print(" ", csv_path)
    print(" ", json_path)
    for path in plot_paths:
        print(" ", path)
    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
