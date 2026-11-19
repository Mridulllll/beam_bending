#!/usr/bin/env python3
"""
Compare 3-D FE through-thickness transverse-shear profiles with the current
HOST shear mode.

No Abaqus execution is required.  The script consumes the existing
post/through_thickness_profiles.csv files from the deep-to-slender 4PB sweep.

Primary questions
-----------------
1. Does the FE gamma_xz(z) shape collapse onto the current HOST f'_x(z)?
2. Does the FE stress field projected onto the HOST shear mode imply the same
   approximately constant shear-stiffness ratio found from the global
   deflection fit?
3. Is the mismatch predominantly amplitude/stiffness, or is the shape itself
   wrong?

Current HOST shear mode
-----------------------
The verified beam audit uses

    gamma_xz(z) = f'_x(z) * beta_x

with

    p(z) = 1 - 4 z^2 / h^2

and

    f'_x = p                            in the core
    f'_x = (G_xz_core/G_xz_face) * p  in the faces.

The free-stress 3-D -> beam constitutive reduction is reconstructed from the
same canonical C_hom manifest used by the reduction audit.

FE section weights
------------------
Each profile row contains the sum of integration-point volume represented by
that z-bin over the sampled axial slab and the computational half width.
Dividing by the sampled axial slab width therefore gives half-section area
weights.  The factor two for quarter_xy reconstructs the full physical width.

For two sampled x slabs with centroid locations x_min and x_max, the physical
sample width is

    dx = (x_max - x_min) * n_slab/(n_slab - 1).

This reproduces 10 mm for the present ne2 profiles, whose two slab centroids
are 5 mm apart.

Projected FE shear stiffness
----------------------------
A weighted least-squares FE generalized shear amplitude is

    beta_FE = argmin_beta integral [gamma_FE - beta f']^2 dA.

The FE stress projection onto the HOST shear mode is

    Q_beta_FE = integral tau_FE f' dA.

The diagnostic modal stiffness is then

    K_beta_FE = Q_beta_FE / beta_FE.

This is compared with the verified HOST condensed K3[2,2].  Because the
section is symmetric and f' is even, odd bending/shear coupling contributions
are strongly suppressed by this projection.

This is a diagnostic projection, not an exact variational identification of
a new theory parameter.
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
        "--shear-fit",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_compliance/"
            "shear_compliance_audit.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_profiles"
        ),
    )

    parser.add_argument(
        "--n-shear",
        default=",".join(str(v) for v in DEFAULT_N_SHEAR),
    )

    parser.add_argument(
        "--plot-n-shear",
        default=",".join(str(v) for v in DEFAULT_PLOT_N_SHEAR),
    )

    parser.add_argument("--station", default="left_shear_mid")

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


def isotropic_stiffness_3d(E: float, nu: float) -> np.ndarray:
    E = float(E)
    nu = float(nu)

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    C = np.zeros((6, 6), dtype=float)
    C[:3, :3] = lam
    C[0, 0] = lam + 2.0 * mu
    C[1, 1] = lam + 2.0 * mu
    C[2, 2] = lam + 2.0 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


def free_stress_x_beam_stiffness(C3: np.ndarray) -> np.ndarray:
    """
    Physical active components [eps_xx, gamma_xz], with
    sigma_yy=sigma_zz=tau_xy=tau_yz=0.
    """
    C3 = np.asarray(C3, dtype=float)

    active = (0, 4)
    inactive = (1, 2, 3, 5)

    Caa = C3[np.ix_(active, active)]
    Cai = C3[np.ix_(active, inactive)]
    Cia = C3[np.ix_(inactive, active)]
    Cii = C3[np.ix_(inactive, inactive)]

    return Caa - Cai @ np.linalg.solve(Cii, Cia)


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
    if not rows:
        raise ValueError("Empty profile station.")

    n_slab = int(float(rows[0]["x_slab_count"]))
    x_min = float(rows[0]["x_sampled_min_mm"])
    x_max = float(rows[0]["x_sampled_max_mm"])

    if n_slab <= 0:
        raise ValueError("x_slab_count must be positive.")

    if n_slab == 1:
        raise ValueError(
            "Cannot infer physical sample width from one slab centroid. "
            "Use a profile with x_slab_count >= 2."
        )

    centroid_spacing_span = x_max - x_min

    if centroid_spacing_span <= 0.0:
        raise ValueError("Invalid sampled x centroid range.")

    return (
        centroid_spacing_span
        * float(n_slab)
        / float(n_slab - 1)
    )


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

    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

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
            "Station {!r} not present in {}.".format(
                station,
                profile_path,
            )
        )

    rows.sort(key=lambda row: float(row["z_ip_mm"]))
    return rows


def weighted_relative_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    numerator = float(
        np.sum(
            weights * (observed - predicted) ** 2
        )
    )

    denominator = float(
        np.sum(
            weights * observed ** 2
        )
    )

    if denominator <= 0.0:
        return float("nan")

    return math.sqrt(numerator / denominator)


def weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    w = np.asarray(weights, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    total = float(np.sum(w))
    if total <= 0.0:
        return float("nan")

    mx = float(np.sum(w * x) / total)
    my = float(np.sum(w * y) / total)

    dx = x - mx
    dy = y - my

    covariance = float(np.sum(w * dx * dy) / total)
    vx = float(np.sum(w * dx * dx) / total)
    vy = float(np.sum(w * dy * dy) / total)

    if vx <= 0.0 or vy <= 0.0:
        return float("nan")

    return covariance / math.sqrt(vx * vy)


def main() -> int:
    args = parse_args()

    run_root = args.run_root.resolve()
    audit_path = args.audit.resolve()
    shear_fit_path = args.shear_fit.resolve()
    output_dir = args.output_dir.resolve()

    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)

    if not shear_fit_path.is_file():
        raise FileNotFoundError(shear_fit_path)

    audit = load_json(audit_path)
    shear_fit = load_json(shear_fit_path)

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
            "Missing through-thickness profiles for N_SHEAR={}".format(
                missing
            )
        )

    source_manifest = Path(audit["manifest"])
    if not source_manifest.is_file():
        raise FileNotFoundError(
            "Canonical constitutive manifest not found: {}".format(
                source_manifest
            )
        )

    source = load_json(source_manifest)

    C_core = np.asarray(
        source["constitutive"]["C_hom_model_MPa"],
        dtype=float,
    )

    face = audit["face_material"]
    C_face = isotropic_stiffness_3d(
        float(face["E_MPa"]),
        float(face["nu"]),
    )

    Cb_core = free_stress_x_beam_stiffness(C_core)
    Cb_face = free_stress_x_beam_stiffness(C_face)

    shear_moduli = audit["shear_moduli"]
    G_warp_core = float(shear_moduli["G_xz_core_MPa"])
    G_warp_face = float(shear_moduli["G_xz_face_MPa"])

    K4 = np.asarray(
        audit["models"]["beam_free_stress"]["K4"],
        dtype=float,
    )
    K3 = np.asarray(
        audit["models"]["beam_free_stress"]["K3_N_zero"],
        dtype=float,
    )

    K_beta_HOST = float(K3[2, 2])

    # Under N=0, a pure beta mode induces a small eps0 through the anisotropic
    # normal/shear coupling.  This produces the exact HOST pure-beta tau mode
    # used below.
    eps0_per_beta = -float(K4[0, 3]) / float(K4[0, 0])

    fit_block = (
        shear_fit.get("fit_shear_plus_constant_local")
        or shear_fit.get("fit")
        or shear_fit.get("fit_relative_error_only")
    )

    if fit_block is None:
        raise KeyError("Could not locate fitted alpha_G in shear-fit JSON.")

    alpha_global = float(fit_block["alpha_G"])

    result_rows = []
    profiles = {}

    for n_shear in n_shear_values:
        item = selected[n_shear]
        manifest = item["manifest"]
        geometry = manifest["geometry"]

        rows = station_rows(
            item["profile_path"],
            args.station,
        )

        dx_sample = physical_sample_width_mm(rows)

        z = np.asarray(
            [float(row["z_ip_mm"]) for row in rows],
            dtype=float,
        )

        region = [
            str(row["region"])
            for row in rows
        ]

        volume = np.asarray(
            [float(row["volume_mm3"]) for row in rows],
            dtype=float,
        )

        # Half-section area weights; quarter_xy reconstruction is applied below.
        area_half = volume / dx_sample

        gamma_fe = np.asarray(
            [float(row["gamma13_engineering"]) for row in rows],
            dtype=float,
        )

        tau_fe = np.asarray(
            [float(row["S13_MPa"]) for row in rows],
            dtype=float,
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

        h_total = h_core + face_bottom + face_top

        fp = fprime_x(
            z,
            h_total=h_total,
            h_core=h_core,
            G_core=G_warp_core,
            G_face=G_warp_face,
        )

        denominator = float(
            np.sum(
                area_half * fp * fp
            )
        )

        if denominator <= 0.0:
            raise RuntimeError("Degenerate f' projection.")

        beta_fe = float(
            np.sum(
                area_half
                * fp
                * gamma_fe
            )
            / denominator
        )

        gamma_host_fit = (
            beta_fe * fp
        )

        gamma_shape_rmse = weighted_relative_rmse(
            gamma_fe,
            gamma_host_fit,
            area_half,
        )

        gamma_shape_corr = weighted_correlation(
            gamma_fe,
            gamma_host_fit,
            area_half,
        )

        # Reconstruct the full physical width from the half-width quarter model.
        width_factor = 2.0

        V_fe = float(
            width_factor
            * np.sum(
                area_half * tau_fe
            )
        )

        Q_beta_fe = float(
            width_factor
            * np.sum(
                area_half
                * tau_fe
                * fp
            )
        )

        work_beta_projection = float(
            width_factor
            * np.sum(
                area_half
                * tau_fe
                * gamma_host_fit
            )
        )

        K_beta_FE_projected = (
            Q_beta_fe / beta_fe
        )

        alpha_profile = (
            K_beta_FE_projected
            / K_beta_HOST
        )

        # Exact current HOST pure-beta stress shape per unit beta, including
        # the small N=0-induced axial strain.
        tau_host_per_beta = np.empty_like(z)

        for i, reg in enumerate(region):
            Cb = Cb_core if reg == "CORE" else Cb_face

            tau_host_per_beta[i] = (
                float(Cb[1, 0])
                * eps0_per_beta
                + float(Cb[1, 1])
                * fp[i]
            )

        tau_host_at_beta_fe = (
            beta_fe
            * tau_host_per_beta
        )

        tau_shape_rmse = weighted_relative_rmse(
            tau_fe,
            tau_host_at_beta_fe,
            area_half,
        )

        # Shape-only stress comparison: allow a free amplitude so that the
        # metric does not confuse shape error with the already identified
        # stiffness/amplitude error.
        tau_amplitude = float(
            np.sum(
                area_half
                * tau_host_per_beta
                * tau_fe
            )
            / np.sum(
                area_half
                * tau_host_per_beta
                * tau_host_per_beta
            )
        )

        tau_host_shape_fit = (
            tau_amplitude
            * tau_host_per_beta
        )

        tau_shape_only_rmse = weighted_relative_rmse(
            tau_fe,
            tau_host_shape_fit,
            area_half,
        )

        tau_shape_corr = weighted_correlation(
            tau_fe,
            tau_host_shape_fit,
            area_half,
        )

        expected_full_shear = 0.5 * 1000.0

        row_out = {
            "n_shear": int(n_shear),
            "a_over_h": float(
                float(
                    geometry.get(
                        "shear_span",
                        geometry.get("shear_span_mm"),
                    )
                )
                / h_total
            ),
            "station": args.station,
            "x_requested_mm": float(rows[0]["x_requested_mm"]),
            "x_sample_width_mm": float(dx_sample),
            "half_section_area_from_profile_mm2": float(np.sum(area_half)),
            "expected_half_section_area_mm2": float(
                0.5
                * float(
                    geometry.get(
                        "width",
                        geometry.get("width_mm"),
                    )
                )
                * h_total
            ),
            "FE_physical_V_from_S13_N": V_fe,
            "expected_abs_physical_V_N": expected_full_shear,
            "FE_physical_V_relative_error_abs": (
                abs(abs(V_fe) - expected_full_shear)
                / expected_full_shear
            ),
            "beta_FE_projection": beta_fe,
            "gamma_shape_relative_rmse": gamma_shape_rmse,
            "gamma_shape_weighted_correlation": gamma_shape_corr,
            "Q_beta_FE_N": Q_beta_fe,
            "K_beta_HOST_N": K_beta_HOST,
            "K_beta_FE_projected_N": K_beta_FE_projected,
            "alpha_G_profile_projection": alpha_profile,
            "alpha_G_global_deflection_fit": alpha_global,
            "alpha_projection_minus_global": (
                alpha_profile - alpha_global
            ),
            "projected_tau_gamma_work_per_length_N": work_beta_projection,
            "tau_rmse_using_gamma_beta_amplitude": tau_shape_rmse,
            "tau_shape_only_relative_rmse": tau_shape_only_rmse,
            "tau_shape_weighted_correlation": tau_shape_corr,
            "tau_shape_fit_beta_equivalent": tau_amplitude,
            "case_hash": manifest.get("case", {}).get("case_hash"),
            "case_path": str(item["case_dir"]),
        }

        result_rows.append(row_out)

        profiles[str(n_shear)] = {
            "z_mm": z.tolist(),
            "z_over_h": (z / h_total).tolist(),
            "region": region,
            "area_half_mm2": area_half.tolist(),
            "gamma_FE": gamma_fe.tolist(),
            "fprime_HOST": fp.tolist(),
            "gamma_FE_normalized_by_beta_projection": (
                gamma_fe / beta_fe
            ).tolist(),
            "gamma_HOST_normalized": fp.tolist(),
            "tau_FE_MPa": tau_fe.tolist(),
            "tau_HOST_at_beta_FE_MPa": tau_host_at_beta_fe.tolist(),
            "tau_HOST_shape_fit_MPa": tau_host_shape_fit.tolist(),
            "tau_HOST_per_beta_MPa": tau_host_per_beta.tolist(),
        }

    result_rows.sort(key=lambda row: row["a_over_h"])

    alpha_profile_values = np.asarray(
        [
            float(row["alpha_G_profile_projection"])
            for row in result_rows
        ],
        dtype=float,
    )

    alpha_mean = float(np.mean(alpha_profile_values))
    alpha_std = float(np.std(alpha_profile_values, ddof=1))
    alpha_range = float(
        np.max(alpha_profile_values)
        - np.min(alpha_profile_values)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "shear_profile_audit.csv"
    json_path = output_dir / "shear_profile_audit.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(result_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for n_shear in n_shear_values:
            if n_shear not in plot_n_shear:
                continue

            profile = profiles[str(n_shear)]

            ax.plot(
                profile["gamma_FE_normalized_by_beta_projection"],
                profile["z_over_h"],
                marker="o",
                markersize=3,
                label="FE N_A={}".format(n_shear),
            )

        reference_n = next(
            value
            for value in n_shear_values
            if value in plot_n_shear
        )

        reference = profiles[str(reference_n)]

        ax.plot(
            reference["gamma_HOST_normalized"],
            reference["z_over_h"],
            linewidth=2.0,
            label="HOST f'_x(z)",
        )

        ax.set_xlabel(r"$\gamma_{xz}/\beta_{\mathrm{proj}}$")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "normalized_gamma13_profiles.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for n_shear in n_shear_values:
            if n_shear not in plot_n_shear:
                continue

            profile = profiles[str(n_shear)]

            ax.plot(
                profile["tau_FE_MPa"],
                profile["z_over_h"],
                marker="o",
                markersize=3,
                label="FE N_A={}".format(n_shear),
            )

        ax.set_xlabel(r"$\tau_{xz}^{FE}$ [MPa]")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "tau13_FE_profiles.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        a_over_h = [
            row["a_over_h"]
            for row in result_rows
        ]

        alpha_values = [
            row["alpha_G_profile_projection"]
            for row in result_rows
        ]

        ax.plot(
            a_over_h,
            alpha_values,
            marker="o",
            label="FE profile projection",
        )

        ax.axhline(
            alpha_global,
            linewidth=1.5,
            label="global deflection fit",
        )

        ax.set_xlabel("a/h")
        ax.set_ylabel(r"$\alpha_G$")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "alpha_G_profile_vs_global.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

    except Exception as exc:
        plot_paths.append(
            "Plotting skipped: {}".format(exc)
        )

    report = {
        "schema": "host_4pb_shear_profile_audit_v1",
        "station": args.station,
        "warning": (
            "K_beta_FE is a projection of the 3-D FE section fields onto the "
            "current HOST shear mode; it is diagnostic rather than a newly "
            "identified constitutive parameter."
        ),
        "HOST": {
            "K_beta_N": K_beta_HOST,
            "eps0_per_beta_under_N_zero": eps0_per_beta,
            "G_warp_core_MPa": G_warp_core,
            "G_warp_face_MPa": G_warp_face,
            "Cb_core": Cb_core.tolist(),
            "Cb_face": Cb_face.tolist(),
        },
        "global_deflection_fit": {
            "alpha_G": alpha_global,
        },
        "profile_projection_summary": {
            "alpha_G_mean": alpha_mean,
            "alpha_G_sample_std": alpha_std,
            "alpha_G_range": alpha_range,
            "alpha_G_mean_minus_global": alpha_mean - alpha_global,
        },
        "rows": result_rows,
        "profiles": profiles,
        "plots": plot_paths,
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print()
    print("=" * 128)
    print("HOST 4PB FE-vs-HOST TRANSVERSE-SHEAR PROFILE AUDIT")
    print("=" * 128)
    print("Station                         :", args.station)
    print("HOST K_beta                     : {:.9f} N".format(K_beta_HOST))
    print("Global deflection-fit alpha_G   : {:.9f}".format(alpha_global))
    print()
    print(
        "{:>6s} {:>8s} {:>10s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
            "N_A",
            "a/h",
            "|V| err%",
            "gamma RMSE",
            "gamma corr",
            "alpha prof",
            "tau shpRMSE",
            "tau corr",
        )
    )

    for row in result_rows:
        print(
            "{:6d} {:8.3f} {:10.4f} {:11.4%} {:11.6f} {:11.7f} {:11.4%} {:11.6f}".format(
                int(row["n_shear"]),
                float(row["a_over_h"]),
                100.0 * float(row["FE_physical_V_relative_error_abs"]),
                float(row["gamma_shape_relative_rmse"]),
                float(row["gamma_shape_weighted_correlation"]),
                float(row["alpha_G_profile_projection"]),
                float(row["tau_shape_only_relative_rmse"]),
                float(row["tau_shape_weighted_correlation"]),
            )
        )

    print()
    print("Profile-projected alpha_G:")
    print("  mean                           : {:.9f}".format(alpha_mean))
    print("  sample std                     : {:.3e}".format(alpha_std))
    print("  range                          : {:.3e}".format(alpha_range))
    print("  mean - global fit              : {:+.3e}".format(
        alpha_mean - alpha_global
    ))
    print()
    print("Wrote:")
    print(" ", csv_path)
    print(" ", json_path)
    for path in plot_paths:
        print(" ", path)
    print("=" * 128)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
