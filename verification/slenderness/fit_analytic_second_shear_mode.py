#!/usr/bin/env python3
"""
Fit an analytic second transverse-shear mode to the FE-informed residual mode
identified by analyze_4pb_shear_profile_modes.py.

The current material-informed HOST mode is

    f1'(z) = Gc/G(z) * (1 - xi^2),       xi = 2 z / h.

The analytic enrichment family is chosen as

    f2'(z) = f1'(z) * q(xi),

where q is an even polynomial.  This choice has three useful properties:

1. f2' is even, so f2 is odd;
2. f2'(+-h/2)=0 because of the common (1-xi^2) factor;
3. G(z) f2'(z) is continuous at a face/core interface when G is piecewise
   constant, because the Gc/G(z) factor is retained.

To keep the second mode linearly independent from the current one, the
candidate polynomial basis is Gram-Schmidt orthogonalized against f1' using
the same area-weighted inner product used in the FE profile projection.

The script fits polynomial degrees 2, 4, 6 and 8 by default, selects the
lowest degree that satisfies --target-rmse, and then independently evaluates
the local two-mode shear block from the canonical section constitutive law.

The local two-mode condensation diagnostic is

    [Q1]   [K11 K12] [beta1]
    [Q2] = [K12 K22] [beta2]

and, if the secondary generalized shear force is locally free,

    Q2 = 0,

then

    K_eff = K11 - K12^2/K22.

The ratio

    alpha_condensed = K_eff/K11

is compared with the global deflection-fit alpha_G.  This is a diagnostic
comparison only; the full enriched beam retains beta2'(x) and therefore has
a gradient/boundary-layer stiffness through the associated f2(z) chi2 term.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_DEGREES = (2, 4, 6, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modes-json",
        type=Path,
        default=Path(
            "verification/slenderness/results/"
            "gyroid_rd12_initial/shear_profile_modes/"
            "shear_profile_modes.json"
        ),
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
            "gyroid_rd12_initial/analytic_second_shear_mode"
        ),
    )

    parser.add_argument(
        "--degrees",
        default=",".join(str(v) for v in DEFAULT_DEGREES),
    )

    parser.add_argument(
        "--target-rmse",
        type=float,
        default=0.01,
        help="Target relative RMSE for choosing the lowest polynomial degree.",
    )

    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=80,
    )

    return parser.parse_args()


def parse_ints(text: str) -> Tuple[int, ...]:
    values = tuple(
        int(token.strip())
        for token in str(text).split(",")
        if token.strip()
    )

    if not values:
        raise ValueError("At least one degree is required.")

    for degree in values:
        if degree <= 0 or degree % 2:
            raise ValueError(
                "All polynomial degrees must be positive even integers."
            )

    return values


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    Active physical components [eps_xx, gamma_xz], with
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
    reference: np.ndarray,
    approximation: np.ndarray,
    weights: np.ndarray,
) -> float:
    denominator = weighted_norm(reference, weights)

    if denominator <= 0.0:
        return float("nan")

    return (
        weighted_norm(
            np.asarray(reference) - np.asarray(approximation),
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
    mx = float(np.sum(w * x) / total)
    my = float(np.sum(w * y) / total)

    dx = x - mx
    dy = y - my

    vx = float(np.sum(w * dx * dx) / total)
    vy = float(np.sum(w * dy * dy) / total)

    if vx <= 0.0 or vy <= 0.0:
        return float("nan")

    return (
        float(np.sum(w * dx * dy) / total)
        / math.sqrt(vx * vy)
    )


def f1_prime(
    z: np.ndarray,
    *,
    h: float,
    hc: float,
    G_core: float,
    G_face: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    xi = 2.0 * z / float(h)

    p = 1.0 - xi * xi

    core = np.abs(z) <= 0.5 * float(hc) + 1.0e-12

    result = np.empty_like(z)
    result[core] = p[core]
    result[~core] = float(G_core) / float(G_face) * p[~core]

    return result


def q_even(
    xi: np.ndarray,
    coefficients: Dict[int, float],
) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    result = np.zeros_like(xi)

    for power, value in coefficients.items():
        result += float(value) * xi ** int(power)

    return result


def fit_degree(
    residual: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    f1: np.ndarray,
    *,
    h: float,
    degree: int,
) -> dict:
    """
    Fit residual with

        f1 * [sum_n c_n (xi^(2n) - mu_n)]

    so every candidate is area-L2 orthogonal to f1.
    """
    xi = 2.0 * np.asarray(z, dtype=float) / float(h)

    f1_norm_sq = float(
        np.sum(
            weights * f1 * f1
        )
    )

    basis_columns = []
    mus = {}
    powers = tuple(range(2, degree + 1, 2))

    for power in powers:
        raw = f1 * xi ** power

        mu = float(
            np.sum(
                weights * raw * f1
            )
            / f1_norm_sq
        )

        mus[power] = mu

        basis_columns.append(
            f1 * (xi ** power - mu)
        )

    A = np.column_stack(basis_columns)

    sqrt_w = np.sqrt(weights)

    Aw = A * sqrt_w[:, None]
    bw = residual * sqrt_w

    coefficients_vector, _, _, _ = np.linalg.lstsq(
        Aw,
        bw,
        rcond=None,
    )

    approximation = A @ coefficients_vector

    relative_rmse = weighted_relative_rmse(
        residual,
        approximation,
        weights,
    )

    correlation = weighted_correlation(
        residual,
        approximation,
        weights,
    )

    polynomial = {
        0: -sum(
            float(c) * float(mus[p])
            for c, p in zip(coefficients_vector, powers)
        )
    }

    for coefficient, power in zip(coefficients_vector, powers):
        polynomial[int(power)] = float(coefficient)

    df2_raw = f1 * q_even(
        xi,
        polynomial,
    )

    scale = float(
        np.max(
            np.abs(
                df2_raw
            )
        )
    )

    if scale <= 0.0:
        raise RuntimeError("Degenerate second shear mode.")

    polynomial_normalized = {
        int(power): float(value) / scale
        for power, value in polynomial.items()
    }

    df2_normalized = df2_raw / scale

    return {
        "degree": int(degree),
        "powers": list(powers),
        "orthogonalization_mu": {
            str(power): float(mus[power])
            for power in powers
        },
        "raw_coefficients": [
            float(value)
            for value in coefficients_vector
        ],
        "q_coefficients_raw": {
            str(power): float(value)
            for power, value in sorted(polynomial.items())
        },
        "q_coefficients_maxabs_normalized": {
            str(power): float(value)
            for power, value in sorted(polynomial_normalized.items())
        },
        "relative_rmse": float(relative_rmse),
        "weighted_correlation": float(correlation),
        "maxabs_scale": float(scale),
        "approximation": approximation.tolist(),
        "df2_maxabs_normalized": df2_normalized.tolist(),
    }


def integrate_local_beta_block(
    *,
    width: float,
    h: float,
    hc: float,
    Cb_core: np.ndarray,
    Cb_face: np.ndarray,
    G_core: float,
    G_face: float,
    q_coefficients: Dict[int, float],
    quadrature_order: int,
) -> dict:
    """
    Integrate the [eps0, beta1, beta2] constitutive subproblem and condense
    eps0 under N=0.
    """
    hf = 0.5 * (h - hc)

    layers = (
        (-0.5 * h, -0.5 * hc, Cb_face),
        (-0.5 * hc, +0.5 * hc, Cb_core),
        (+0.5 * hc, +0.5 * h, Cb_face),
    )

    points, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )

    K = np.zeros((3, 3), dtype=float)

    for z0, z1, Cb in layers:
        jac = 0.5 * (z1 - z0)
        mid = 0.5 * (z1 + z0)

        for point, weight in zip(points, weights):
            z = mid + jac * float(point)
            xi = 2.0 * z / h

            df1 = float(
                f1_prime(
                    np.asarray([z]),
                    h=h,
                    hc=hc,
                    G_core=G_core,
                    G_face=G_face,
                )[0]
            )

            df2 = df1 * float(
                q_even(
                    np.asarray([xi]),
                    q_coefficients,
                )[0]
            )

            B = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, df1, df2],
                ],
                dtype=float,
            )

            K += (
                float(width)
                * jac
                * float(weight)
                * (B.T @ Cb @ B)
            )

    K00 = float(K[0, 0])
    K0b = K[0, 1:].copy()
    Kbb = K[1:, 1:].copy()

    Kbeta = (
        Kbb
        - np.outer(K0b, K0b) / K00
    )

    K11 = float(Kbeta[0, 0])
    K12 = float(Kbeta[0, 1])
    K22 = float(Kbeta[1, 1])

    K_eff = (
        K11
        - K12 * K12 / K22
    )

    rho = (
        K12
        / math.sqrt(
            K11 * K22
        )
    )

    return {
        "K_eps0_beta_precondensation": K.tolist(),
        "K_beta_N_zero": Kbeta.tolist(),
        "K11_N": K11,
        "K12_N": K12,
        "K22_N": K22,
        "coupling_coefficient_rho": float(rho),
        "rho_squared": float(rho * rho),
        "K_eff_Q2_zero_N": float(K_eff),
        "alpha_local_condensed": float(K_eff / K11),
    }


def primitive_string(
    coefficients: Dict[int, float],
) -> str:
    """
    Human-readable dimensionless primitive R2(xi) where

        f2(z) = (h/2) * Gc/G_layer * R2(xi) + continuity constant.
    """
    terms = []

    # f2' = (Gc/G) (1-xi^2) q(xi).
    expanded = {}

    for power, coefficient in coefficients.items():
        power = int(power)
        coefficient = float(coefficient)

        expanded[power] = expanded.get(power, 0.0) + coefficient
        expanded[power + 2] = (
            expanded.get(power + 2, 0.0) - coefficient
        )

    for power in sorted(expanded):
        coefficient = expanded[power] / float(power + 1)

        if abs(coefficient) < 1.0e-14:
            continue

        terms.append(
            "{:+.12g}*xi^{}".format(
                coefficient,
                power + 1,
            )
        )

    return " ".join(terms) if terms else "0"


def main() -> int:
    args = parse_args()

    modes_path = args.modes_json.resolve()
    audit_path = args.audit.resolve()
    shear_fit_path = args.shear_fit.resolve()
    output_dir = args.output_dir.resolve()

    for path in (modes_path, audit_path, shear_fit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    modes = load_json(modes_path)
    audit = load_json(audit_path)
    shear_fit = load_json(shear_fit_path)

    degrees = parse_ints(args.degrees)

    residual_svd = modes["residual_svd"]

    z = np.asarray(
        residual_svd["common_residual_mode_z_mm"],
        dtype=float,
    )

    residual = np.asarray(
        residual_svd["common_residual_mode"],
        dtype=float,
    )

    # Use the first stored sweep profile for the exact profile weights and f1'.
    profile_key = sorted(
        modes["profiles"],
        key=lambda value: int(value),
    )[0]

    profile = modes["profiles"][profile_key]

    z_profile = np.asarray(
        profile["z_mm"],
        dtype=float,
    )

    if not np.allclose(
        z,
        z_profile,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Residual-mode z-grid does not match stored profile z-grid."
        )

    weights = np.asarray(
        profile["area_half_mm2"],
        dtype=float,
    )

    f1_profile = np.asarray(
        profile["HOST_fprime"],
        dtype=float,
    )

    # Recover geometry/material definition from one actual sweep case.
    row0 = modes["rows"][0]
    case_manifest = load_json(
        Path(row0["case_path"]) / "manifest.json"
    )
    geometry = case_manifest["geometry"]

    hc = float(
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

    h = hc + face_bottom + face_top

    width = float(
        geometry.get(
            "width",
            geometry.get("width_mm"),
        )
    )

    G_core = float(
        audit["shear_moduli"]["G_xz_core_MPa"]
    )
    G_face = float(
        audit["shear_moduli"]["G_xz_face_MPa"]
    )

    # Cross-check the analytical f1' against the exact stored one.
    f1_check = f1_prime(
        z,
        h=h,
        hc=hc,
        G_core=G_core,
        G_face=G_face,
    )

    f1_relative_error = weighted_relative_rmse(
        f1_profile,
        f1_check,
        weights,
    )

    fits = []

    for degree in degrees:
        fits.append(
            fit_degree(
                residual,
                z,
                weights,
                f1_profile,
                h=h,
                degree=degree,
            )
        )

    selected_fit = None

    for fit in fits:
        if float(fit["relative_rmse"]) <= float(args.target_rmse):
            selected_fit = fit
            break

    if selected_fit is None:
        selected_fit = min(
            fits,
            key=lambda item: float(item["relative_rmse"]),
        )

    q_selected = {
        int(power): float(value)
        for power, value in selected_fit[
            "q_coefficients_maxabs_normalized"
        ].items()
    }

    source_manifest = load_json(
        Path(audit["manifest"])
    )

    C_core = np.asarray(
        source_manifest["constitutive"]["C_hom_model_MPa"],
        dtype=float,
    )

    face = audit["face_material"]

    C_face = isotropic_stiffness_3d(
        float(face["E_MPa"]),
        float(face["nu"]),
    )

    Cb_core = free_stress_x_beam_stiffness(C_core)
    Cb_face = free_stress_x_beam_stiffness(C_face)

    local_block = integrate_local_beta_block(
        width=width,
        h=h,
        hc=hc,
        Cb_core=Cb_core,
        Cb_face=Cb_face,
        G_core=G_core,
        G_face=G_face,
        q_coefficients=q_selected,
        quadrature_order=int(args.quadrature_order),
    )

    fit_block = (
        shear_fit.get("fit_shear_plus_constant_local")
        or shear_fit.get("fit_relative_error_only")
        or shear_fit.get("fit")
    )

    alpha_global = float(
        fit_block["alpha_G"]
    )

    K_beta_current = float(
        audit["models"]["beam_free_stress"]["K3_N_zero"][2][2]
    )

    selected_fit["local_two_mode_beta_block"] = local_block
    selected_fit["dimensionless_primitive_R2"] = primitive_string(
        q_selected
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analytic_second_shear_mode.json"
    csv_path = output_dir / "analytic_mode_candidate_fits.csv"

    report = {
        "schema": "host_analytic_second_shear_mode_v1",
        "definition": {
            "xi": "2*z/h",
            "f1_prime": "Gc/G(z) * (1 - xi^2)",
            "f2_prime": "f1_prime * q(xi)",
            "q_even_coefficients": (
                "q(xi)=sum_p a_p*xi^p, p even"
            ),
            "normalization": "max(abs(f2_prime)) = 1 on FE profile z-grid",
        },
        "geometry": {
            "h_mm": h,
            "hc_mm": hc,
            "width_mm": width,
        },
        "material": {
            "G_core_warp_MPa": G_core,
            "G_face_warp_MPa": G_face,
            "Cb_core": Cb_core.tolist(),
            "Cb_face": Cb_face.tolist(),
        },
        "source_residual_svd": {
            "first_mode_explained_fraction": float(
                residual_svd["first_mode_explained_fraction"]
            ),
            "f1_reconstruction_relative_rmse": float(
                f1_relative_error
            ),
        },
        "candidate_fits": fits,
        "selected_degree": int(selected_fit["degree"]),
        "selected_fit": selected_fit,
        "comparison": {
            "current_HOST_Kbeta_N": K_beta_current,
            "local_block_K11_over_current_HOST": (
                float(local_block["K11_N"]) / K_beta_current
            ),
            "alpha_local_condensed": float(
                local_block["alpha_local_condensed"]
            ),
            "alpha_global_deflection_fit": alpha_global,
            "alpha_local_minus_global": (
                float(local_block["alpha_local_condensed"])
                - alpha_global
            ),
        },
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "degree",
                "relative_rmse",
                "weighted_correlation",
            ),
        )
        writer.writeheader()

        for fit in fits:
            writer.writerow(
                {
                    "degree": fit["degree"],
                    "relative_rmse": fit["relative_rmse"],
                    "weighted_correlation": fit["weighted_correlation"],
                }
            )

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        xi = 2.0 * z / h

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(
            residual,
            z / h,
            marker="o",
            markersize=3,
            label="FE/SVD residual mode",
        )

        for fit in fits:
            ax.plot(
                fit["approximation"],
                z / h,
                label="degree {}".format(fit["degree"]),
            )

        ax.set_xlabel("residual shear-mode amplitude")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "analytic_residual_mode_fits.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        df2 = np.asarray(
            selected_fit["df2_maxabs_normalized"],
            dtype=float,
        )

        ax.plot(
            f1_profile,
            z / h,
            marker="o",
            markersize=3,
            label="current f1'",
        )

        ax.plot(
            df2,
            z / h,
            marker="s",
            markersize=3,
            label="analytic f2'",
        )

        ax.set_xlabel("normalized shear mode")
        ax.set_ylabel("z/h")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        path = output_dir / "selected_f1_f2_modes.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

    except Exception as exc:
        plot_paths.append("Plotting skipped: {}".format(exc))

    print()
    print("=" * 112)
    print("HOST ANALYTIC SECOND SHEAR-MODE FIT")
    print("=" * 112)
    print("Residual SVD first-mode fraction : {:.8%}".format(
        float(residual_svd["first_mode_explained_fraction"])
    ))
    print("Stored vs reconstructed f1' RMSE: {:.3e}".format(
        f1_relative_error
    ))
    print()
    print(
        "{:>10s} {:>16s} {:>16s}".format(
            "degree",
            "relative RMSE",
            "correlation",
        )
    )

    for fit in fits:
        print(
            "{:10d} {:15.6%} {:16.9f}".format(
                int(fit["degree"]),
                float(fit["relative_rmse"]),
                float(fit["weighted_correlation"]),
            )
        )

    print()
    print("Selected degree                  :", selected_fit["degree"])
    print("Selected q(xi) coefficients:")
    for power, value in sorted(
        q_selected.items()
    ):
        print(
            "  xi^{:<2d} : {:+.12e}".format(
                int(power),
                float(value),
            )
        )

    print()
    print("Dimensionless primitive R2(xi):")
    print(" ", selected_fit["dimensionless_primitive_R2"])
    print()
    print("Two-mode local shear block after N=0:")
    print(
        np.asarray(
            local_block["K_beta_N_zero"],
            dtype=float,
        )
    )
    print(
        "K11/current HOST                : {:.9f}".format(
            float(local_block["K11_N"]) / K_beta_current
        )
    )
    print(
        "rho=K12/sqrt(K11*K22)           : {:+.9f}".format(
            float(local_block["coupling_coefficient_rho"])
        )
    )
    print(
        "rho^2                           : {:.6%}".format(
            float(local_block["rho_squared"])
        )
    )
    print(
        "alpha local condensed           : {:.9f}".format(
            float(local_block["alpha_local_condensed"])
        )
    )
    print(
        "alpha global deflection fit     : {:.9f}".format(
            alpha_global
        )
    )
    print(
        "local - global                  : {:+.3e}".format(
            float(local_block["alpha_local_condensed"])
            - alpha_global
        )
    )
    print()
    print("Wrote:")
    print(" ", json_path)
    print(" ", csv_path)
    for path in plot_paths:
        print(" ", path)
    print("=" * 112)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
