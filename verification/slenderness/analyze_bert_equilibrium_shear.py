#!/usr/bin/env python3
"""
Gordaninejad-Bert equilibrium/energy shear audit for the canonical HOST 4PB
sandwich beam.

Purpose
-------
Gordaninejad & Bert (1989) improve a Timoshenko-type sandwich beam by deriving
an energy-equivalent shear correction from the nonuniform through-thickness
shear stress implied by beam equilibrium.  This script translates that idea
to the canonical homogenized TPMS sandwich section and asks four separate
questions:

1. What equilibrium/energy-equivalent physical shear stiffness S_eq follows
   from the actual layered E_x(z), G_xz(z)?
2. How does that stiffness compare with the physical shear stiffness represented
   by the current material-informed HOST shear mode?
3. What happens if the primary HOST warping derivative f1'(z) is replaced by
   the equilibrium-derived shear-strain profile?
4. If a variational second mode is then derived orthogonally to that new
   equilibrium primary mode, does the full two-mode beam improve or degrade the
   existing variational two-mode result?

Important distinction
---------------------
The paper's S_eq maps the *physical shear resultant* V to shear strain energy:

    U_s = V^2 / (2 S_eq).

The current HOST K_beta maps a generalized shear amplitude beta to a
generalized resultant Q_beta.  Their numerical ratio depends on modal
normalization and is NOT an invariant "alpha".  Therefore this script compares
S_eq only against another physical-resultant-equivalent stiffness obtained from
each candidate warping mode:

    S_mode = V(beta)^2 / K_beta,

under N=0.

The script also reports the already established generalized local condensation
ratio of the variational two-mode model separately.

Paper-compatible scalar equilibrium construction
------------------------------------------------
With

    A = int_A E dA
    B = int_A z E dA
    D = int_A z^2 E dA
    z_n = B/A
    D_R = D - B^2/A,

N=0 gives

    sigma_x = E(z) (z-z_n) kappa,
    M = D_R kappa.

Since M' = V,

    kappa' = V / D_R.

The x-z equilibrium equation then yields, per unit physical V,

    tau_hat(z)
        = - 1/D_R
          int_{z_b}^{z} E(s) (s-z_n) ds.

The corresponding paper-compatible shear compliance is

    1/S_eq
        = int_A tau_hat(z)^2 / G(z) dA.

The associated equilibrium shear-strain shape is

    gamma_hat(z) = tau_hat(z) / G(z).

For the canonical symmetric section this is even, so its integral defines an
odd equilibrium primary warping f_eq(z).

Anisotropic core
----------------
The equilibrium construction above intentionally uses the diagonal entries of
the same free-stress-condensed 2x2 beam constitutive matrix used by HOST:

    E_x   = Cb[0,0]
    G_xz  = Cb[1,1].

The normalized Cb[0,1] coupling is reported.  Full 2x2 constitutive coupling is
retained when candidate warping modes are inserted back into the HOST section
matrices.  This makes the paper-compatible scalar construction and its small
anisotropic correction distinguishable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.integrate import quad

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
from host.constitutive import isotropic_stiffness_3d
from host.sandwich_warping import material_informed_symmetric_warping
from host.variational_shear_modes import (
    BeamLayer2D,
    derive_variational_second_shear_mode,
    free_stress_beam_stiffness,
)


DEFAULT_REFERENCE = Path(
    "reference_results/4pb/"
    "Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1/"
    "source_case_manifest.json"
)

DEFAULT_SWEEP = Path(
    "verification/slenderness/results/"
    "gyroid_rd12_initial/slenderness_sweep.json"
)

DEFAULT_SHEAR_FIT = Path(
    "verification/slenderness/results/"
    "gyroid_rd12_initial/shear_compliance/"
    "shear_compliance_audit.json"
)

DEFAULT_MODES = Path(
    "verification/slenderness/results/"
    "gyroid_rd12_initial/shear_profile_modes/"
    "shear_profile_modes.json"
)

DEFAULT_OUTPUT = Path(
    "verification/slenderness/results/"
    "gyroid_rd12_initial/bert_equilibrium_shear"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
    )

    parser.add_argument(
        "--sweep",
        type=Path,
        default=DEFAULT_SWEEP,
    )

    parser.add_argument(
        "--shear-fit",
        type=Path,
        default=DEFAULT_SHEAR_FIT,
    )

    parser.add_argument(
        "--modes-json",
        type=Path,
        default=DEFAULT_MODES,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
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
        "--max-segment-mm",
        type=float,
        default=10.0,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def layer_at(
    layers: Sequence[BeamLayer2D],
    z: float,
) -> BeamLayer2D:
    z = float(z)

    for index, layer in enumerate(layers):
        last = index == len(layers) - 1

        if (
            layer.z0 <= z < layer.z1
            or (
                last
                and layer.z0 <= z <= layer.z1
            )
        ):
            return layer

    raise RuntimeError(
        "Could not identify layer at z={:.12g}.".format(z)
    )


def integrate_layerwise(
    layers: Sequence[BeamLayer2D],
    function: Callable[[float, BeamLayer2D], float],
    *,
    epsabs: float = 1.0e-10,
    epsrel: float = 1.0e-11,
) -> float:
    value = 0.0

    for layer in layers:
        value += quad(
            lambda z: float(function(z, layer)),
            float(layer.z0),
            float(layer.z1),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=100,
        )[0]

    return float(value)


def section_ABD(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
) -> dict:
    width = float(width)

    A = width * integrate_layerwise(
        layers,
        lambda z, layer: layer.C[0, 0],
    )

    B = width * integrate_layerwise(
        layers,
        lambda z, layer: z * layer.C[0, 0],
    )

    D = width * integrate_layerwise(
        layers,
        lambda z, layer: z * z * layer.C[0, 0],
    )

    zn = B / A
    DR = D - B * B / A

    return {
        "A_N": A,
        "B_Nmm": B,
        "D_Nmm2": D,
        "neutral_axis_mm": zn,
        "D_R_Nmm2": DR,
    }


class EquilibriumShearWarping:
    """
    Paper-compatible equilibrium shear profile and its warping primitive.

    The raw derivative is gamma_hat(z) = tau_hat(z)/G(z) for unit physical
    shear resultant V=1 N.  A deterministic max-absolute normalization is used
    for the HOST warping representation.  S_eq itself is invariant and uses
    the unnormalized unit-V profile.
    """

    def __init__(
        self,
        layers: Sequence[BeamLayer2D],
        *,
        width: float,
        neutral_axis: float,
        D_R: float,
    ) -> None:
        self.layers = tuple(layers)
        self.width = float(width)
        self.neutral_axis = float(neutral_axis)
        self.D_R = float(D_R)

        self.z_bottom = float(self.layers[0].z0)
        self.z_top = float(self.layers[-1].z1)

        grid_parts = []

        for layer in self.layers:
            grid_parts.append(
                np.linspace(
                    layer.z0,
                    layer.z1,
                    401,
                )
            )

        grid = np.unique(
            np.concatenate(grid_parts)
        )

        raw = np.asarray(
            [
                self.gamma_hat(z)
                for z in grid
            ],
            dtype=float,
        )

        scale = float(np.max(np.abs(raw)))

        if scale <= 0.0:
            raise RuntimeError(
                "Equilibrium shear-strain profile is degenerate."
            )

        self.normalization = scale

    def first_moment_below(self, z: float) -> float:
        z = float(z)

        if z <= self.z_bottom:
            return 0.0

        if z >= self.z_top:
            z = self.z_top

        total = 0.0

        for layer in self.layers:
            if z <= layer.z0:
                break

            upper = min(
                z,
                float(layer.z1),
            )

            E = float(
                layer.C[0, 0]
            )

            z0 = float(layer.z0)

            total += E * (
                0.5
                * (
                    upper * upper
                    - z0 * z0
                )
                - self.neutral_axis
                * (
                    upper
                    - z0
                )
            )

            if z <= layer.z1:
                break

        return float(total)

    def tau_hat(self, z: float) -> float:
        """
        tau_xz / V for V=1 N, units 1/mm^2.
        """
        return (
            -self.first_moment_below(z)
            / self.D_R
        )

    def gamma_hat(self, z: float) -> float:
        """
        gamma_xz / V for V=1 N, units 1/N.
        """
        layer = layer_at(
            self.layers,
            z,
        )

        G = float(
            layer.C[1, 1]
        )

        return self.tau_hat(z) / G

    def df(self, z: float) -> float:
        return (
            self.gamma_hat(z)
            / self.normalization
        )

    def f(self, z: float) -> float:
        z = float(z)

        if z == 0.0:
            return 0.0

        lower = min(0.0, z)
        upper = max(0.0, z)

        points = sorted(
            {
                float(layer.z0)
                for layer in self.layers
                if lower < layer.z0 < upper
            }
            | {
                float(layer.z1)
                for layer in self.layers
                if lower < layer.z1 < upper
            }
        )

        value = quad(
            lambda s: self.df(s),
            lower,
            upper,
            points=points if points else None,
            epsabs=1.0e-11,
            epsrel=1.0e-11,
            limit=200,
        )[0]

        return float(
            value
            if z > 0.0
            else -value
        )


def bert_equivalent_shear_stiffness(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
    equilibrium_mode: EquilibriumShearWarping,
) -> dict:
    width = float(width)

    compliance = width * integrate_layerwise(
        layers,
        lambda z, layer: (
            equilibrium_mode.tau_hat(z) ** 2
            / float(layer.C[1, 1])
        ),
    )

    S_eq = 1.0 / compliance

    S_raw = width * integrate_layerwise(
        layers,
        lambda z, layer: layer.C[1, 1],
    )

    resultant_check = width * integrate_layerwise(
        layers,
        lambda z, layer: equilibrium_mode.tau_hat(z),
    )

    return {
        "S_eq_N": float(S_eq),
        "S_raw_integral_G_dA_N": float(S_raw),
        "K_energy_equivalent": float(S_eq / S_raw),
        "unit_V_resultant_check_N": float(resultant_check),
        "tau_bottom_per_N_mm2": float(
            equilibrium_mode.tau_hat(
                equilibrium_mode.z_bottom
            )
        ),
        "tau_top_per_N_mm2": float(
            equilibrium_mode.tau_hat(
                equilibrium_mode.z_top
            )
        ),
    }


def physical_mode_shear_stiffness(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
    df: Callable[[float], float],
    include_normal_shear_coupling: bool,
) -> dict:
    """
    Convert an arbitrary modal derivative into an invariant physical-resultant
    equivalent stiffness.

    beta is assigned an arbitrary unit amplitude.  eps0 is chosen so N=0.
    """

    width = float(width)

    A = width * integrate_layerwise(
        layers,
        lambda z, layer: layer.C[0, 0],
    )

    if include_normal_shear_coupling:
        N_beta = width * integrate_layerwise(
            layers,
            lambda z, layer: (
                layer.C[0, 1]
                * float(df(z))
            ),
        )
    else:
        N_beta = 0.0

    eps0 = -N_beta / A

    def strain_vector(z, layer):
        shear = float(df(z))

        return np.asarray(
            [
                eps0,
                shear,
            ],
            dtype=float,
        )

    if include_normal_shear_coupling:
        V = width * integrate_layerwise(
            layers,
            lambda z, layer: (
                layer.C[1, 0]
                * eps0
                + layer.C[1, 1]
                * float(df(z))
            ),
        )

        K_beta = width * integrate_layerwise(
            layers,
            lambda z, layer: float(
                strain_vector(z, layer)
                @ layer.C
                @ strain_vector(z, layer)
            ),
        )

    else:
        V = width * integrate_layerwise(
            layers,
            lambda z, layer: (
                layer.C[1, 1]
                * float(df(z))
            ),
        )

        K_beta = width * integrate_layerwise(
            layers,
            lambda z, layer: (
                layer.C[1, 1]
                * float(df(z)) ** 2
            ),
        )

    S_physical = V * V / K_beta

    return {
        "eps0_per_beta": float(eps0),
        "V_per_beta_N": float(V),
        "K_beta_N": float(K_beta),
        "S_physical_N": float(S_physical),
    }


def assemble_one_mode_K4(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
    f: Callable[[float], float],
    df: Callable[[float], float],
    quadrature_order: int = 18,
) -> np.ndarray:
    """
    Generalized order:
        [eps0, kappa, chi, beta].
    """

    points, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )

    K4 = np.zeros(
        (4, 4),
        dtype=float,
    )

    for layer in layers:
        z0 = float(layer.z0)
        z1 = float(layer.z1)

        jac = 0.5 * (z1 - z0)
        mid = 0.5 * (z1 + z0)

        for point, weight in zip(
            points,
            weights,
        ):
            z = mid + jac * float(point)

            B = np.asarray(
                [
                    [
                        1.0,
                        z,
                        float(f(z)),
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        float(df(z)),
                    ],
                ],
                dtype=float,
            )

            K4 += (
                float(width)
                * jac
                * float(weight)
                * (B.T @ layer.C @ B)
            )

    return 0.5 * (K4 + K4.T)


def condense_one_mode_N_zero(
    K4: np.ndarray,
) -> np.ndarray:
    K4 = np.asarray(
        K4,
        dtype=float,
    )

    K00 = float(
        K4[0, 0]
    )

    K0t = K4[0, 1:]
    Ktt = K4[1:, 1:]

    K3 = (
        Ktt
        - np.outer(K0t, K0t)
        / K00
    )

    return 0.5 * (
        K3 + K3.T
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


def solve_one_mode_sweep(
    K3: np.ndarray,
    rows_source: Sequence[dict],
) -> List[dict]:
    results = []

    for row in rows_source:
        solution = solve_simply_supported_point_loads(
            K3,
            length=float(row["length_mm"]),
            support_left=float(row["support_left_mm"]),
            support_right=float(row["support_right_mm"]),
            applied_forces=(
                PointForce(
                    float(row["load_left_mm"]),
                    500.0,
                ),
                PointForce(
                    float(row["load_right_mm"]),
                    500.0,
                ),
            ),
        )

        mid = solution.state(
            0.5 * float(row["length_mm"])
        )

        results.append(
            {
                "n_shear": int(
                    row["n_shear"]
                ),
                "a_over_h": float(
                    row["a_over_h"]
                ),
                "w_abs_mm": abs(
                    float(mid[0])
                ),
            }
        )

    return results


def solve_two_mode_sweep(
    K5: np.ndarray,
    rows_source: Sequence[dict],
    *,
    max_segment_mm: float,
) -> List[dict]:
    results = []

    for row in rows_source:
        solution = solve_simply_supported_two_mode_point_loads(
            K5,
            length=float(row["length_mm"]),
            support_left=float(row["support_left_mm"]),
            support_right=float(row["support_right_mm"]),
            applied_forces=(
                PointForce(
                    float(row["load_left_mm"]),
                    500.0,
                ),
                PointForce(
                    float(row["load_right_mm"]),
                    500.0,
                ),
            ),
            max_segment_length=float(
                max_segment_mm
            ),
        )

        mid = solution.state(
            0.5 * float(row["length_mm"])
        )

        results.append(
            {
                "n_shear": int(
                    row["n_shear"]
                ),
                "a_over_h": float(
                    row["a_over_h"]
                ),
                "w_abs_mm": abs(
                    float(mid[0])
                ),
            }
        )

    return results


def weighted_norm(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    return math.sqrt(
        float(
            np.sum(
                np.asarray(weights, dtype=float)
                * np.asarray(values, dtype=float) ** 2
            )
        )
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
            "amplitude": float("nan"),
            "relative_rmse": float("nan"),
            "correlation": float("nan"),
        }

    amplitude = float(
        np.sum(
            weights
            * candidate
            * reference
        )
        / denominator
    )

    fitted = amplitude * candidate

    reference_norm = weighted_norm(
        reference,
        weights,
    )

    relative_rmse = (
        weighted_norm(
            reference - fitted,
            weights,
        )
        / reference_norm
        if reference_norm > 0.0
        else float("nan")
    )

    total = float(
        np.sum(weights)
    )

    ref_mean = float(
        np.sum(weights * reference)
        / total
    )

    fit_mean = float(
        np.sum(weights * fitted)
        / total
    )

    ref_centered = (
        reference - ref_mean
    )

    fit_centered = (
        fitted - fit_mean
    )

    variance_product = float(
        np.sum(
            weights
            * ref_centered ** 2
        )
        * np.sum(
            weights
            * fit_centered ** 2
        )
    )

    correlation = (
        float(
            np.sum(
                weights
                * ref_centered
                * fit_centered
            )
            / math.sqrt(
                variance_product
            )
        )
        if variance_product > 0.0
        else float("nan")
    )

    return {
        "amplitude": amplitude,
        "relative_rmse": float(
            relative_rmse
        ),
        "correlation": float(
            correlation
        ),
    }


def extract_alpha_global(
    path: Path,
) -> float:
    if not path.is_file():
        return float("nan")

    data = load_json(path)

    priority = (
        "fit_shear_plus_constant_local",
        "fit_shear_plus_local",
        "fit_relative_error_only",
        "fit",
    )

    for key in priority:
        value = data.get(key)

        if (
            isinstance(value, dict)
            and "alpha_G" in value
        ):
            return float(
                value["alpha_G"]
            )

    def walk(value):
        if isinstance(value, dict):
            if "alpha_G" in value:
                try:
                    return float(
                        value["alpha_G"]
                    )
                except Exception:
                    pass

            for nested in value.values():
                found = walk(nested)

                if found is not None:
                    return found

        elif isinstance(value, list):
            for nested in value:
                found = walk(nested)

                if found is not None:
                    return found

        return None

    found = walk(data)

    return (
        float(found)
        if found is not None
        else float("nan")
    )


def profile_case_paths(
    modes_json: Path,
) -> List[dict]:
    if not modes_json.is_file():
        return []

    data = load_json(
        modes_json
    )

    result = []

    for index, row in enumerate(
        data.get("rows", [])
    ):
        case_path = row.get(
            "case_path"
        )

        if not case_path:
            continue

        n_shear = row.get(
            "n_shear",
            row.get(
                "N_A",
                index,
            ),
        )

        result.append(
            {
                "n_shear": n_shear,
                "case_path": str(
                    case_path
                ),
            }
        )

    return result


def load_through_thickness_profile(
    case_path: Path,
    *,
    station: str = "left_shear_mid",
) -> dict | None:
    path = (
        case_path
        / "post"
        / "through_thickness_profiles.csv"
    )

    if not path.is_file():
        return None

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
            if row.get("station") != station:
                continue

            region = row.get(
                "region",
                "",
            )

            if region not in {
                "FACE_BOTTOM",
                "CORE",
                "FACE_TOP",
            }:
                continue

            try:
                rows.append(
                    {
                        "z_mm": float(
                            row["z_ip_mm"]
                        ),
                        "gamma13": float(
                            row[
                                "gamma13_engineering"
                            ]
                        ),
                        "tau13_MPa": float(
                            row["S13_MPa"]
                        ),
                        "weight": float(
                            row["volume_mm3"]
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
        return None

    rows.sort(
        key=lambda item: item["z_mm"]
    )

    return {
        "path": str(path),
        "z_mm": np.asarray(
            [
                row["z_mm"]
                for row in rows
            ],
            dtype=float,
        ),
        "gamma13": np.asarray(
            [
                row["gamma13"]
                for row in rows
            ],
            dtype=float,
        ),
        "tau13_MPa": np.asarray(
            [
                row["tau13_MPa"]
                for row in rows
            ],
            dtype=float,
        ),
        "weights": np.asarray(
            [
                row["weight"]
                for row in rows
            ],
            dtype=float,
        ),
    }


def main() -> int:
    args = parse_args()

    reference_path = args.reference.resolve()
    sweep_path = args.sweep.resolve()
    shear_fit_path = args.shear_fit.resolve()
    modes_path = args.modes_json.resolve()
    output_dir = args.output_dir.resolve()

    if not reference_path.is_file():
        raise FileNotFoundError(
            reference_path
        )

    if not sweep_path.is_file():
        raise FileNotFoundError(
            sweep_path
        )

    reference = load_json(
        reference_path
    )

    sweep = load_json(
        sweep_path
    )

    rows_source = [
        row
        for row in sweep.get(
            "rows",
            [],
        )
        if row.get(
            "FE_3D_abs_w_mid_mm"
        )
        not in ("", None)
    ]

    if not rows_source:
        raise RuntimeError(
            "No FE-populated rows found in slenderness sweep."
        )

    first = rows_source[0]

    h = float(
        first["total_height_mm"]
    )

    hc = float(
        first["core_height_mm"]
    )

    width = float(
        first["width_mm"]
    )

    Ccore3 = np.asarray(
        reference[
            "constitutive"
        ][
            "C_hom_model_MPa"
        ],
        dtype=float,
    )

    Cface3 = isotropic_stiffness_3d(
        E=float(
            args.face_E_MPa
        ),
        nu=float(
            args.face_nu
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

    full_layers = (
        BeamLayer2D(
            "face_bottom",
            -0.5 * h,
            -0.5 * hc,
            Cface,
        ),
        BeamLayer2D(
            "core",
            -0.5 * hc,
            +0.5 * hc,
            Ccore,
        ),
        BeamLayer2D(
            "face_top",
            +0.5 * hc,
            +0.5 * h,
            Cface,
        ),
    )

    half_layers = (
        BeamLayer2D(
            "core+",
            0.0,
            0.5 * hc,
            Ccore,
        ),
        BeamLayer2D(
            "face+",
            0.5 * hc,
            0.5 * h,
            Cface,
        ),
    )

    abd = section_ABD(
        full_layers,
        width=width,
    )

    equilibrium = EquilibriumShearWarping(
        full_layers,
        width=width,
        neutral_axis=abd[
            "neutral_axis_mm"
        ],
        D_R=abd[
            "D_R_Nmm2"
        ],
    )

    bert = bert_equivalent_shear_stiffness(
        full_layers,
        width=width,
        equilibrium_mode=equilibrium,
    )

    current_f1 = material_informed_symmetric_warping(
        h=h,
        hc=hc,
        G_face=float(
            Cface[1, 1]
        ),
        G_core=float(
            Ccore[1, 1]
        ),
    )

    current_physical_diag = physical_mode_shear_stiffness(
        full_layers,
        width=width,
        df=current_f1.df,
        include_normal_shear_coupling=False,
    )

    current_physical_full = physical_mode_shear_stiffness(
        full_layers,
        width=width,
        df=current_f1.df,
        include_normal_shear_coupling=True,
    )

    equilibrium_physical_diag = physical_mode_shear_stiffness(
        full_layers,
        width=width,
        df=equilibrium.df,
        include_normal_shear_coupling=False,
    )

    equilibrium_physical_full = physical_mode_shear_stiffness(
        full_layers,
        width=width,
        df=equilibrium.df,
        include_normal_shear_coupling=True,
    )

    # ----------------------------------------------------------
    # Current one-mode and equilibrium-primary one-mode sections.
    # ----------------------------------------------------------

    K4_current = assemble_one_mode_K4(
        full_layers,
        width=width,
        f=current_f1.f,
        df=current_f1.df,
    )

    K3_current_rebuilt = condense_one_mode_N_zero(
        K4_current
    )

    K4_equilibrium = assemble_one_mode_K4(
        full_layers,
        width=width,
        f=equilibrium.f,
        df=equilibrium.df,
    )

    K3_equilibrium = condense_one_mode_N_zero(
        K4_equilibrium
    )

    current_one = solve_one_mode_sweep(
        K3_current_rebuilt,
        rows_source,
    )

    equilibrium_one = solve_one_mode_sweep(
        K3_equilibrium,
        rows_source,
    )

    # ----------------------------------------------------------
    # Current-f1 + variational-f2 two-mode reference.
    # ----------------------------------------------------------

    current_f2 = derive_variational_second_shear_mode(
        half_layers,
        current_f1.f,
        elements_per_layer=(
            int(
                args.core_mode_elements
            ),
            int(
                args.face_mode_elements
            ),
        ),
    )

    current_two_section = build_two_mode_beam_section(
        full_layers,
        width=width,
        mode1=WarpingMode(
            current_f1.f,
            current_f1.df,
        ),
        mode2=WarpingMode(
            current_f2.f,
            current_f2.df,
        ),
        quadrature_order=8,
        integration_breakpoints=mirrored_breakpoints(
            current_f2.nodes_half
        ),
    )

    current_two = solve_two_mode_sweep(
        current_two_section.K5,
        rows_source,
        max_segment_mm=float(
            args.max_segment_mm
        ),
    )

    # ----------------------------------------------------------
    # Equilibrium-f1 + variational-f2 proposed hierarchy.
    # ----------------------------------------------------------

    equilibrium_f2 = derive_variational_second_shear_mode(
        half_layers,
        equilibrium.f,
        elements_per_layer=(
            int(
                args.core_mode_elements
            ),
            int(
                args.face_mode_elements
            ),
        ),
    )

    equilibrium_two_section = build_two_mode_beam_section(
        full_layers,
        width=width,
        mode1=WarpingMode(
            equilibrium.f,
            equilibrium.df,
        ),
        mode2=WarpingMode(
            equilibrium_f2.f,
            equilibrium_f2.df,
        ),
        quadrature_order=8,
        integration_breakpoints=mirrored_breakpoints(
            equilibrium_f2.nodes_half
        ),
    )

    equilibrium_two = solve_two_mode_sweep(
        equilibrium_two_section.K5,
        rows_source,
        max_segment_mm=float(
            args.max_segment_mm
        ),
    )

    current_spectrum = boundary_layer_spectrum(
        current_two_section.K5
    )

    equilibrium_spectrum = boundary_layer_spectrum(
        equilibrium_two_section.K5
    )

    alpha_global = extract_alpha_global(
        shear_fit_path
    )

    alpha_variational_current = local_shear_condensation_ratio(
        current_two_section.K5
    )

    alpha_variational_equilibrium = local_shear_condensation_ratio(
        equilibrium_two_section.K5
    )

    # ----------------------------------------------------------
    # Beam comparison table.
    # ----------------------------------------------------------

    result_rows = []

    for source, cur1, eq1, cur2, eq2 in zip(
        rows_source,
        current_one,
        equilibrium_one,
        current_two,
        equilibrium_two,
    ):
        w_fe = float(
            source[
                "FE_3D_abs_w_mid_mm"
            ]
        )

        w_stored_current = float(
            source[
                "HOST_1D_abs_w_mid_mm"
            ]
        )

        row = {
            "n_shear": int(
                source["n_shear"]
            ),
            "a_over_h": float(
                source["a_over_h"]
            ),
            "FE_3D_abs_w_mid_mm": w_fe,
            "HOST_stored_current_1mode_mm": w_stored_current,
            "HOST_rebuilt_current_1mode_mm": float(
                cur1["w_abs_mm"]
            ),
            "HOST_equilibrium_1mode_mm": float(
                eq1["w_abs_mm"]
            ),
            "HOST_current_variational_2mode_mm": float(
                cur2["w_abs_mm"]
            ),
            "HOST_equilibrium_variational_2mode_mm": float(
                eq2["w_abs_mm"]
            ),
        }

        for key in (
            "HOST_stored_current_1mode_mm",
            "HOST_rebuilt_current_1mode_mm",
            "HOST_equilibrium_1mode_mm",
            "HOST_current_variational_2mode_mm",
            "HOST_equilibrium_variational_2mode_mm",
        ):
            row[
                key.replace(
                    "_mm",
                    "_error_relative",
                )
            ] = (
                float(row[key])
                - w_fe
            ) / w_fe

        result_rows.append(
            row
        )

    error_keys = (
        "HOST_stored_current_1mode_error_relative",
        "HOST_rebuilt_current_1mode_error_relative",
        "HOST_equilibrium_1mode_error_relative",
        "HOST_current_variational_2mode_error_relative",
        "HOST_equilibrium_variational_2mode_error_relative",
    )

    beam_summary = {}

    for key in error_keys:
        values = np.asarray(
            [
                row[key]
                for row in result_rows
            ],
            dtype=float,
        )

        beam_summary[key] = {
            "RMS_percent": float(
                100.0
                * np.sqrt(
                    np.mean(
                        values ** 2
                    )
                )
            ),
            "max_abs_percent": float(
                100.0
                * np.max(
                    np.abs(values)
                )
            ),
        }

    # ----------------------------------------------------------
    # FE through-thickness profile comparison, if case paths are
    # available from the existing shear-profile-mode audit.
    # ----------------------------------------------------------

    profile_results = []

    for case_info in profile_case_paths(
        modes_path
    ):
        profile = load_through_thickness_profile(
            Path(
                case_info["case_path"]
            )
        )

        if profile is None:
            continue

        z = profile["z_mm"]
        weights = profile["weights"]

        current_gamma = np.asarray(
            [
                current_f1.df(value)
                for value in z
            ],
            dtype=float,
        )

        equilibrium_gamma = np.asarray(
            [
                equilibrium.df(value)
                for value in z
            ],
            dtype=float,
        )

        current_tau = np.asarray(
            [
                float(
                    layer_at(
                        full_layers,
                        value,
                    ).C[1, 1]
                )
                * current_f1.df(
                    value
                )
                for value in z
            ],
            dtype=float,
        )

        equilibrium_tau = np.asarray(
            [
                equilibrium.tau_hat(
                    value
                )
                for value in z
            ],
            dtype=float,
        )

        gamma_current_fit = weighted_shape_fit(
            profile["gamma13"],
            current_gamma,
            weights,
        )

        gamma_equilibrium_fit = weighted_shape_fit(
            profile["gamma13"],
            equilibrium_gamma,
            weights,
        )

        tau_current_fit = weighted_shape_fit(
            profile["tau13_MPa"],
            current_tau,
            weights,
        )

        tau_equilibrium_fit = weighted_shape_fit(
            profile["tau13_MPa"],
            equilibrium_tau,
            weights,
        )

        profile_results.append(
            {
                "n_shear": case_info[
                    "n_shear"
                ],
                "profile_path": profile[
                    "path"
                ],
                "gamma_current_RMSE_percent": (
                    100.0
                    * gamma_current_fit[
                        "relative_rmse"
                    ]
                ),
                "gamma_current_correlation": gamma_current_fit[
                    "correlation"
                ],
                "gamma_equilibrium_RMSE_percent": (
                    100.0
                    * gamma_equilibrium_fit[
                        "relative_rmse"
                    ]
                ),
                "gamma_equilibrium_correlation": gamma_equilibrium_fit[
                    "correlation"
                ],
                "tau_current_RMSE_percent": (
                    100.0
                    * tau_current_fit[
                        "relative_rmse"
                    ]
                ),
                "tau_current_correlation": tau_current_fit[
                    "correlation"
                ],
                "tau_equilibrium_RMSE_percent": (
                    100.0
                    * tau_equilibrium_fit[
                        "relative_rmse"
                    ]
                ),
                "tau_equilibrium_correlation": tau_equilibrium_fit[
                    "correlation"
                ],
            }
        )

    profile_summary = {}

    if profile_results:
        metric_keys = (
            "gamma_current_RMSE_percent",
            "gamma_equilibrium_RMSE_percent",
            "tau_current_RMSE_percent",
            "tau_equilibrium_RMSE_percent",
        )

        for key in metric_keys:
            profile_summary[
                "mean_" + key
            ] = float(
                np.mean(
                    [
                        row[key]
                        for row in profile_results
                    ]
                )
            )

    # ----------------------------------------------------------
    # Output.
    # ----------------------------------------------------------

    coupling_core = float(
        abs(Ccore[0, 1])
        / math.sqrt(
            Ccore[0, 0]
            * Ccore[1, 1]
        )
    )

    coupling_face = float(
        abs(Cface[0, 1])
        / math.sqrt(
            Cface[0, 0]
            * Cface[1, 1]
        )
    )

    scalar_comparisons = {
        "alpha_global_generalized_fit": alpha_global,
        "alpha_current_variational_local_generalized": (
            alpha_variational_current
        ),
        "alpha_equilibrium_variational_local_generalized": (
            alpha_variational_equilibrium
        ),
        "Bert_S_eq_over_current_f1_physical_diag": (
            bert["S_eq_N"]
            / current_physical_diag[
                "S_physical_N"
            ]
        ),
        "Bert_S_eq_over_current_f1_physical_full": (
            bert["S_eq_N"]
            / current_physical_full[
                "S_physical_N"
            ]
        ),
        "Bert_S_eq_over_equilibrium_mode_physical_diag": (
            bert["S_eq_N"]
            / equilibrium_physical_diag[
                "S_physical_N"
            ]
        ),
        "Bert_S_eq_over_equilibrium_mode_physical_full": (
            bert["S_eq_N"]
            / equilibrium_physical_full[
                "S_physical_N"
            ]
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "bert_equilibrium_shear_audit.json"
    )

    beam_csv_path = (
        output_dir
        / "bert_equilibrium_beam_comparison.csv"
    )

    profile_csv_path = (
        output_dir
        / "bert_equilibrium_profile_comparison.csv"
    )

    profile_curve_path = (
        output_dir
        / "bert_equilibrium_section_profile.csv"
    )

    z_curve = np.linspace(
        -0.5 * h,
        +0.5 * h,
        801,
    )

    with profile_curve_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            (
                "z_mm",
                "current_HOST_f1_prime",
                "equilibrium_f1_prime",
                "equilibrium_tau_per_unit_V_1_per_mm2",
                "equilibrium_gamma_per_unit_V_1_per_N",
            )
        )

        for z in z_curve:
            writer.writerow(
                (
                    float(z),
                    float(
                        current_f1.df(z)
                    ),
                    float(
                        equilibrium.df(z)
                    ),
                    float(
                        equilibrium.tau_hat(z)
                    ),
                    float(
                        equilibrium.gamma_hat(z)
                    ),
                )
            )

    with beam_csv_path.open(
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

    if profile_results:
        with profile_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    profile_results[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                profile_results
            )

    report = {
        "schema": "host_bert_equilibrium_shear_audit_v1",
        "paper_translation": {
            "construction": (
                "energy-equivalent physical shear stiffness from the "
                "equilibrium shear-stress distribution implied by layered "
                "bending under N=0"
            ),
            "note": (
                "Physical-resultant-equivalent stiffnesses are compared "
                "separately from generalized HOST beta-sector condensation "
                "ratios because their modal normalizations differ."
            ),
        },
        "geometry": {
            "h_mm": h,
            "hc_mm": hc,
            "width_mm": width,
        },
        "constitutive": {
            "Cb_core": Ccore.tolist(),
            "Cb_face": Cface.tolist(),
            "normalized_C15_core": coupling_core,
            "normalized_C15_face": coupling_face,
        },
        "ABD": abd,
        "Bert_equilibrium": bert,
        "current_f1_physical_diag": current_physical_diag,
        "current_f1_physical_full": current_physical_full,
        "equilibrium_f1_physical_diag": equilibrium_physical_diag,
        "equilibrium_f1_physical_full": equilibrium_physical_full,
        "scalar_comparisons": scalar_comparisons,
        "current_variational_second_mode": {
            "eigenvalue_per_mm2": float(
                current_f2.eigenvalue
            ),
            "section_length_mm": float(
                current_f2.characteristic_length
            ),
            "local_generalized_alpha": float(
                alpha_variational_current
            ),
            "stable_boundary_layer_lengths_mm": [
                float(
                    item["decay_length"]
                )
                for item in current_spectrum[
                    "stable_modes"
                ]
            ],
        },
        "equilibrium_variational_second_mode": {
            "eigenvalue_per_mm2": float(
                equilibrium_f2.eigenvalue
            ),
            "section_length_mm": float(
                equilibrium_f2.characteristic_length
            ),
            "local_generalized_alpha": float(
                alpha_variational_equilibrium
            ),
            "stable_boundary_layer_lengths_mm": [
                float(
                    item["decay_length"]
                )
                for item in equilibrium_spectrum[
                    "stable_modes"
                ]
            ],
        },
        "beam_summary": beam_summary,
        "beam_rows": result_rows,
        "FE_profile_summary": profile_summary,
        "FE_profile_rows": profile_results,
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

    plot_paths = []

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)

        current_curve = np.asarray(
            [
                current_f1.df(z)
                for z in z_curve
            ],
            dtype=float,
        )

        equilibrium_curve = np.asarray(
            [
                equilibrium.df(z)
                for z in z_curve
            ],
            dtype=float,
        )

        ax.plot(
            current_curve,
            z_curve / h,
            label="current material-informed f1'",
        )

        ax.plot(
            equilibrium_curve,
            z_curve / h,
            label="equilibrium-energy f_eq'",
        )

        ax.set_xlabel(
            "normalized primary shear-mode derivative"
        )
        ax.set_ylabel(
            "z/h"
        )
        ax.grid(
            True
        )
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "current_vs_equilibrium_primary_mode.png"
        )

        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(
            fig
        )
        plot_paths.append(
            str(path)
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        a = np.asarray(
            [
                row["a_over_h"]
                for row in result_rows
            ],
            dtype=float,
        )

        for key, label in (
            (
                "HOST_stored_current_1mode_error_relative",
                "current one-mode HOST",
            ),
            (
                "HOST_equilibrium_1mode_error_relative",
                "equilibrium-primary one-mode",
            ),
            (
                "HOST_current_variational_2mode_error_relative",
                "current variational two-mode",
            ),
            (
                "HOST_equilibrium_variational_2mode_error_relative",
                "equilibrium + variational two-mode",
            ),
        ):
            ax.plot(
                a,
                100.0
                * np.asarray(
                    [
                        row[key]
                        for row in result_rows
                    ],
                    dtype=float,
                ),
                marker="o",
                label=label,
            )

        ax.axhline(
            0.0,
            linewidth=1.0,
        )

        ax.set_xlabel(
            "a/h"
        )
        ax.set_ylabel(
            "deflection error [%]"
        )
        ax.grid(
            True
        )
        ax.legend()
        fig.tight_layout()

        path = (
            output_dir
            / "bert_equilibrium_beam_errors.png"
        )

        fig.savefig(
            path,
            dpi=180,
        )
        plt.close(
            fig
        )
        plot_paths.append(
            str(path)
        )

    except Exception as exc:
        plot_paths.append(
            "Plotting skipped: {}".format(
                exc
            )
        )

    print()
    print("=" * 126)
    print("GORDANINEJAD-BERT EQUILIBRIUM / ENERGY SHEAR AUDIT")
    print("=" * 126)
    print(
        "Neutral axis z_n [mm]                         : {:+.9e}".format(
            abd["neutral_axis_mm"]
        )
    )
    print(
        "Reduced bending stiffness D_R [N mm^2]        : {:.9e}".format(
            abd["D_R_Nmm2"]
        )
    )
    print(
        "Normalized C15 coupling core / face           : {:.6e} / {:.6e}".format(
            coupling_core,
            coupling_face,
        )
    )
    print()
    print("Paper-compatible equilibrium/energy result:")
    print(
        "  raw int(G dA) [N]                           : {:.9f}".format(
            bert["S_raw_integral_G_dA_N"]
        )
    )
    print(
        "  S_eq [N]                                    : {:.9f}".format(
            bert["S_eq_N"]
        )
    )
    print(
        "  equivalent shear coefficient K              : {:.9f}".format(
            bert["K_energy_equivalent"]
        )
    )
    print(
        "  unit-V resultant check [N]                  : {:.12f}".format(
            bert["unit_V_resultant_check_N"]
        )
    )
    print(
        "  free-surface tau_hat bottom/top             : {:+.3e} / {:+.3e}".format(
            bert["tau_bottom_per_N_mm2"],
            bert["tau_top_per_N_mm2"],
        )
    )
    print()
    print("Invariant physical-resultant shear stiffness:")
    print(
        "  current f1, diagonal Cb [N]                 : {:.9f}".format(
            current_physical_diag["S_physical_N"]
        )
    )
    print(
        "  current f1, full 2x2 Cb [N]                 : {:.9f}".format(
            current_physical_full["S_physical_N"]
        )
    )
    print(
        "  equilibrium f1, diagonal Cb [N]             : {:.9f}".format(
            equilibrium_physical_diag["S_physical_N"]
        )
    )
    print(
        "  equilibrium f1, full 2x2 Cb [N]             : {:.9f}".format(
            equilibrium_physical_full["S_physical_N"]
        )
    )
    print()
    print("Scalar ratios — keep the two meanings separate:")
    print(
        "  Bert S_eq / current-f1 physical S (diag)    : {:.9f}".format(
            scalar_comparisons[
                "Bert_S_eq_over_current_f1_physical_diag"
            ]
        )
    )
    print(
        "  Bert S_eq / current-f1 physical S (full)    : {:.9f}".format(
            scalar_comparisons[
                "Bert_S_eq_over_current_f1_physical_full"
            ]
        )
    )
    print(
        "  global generalized alpha_G                  : {:.9f}".format(
            alpha_global
        )
    )
    print(
        "  current variational local generalized alpha : {:.9f}".format(
            alpha_variational_current
        )
    )
    print(
        "  eq-primary variational generalized alpha    : {:.9f}".format(
            alpha_variational_equilibrium
        )
    )
    print()
    print("Section-mode hierarchy:")
    print(
        "  current-primary f2: lambda={:.9e}, ell={:.6f} mm".format(
            current_f2.eigenvalue,
            current_f2.characteristic_length,
        )
    )
    print(
        "  equilibrium-primary f2: lambda={:.9e}, ell={:.6f} mm".format(
            equilibrium_f2.eigenvalue,
            equilibrium_f2.characteristic_length,
        )
    )
    print(
        "  current slow BL [mm]                        : {:.6f}".format(
            current_spectrum[
                "stable_modes"
            ][0][
                "decay_length"
            ]
        )
    )
    print(
        "  equilibrium-primary slow BL [mm]            : {:.6f}".format(
            equilibrium_spectrum[
                "stable_modes"
            ][0][
                "decay_length"
            ]
        )
    )
    print()
    print(
        "{:>6s} {:>8s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
            "N_A",
            "a/h",
            "FE",
            "cur-1",
            "eq-1",
            "cur-2",
            "eq-2",
        )
    )

    for row in result_rows:
        print(
            "{:6d} {:8.3f} {:11.8f} {:11.8f} {:11.8f} {:11.8f} {:11.8f}".format(
                int(row["n_shear"]),
                float(row["a_over_h"]),
                float(row["FE_3D_abs_w_mid_mm"]),
                float(row["HOST_rebuilt_current_1mode_mm"]),
                float(row["HOST_equilibrium_1mode_mm"]),
                float(row["HOST_current_variational_2mode_mm"]),
                float(row["HOST_equilibrium_variational_2mode_mm"]),
            )
        )

    print()
    print("Beam RMS / max absolute error [%]:")
    for key in error_keys:
        values = beam_summary[key]

        print(
            "  {:52s}: {:8.4f} / {:8.4f}".format(
                key.replace(
                    "_error_relative",
                    "",
                ),
                values[
                    "RMS_percent"
                ],
                values[
                    "max_abs_percent"
                ],
            )
        )

    if profile_results:
        print()
        print(
            "FE through-thickness profile comparison at left_shear_mid:"
        )
        print(
            "  cases read                                    : {}".format(
                len(profile_results)
            )
        )
        print(
            "  mean gamma RMSE current f1 [%]                : {:.4f}".format(
                profile_summary[
                    "mean_gamma_current_RMSE_percent"
                ]
            )
        )
        print(
            "  mean gamma RMSE equilibrium f1 [%]            : {:.4f}".format(
                profile_summary[
                    "mean_gamma_equilibrium_RMSE_percent"
                ]
            )
        )
        print(
            "  mean tau RMSE current f1 [%]                  : {:.4f}".format(
                profile_summary[
                    "mean_tau_current_RMSE_percent"
                ]
            )
        )
        print(
            "  mean tau RMSE equilibrium profile [%]         : {:.4f}".format(
                profile_summary[
                    "mean_tau_equilibrium_RMSE_percent"
                ]
            )
        )
    else:
        print()
        print(
            "FE profile comparison                          : skipped "
            "(case paths/profile CSVs not available)"
        )

    print()
    print("Wrote:")
    print(
        " ",
        json_path,
    )
    print(
        " ",
        beam_csv_path,
    )
    print(
        " ",
        profile_curve_path,
    )

    if profile_results:
        print(
            " ",
            profile_csv_path,
        )

    for path in plot_paths:
        print(
            " ",
            path,
        )

    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
