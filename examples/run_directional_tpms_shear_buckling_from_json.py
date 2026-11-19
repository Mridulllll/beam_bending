#!/usr/bin/env python3

"""
Directional multimode pure-shear buckling benchmark for a TPMS sandwich.

Evaluates convergence with increasing Navier modal truncation and compares:

    - scalar Reddy HSDT,
    - scalar material-informed HOST when admissible,
    - directional material-informed HOST.

Pure shear reference loading:

    Nx_ref  = 0
    Ny_ref  = 0
    Nxy_ref = 1

Hence the critical eigenvalue directly equals |Nxy_cr| in N/mm.
"""

from __future__ import annotations

import argparse

import numpy as np

from host.constitutive import isotropic_stiffness_3d

from host.generalized_stiffness import (
    assemble_directional_generalized_stiffness,
)

from host.hsdt import (
    extract_directional_hsdt_stiffness,
)

from host.multimode_buckling import (
    solve_directional_multimode_buckling,
    solve_multimode_buckling,
)

from host.sandwich import (
    build_three_layer_sandwich,
)

from host.sandwich_warping import (
    material_informed_directional_generalized_warping,
    material_informed_generalized_warping,
)

from host.tpms_core import (
    representative_transverse_shear_modulus,
)

from host.tpms_io import (
    load_tpms_homogenization_json,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def solve_scalar(
    K11,
    a,
    b,
    max_mode,
):
    return solve_multimode_buckling(
        K11,
        a=a,
        b=b,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
        max_m=max_mode,
        max_n=max_mode,
    )


def solve_directional(
    K12,
    a,
    b,
    max_mode,
):
    return solve_directional_multimode_buckling(
        K12,
        a=a,
        b=b,
        Nx_ref=0.0,
        Ny_ref=0.0,
        Nxy_ref=1.0,
        max_m=max_mode,
        max_n=max_mode,
    )


def modal_participation(result, count=8):
    values = np.abs(
        np.asarray(
            result.eigenvector,
            dtype=float,
        )
    )

    order = np.argsort(values)[::-1]

    entries = []

    for index in order[:count]:
        mode = result.modes[
            int(index)
        ]

        entries.append(
            (
                mode.m,
                mode.n,
                float(values[index]),
            )
        )

    return entries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
    )

    parser.add_argument(
        "--face-E",
        type=float,
        default=70000.0,
    )

    parser.add_argument(
        "--face-nu",
        type=float,
        default=0.33,
    )

    parser.add_argument(
        "--hf",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--hc",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--a",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--b",
        type=float,
        default=100.0,
    )

    args = parser.parse_args()

    record = load_tpms_homogenization_json(
        args.json_file
    )

    E_face = float(args.face_E)
    nu_face = float(args.face_nu)

    hf = float(args.hf)
    hc = float(args.hc)

    h = 2.0 * hf + hc

    a = float(args.a)
    b = float(args.b)

    C_face = isotropic_stiffness_3d(
        E=E_face,
        nu=nu_face,
    )

    G_face = (
        E_face
        / (
            2.0
            * (1.0 + nu_face)
        )
    )

    G_xz = float(
        record.core.G_xz
    )

    G_yz = float(
        record.core.G_yz
    )

    relative_difference = (
        abs(G_xz - G_yz)
        / max(
            abs(G_xz),
            abs(G_yz),
        )
    )

    # --------------------------------------------------------------
    # Reddy
    # --------------------------------------------------------------

    reddy = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=record.C_hom,
        C_top=C_face,
        hf_bottom=hf,
        hc=hc,
        hf_top=hf,
        warping=reddy_baseline_generalized_warping(
            h
        ),
    )

    # --------------------------------------------------------------
    # Directional material-informed HOST
    # --------------------------------------------------------------

    directional_warping = (
        material_informed_directional_generalized_warping(
            h=h,
            hc=hc,
            G_face_xz=G_face,
            G_face_yz=G_face,
            G_core_xz=G_xz,
            G_core_yz=G_yz,
        )
    )

    K15 = assemble_directional_generalized_stiffness(
        reddy.layers,
        directional_warping,
        quadrature_order=10,
    )

    K12 = extract_directional_hsdt_stiffness(
        K15
    )

    # --------------------------------------------------------------
    # Scalar material-informed HOST where admissible
    # --------------------------------------------------------------

    scalar_mi = None

    try:
        G_rep = (
            representative_transverse_shear_modulus(
                record.core,
                relative_tolerance=0.05,
            )
        )

        scalar_warping = (
            material_informed_generalized_warping(
                h=h,
                hc=hc,
                G_face=G_face,
                G_core=G_rep,
            )
        )

        scalar_mi = build_three_layer_sandwich(
            C_bottom=C_face,
            C_core=record.C_hom,
            C_top=C_face,
            hf_bottom=hf,
            hc=hc,
            hf_top=hf,
            warping=scalar_warping,
        )

    except ValueError:
        scalar_mi = None

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print()
    print("=" * 84)
    print("Directional TPMS pure-shear multimode buckling")
    print("=" * 84)

    print(
        "Topology                    : {}".format(
            record.topology
        )
    )

    print(
        "RD target                   : {:.6f}".format(
            record.relative_density_target
        )
    )

    print(
        "RD actual                   : {:.6f}".format(
            record.relative_density_actual
        )
    )

    print(
        "Plate                       : {:.3f} x {:.3f} mm".format(
            a,
            b,
        )
    )

    print(
        "Sandwich thickness          : {:.3f} mm".format(
            h
        )
    )

    print(
        "G_xz [MPa]                  : {:.6f}".format(
            G_xz
        )
    )

    print(
        "G_yz [MPa]                  : {:.6f}".format(
            G_yz
        )
    )

    print(
        "relative shear difference   : {:.4f} %".format(
            100.0 * relative_difference
        )
    )

    # --------------------------------------------------------------
    # Directional convergence
    # --------------------------------------------------------------

    truncations = [
        4,
        6,
        8,
        10,
        12,
        16,
    ]

    directional_results = []

    print()
    print("-" * 84)
    print("Directional HOST modal convergence")
    print("-" * 84)

    print(
        "{:>8s} {:>18s} {:>14s} {:>16s}".format(
            "M=N",
            "Nxy_cr [N/mm]",
            "dominant",
            "change [%]",
        )
    )

    previous = None

    for size in truncations:
        result = solve_directional(
            K12,
            a,
            b,
            size,
        )

        directional_results.append(
            result
        )

        if previous is None:
            change = np.nan
            change_text = "-"
        else:
            change = (
                100.0
                * abs(
                    result.load_factor
                    - previous.load_factor
                )
                / abs(
                    result.load_factor
                )
            )

            change_text = "{:.6f}".format(
                change
            )

        print(
            "{:8d} {:18.6f} {:>14s} {:>16s}".format(
                size,
                result.load_factor,
                "({},{})".format(
                    result.dominant_mode.m,
                    result.dominant_mode.n,
                ),
                change_text,
            )
        )

        previous = result

    final_directional = (
        directional_results[-1]
    )

    # --------------------------------------------------------------
    # Final 16 x 16 comparison
    # --------------------------------------------------------------

    reddy_result = solve_scalar(
        reddy.K11,
        a,
        b,
        16,
    )

    if scalar_mi is not None:
        scalar_mi_result = solve_scalar(
            scalar_mi.K11,
            a,
            b,
            16,
        )
    else:
        scalar_mi_result = None

    print()
    print("-" * 84)
    print("16 x 16 comparison")
    print("-" * 84)

    print(
        "Reddy Nxy_cr [N/mm]         : {:.6f}".format(
            reddy_result.load_factor
        )
    )

    print(
        "Reddy dominant mode         : ({},{})".format(
            reddy_result.dominant_mode.m,
            reddy_result.dominant_mode.n,
        )
    )

    if scalar_mi_result is not None:
        print(
            "Scalar HOST-MI [N/mm]      : {:.6f}".format(
                scalar_mi_result.load_factor
            )
        )

        print(
            "Scalar HOST dominant       : ({},{})".format(
                scalar_mi_result.dominant_mode.m,
                scalar_mi_result.dominant_mode.n,
            )
        )
    else:
        print(
            "Scalar HOST-MI              : not admissible"
        )

    print(
        "Directional HOST [N/mm]     : {:.6f}".format(
            final_directional.load_factor
        )
    )

    print(
        "Directional dominant        : ({},{})".format(
            final_directional.dominant_mode.m,
            final_directional.dominant_mode.n,
        )
    )

    print()

    print(
        "Directional/Reddy           : {:.8f}".format(
            final_directional.load_factor
            / reddy_result.load_factor
        )
    )

    if scalar_mi_result is not None:
        print(
            "Directional/scalar HOST-MI : {:.10f}".format(
                final_directional.load_factor
                / scalar_mi_result.load_factor
            )
        )

    print()
    print("-" * 84)
    print("Largest directional modal amplitudes at 16 x 16")
    print("-" * 84)

    for m, n, amplitude in modal_participation(
        final_directional,
        count=10,
    ):
        print(
            "mode ({:2d},{:2d}) : {:.8f}".format(
                m,
                n,
                amplitude,
            )
        )


if __name__ == "__main__":
    main()