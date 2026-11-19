#!/usr/bin/env python3

"""
Directional HOST buckling benchmark for homogenized TPMS sandwich cores.

Evaluates:

    - scalar Reddy HSDT,
    - scalar material-informed HOST when admissible,
    - directional material-informed HOST.

Critical resultants:

    Nx_cr
    Ny_cr
    Nbiax_cr, with Nx = Ny = Nbiax_cr.

Units are N/mm when stiffness is supplied in MPa and geometry in mm.
"""

from __future__ import annotations

import argparse

from host.buckling import (
    critical_equal_biaxial_load,
    critical_uniaxial_load_x,
    directional_critical_equal_biaxial_load,
    directional_critical_uniaxial_load_x,
    directional_critical_uniaxial_load_y,
    find_critical_buckling_mode,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.generalized_stiffness import (
    assemble_directional_generalized_stiffness,
)

from host.hsdt import (
    extract_directional_hsdt_stiffness,
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


def scalar_buckling_results(
    K11,
    a,
    b,
    max_mode,
):
    Nx = critical_uniaxial_load_x(
        K11,
        a=a,
        b=b,
        max_m=max_mode,
        max_n=max_mode,
    )

    Ny = find_critical_buckling_mode(
        K11,
        a=a,
        b=b,
        Nx_ref=0.0,
        Ny_ref=1.0,
        max_m=max_mode,
        max_n=max_mode,
    )

    Nbiax = critical_equal_biaxial_load(
        K11,
        a=a,
        b=b,
        max_m=max_mode,
        max_n=max_mode,
    )

    return Nx, Ny, Nbiax


def directional_buckling_results(
    K12,
    a,
    b,
    max_mode,
):
    Nx = directional_critical_uniaxial_load_x(
        K12,
        a=a,
        b=b,
        max_m=max_mode,
        max_n=max_mode,
    )

    Ny = directional_critical_uniaxial_load_y(
        K12,
        a=a,
        b=b,
        max_m=max_mode,
        max_n=max_mode,
    )

    Nbiax = directional_critical_equal_biaxial_load(
        K12,
        a=a,
        b=b,
        max_m=max_mode,
        max_n=max_mode,
    )

    return Nx, Ny, Nbiax


def print_result(
    label,
    result,
):
    print(
        "{:<28s}: {:14.6f} N/mm   mode ({},{})".format(
            label,
            result.load_factor,
            result.m,
            result.n,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
        help="Homogenized TPMS JSON record.",
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

    parser.add_argument(
        "--max-mode",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    record = load_tpms_homogenization_json(
        args.json_file
    )

    E_face = float(
        args.face_E
    )

    nu_face = float(
        args.face_nu
    )

    hf = float(
        args.hf
    )

    hc = float(
        args.hc
    )

    h = (
        2.0 * hf
        + hc
    )

    a = float(
        args.a
    )

    b = float(
        args.b
    )

    max_mode = int(
        args.max_mode
    )

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
        abs(
            G_xz - G_yz
        )
        / max(
            abs(G_xz),
            abs(G_yz),
        )
    )

    # --------------------------------------------------------------
    # Scalar Reddy baseline
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

    reddy_results = scalar_buckling_results(
        reddy.K11,
        a,
        b,
        max_mode,
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

    directional_results = (
        directional_buckling_results(
            K12,
            a,
            b,
            max_mode,
        )
    )

    # --------------------------------------------------------------
    # Scalar material-informed HOST, only where admissible
    # --------------------------------------------------------------

    scalar_mi_results = None

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

        scalar_mi_results = (
            scalar_buckling_results(
                scalar_mi.K11,
                a,
                b,
                max_mode,
            )
        )

    except ValueError:
        scalar_mi_results = None

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("Directional TPMS HOST buckling benchmark")
    print("=" * 78)

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
        "Orientation                 : {}".format(
            record.orientation
        )
    )

    print(
        "Plate dimensions            : {:.3f} x {:.3f} mm".format(
            a,
            b,
        )
    )

    print(
        "Sandwich thickness          : {:.3f} mm".format(
            h
        )
    )

    print()

    print(
        "G_xz hom [MPa]              : {:.6f}".format(
            G_xz
        )
    )

    print(
        "G_yz hom [MPa]              : {:.6f}".format(
            G_yz
        )
    )

    print(
        "Gmax/Gmin                   : {:.8f}".format(
            max(G_xz, G_yz)
            / min(G_xz, G_yz)
        )
    )

    print(
        "relative shear difference   : {:.4f} %".format(
            100.0
            * relative_difference
        )
    )

    print()

    print("-" * 78)
    print("Scalar Reddy HSDT")
    print("-" * 78)

    print_result(
        "Nx_cr",
        reddy_results[0],
    )

    print_result(
        "Ny_cr",
        reddy_results[1],
    )

    print_result(
        "Nbiax_cr",
        reddy_results[2],
    )

    if scalar_mi_results is not None:
        print()
        print("-" * 78)
        print("Scalar material-informed HOST")
        print("-" * 78)

        print_result(
            "Nx_cr",
            scalar_mi_results[0],
        )

        print_result(
            "Ny_cr",
            scalar_mi_results[1],
        )

        print_result(
            "Nbiax_cr",
            scalar_mi_results[2],
        )

    else:
        print()
        print("-" * 78)
        print("Scalar material-informed HOST")
        print("-" * 78)

        print(
            "Not evaluated: transverse-shear anisotropy "
            "exceeds scalar-model tolerance."
        )

    print()
    print("-" * 78)
    print("Directional material-informed HOST")
    print("-" * 78)

    print_result(
        "Nx_cr",
        directional_results[0],
    )

    print_result(
        "Ny_cr",
        directional_results[1],
    )

    print_result(
        "Nbiax_cr",
        directional_results[2],
    )

    print()
    print("-" * 78)
    print("Directional / Reddy ratios")
    print("-" * 78)

    print(
        "Nx ratio                    : {:.8f}".format(
            directional_results[0].load_factor
            / reddy_results[0].load_factor
        )
    )

    print(
        "Ny ratio                    : {:.8f}".format(
            directional_results[1].load_factor
            / reddy_results[1].load_factor
        )
    )

    print(
        "Nbiax ratio                 : {:.8f}".format(
            directional_results[2].load_factor
            / reddy_results[2].load_factor
        )
    )

    print(
        "Nx/Ny directional           : {:.8f}".format(
            directional_results[0].load_factor
            / directional_results[1].load_factor
        )
    )

    if scalar_mi_results is not None:
        print()
        print("-" * 78)
        print("Directional / scalar HOST-MI")
        print("-" * 78)

        print(
            "Nx ratio                    : {:.10f}".format(
                directional_results[0].load_factor
                / scalar_mi_results[0].load_factor
            )
        )

        print(
            "Ny ratio                    : {:.10f}".format(
                directional_results[1].load_factor
                / scalar_mi_results[1].load_factor
            )
        )

        print(
            "Nbiax ratio                 : {:.10f}".format(
                directional_results[2].load_factor
                / scalar_mi_results[2].load_factor
            )
        )


if __name__ == "__main__":
    main()