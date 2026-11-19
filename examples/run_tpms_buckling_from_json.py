#!/usr/bin/env python3

"""
Run HOST linear buckling for one homogenized TPMS sandwich JSON.

The same physical sandwich is evaluated with

    1. global Reddy HSDT
    2. material-informed HOST warping

for

    - uniaxial x compression
    - uniaxial y compression
    - equal biaxial compression

The complete homogenized TPMS stiffness tensor is retained.
"""

from __future__ import annotations

import argparse

from host.buckling import (
    critical_equal_biaxial_load,
    critical_uniaxial_load_x,
    find_critical_buckling_mode,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.sandwich import (
    build_three_layer_sandwich,
)

from host.sandwich_warping import (
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


def critical_uniaxial_load_y(
    K_hsdt,
    a,
    b,
    *,
    max_m=10,
    max_n=10,
):
    return find_critical_buckling_mode(
        K_hsdt,
        a,
        b,
        Nx_ref=0.0,
        Ny_ref=1.0,
        max_m=max_m,
        max_n=max_n,
    )


def show_result(
    label,
    result,
):
    print(
        "{:<20s} {:>14.6f}   mode=({:d},{:d})".format(
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
    )

    parser.add_argument(
        "--face-E",
        type=float,
        default=70000.0,
        help="Face Young's modulus [MPa].",
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
        help="Each face thickness [mm].",
    )

    parser.add_argument(
        "--hc",
        type=float,
        default=10.0,
        help="Core thickness [mm].",
    )

    parser.add_argument(
        "--a",
        type=float,
        default=100.0,
        help="Plate dimension a [mm].",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=100.0,
        help="Plate dimension b [mm].",
    )

    parser.add_argument(
        "--max-m",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-n",
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

    G_core = (
        representative_transverse_shear_modulus(
            record.core,
            relative_tolerance=0.05,
        )
    )

    # --------------------------------------------------------------
    # Reddy baseline
    # --------------------------------------------------------------

    reddy = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=record.C_hom,
        C_top=C_face,
        hf_bottom=hf,
        hc=hc,
        hf_top=hf,
        warping=(
            reddy_baseline_generalized_warping(
                h
            )
        ),
    )

    # --------------------------------------------------------------
    # Material-informed HOST
    # --------------------------------------------------------------

    mi = build_three_layer_sandwich(
        C_bottom=C_face,
        C_core=record.C_hom,
        C_top=C_face,
        hf_bottom=hf,
        hc=hc,
        hf_top=hf,
        warping=(
            material_informed_generalized_warping(
                h=h,
                hc=hc,
                G_face=G_face,
                G_core=G_core,
            )
        ),
    )

    common = dict(
        a=float(
            args.a
        ),
        b=float(
            args.b
        ),
        max_m=int(
            args.max_m
        ),
        max_n=int(
            args.max_n
        ),
    )

    # --------------------------------------------------------------
    # Reddy
    # --------------------------------------------------------------

    reddy_x = critical_uniaxial_load_x(
        reddy.K11,
        **common,
    )

    reddy_y = critical_uniaxial_load_y(
        reddy.K11,
        **common,
    )

    reddy_bi = critical_equal_biaxial_load(
        reddy.K11,
        **common,
    )

    # --------------------------------------------------------------
    # HOST-MI
    # --------------------------------------------------------------

    mi_x = critical_uniaxial_load_x(
        mi.K11,
        **common,
    )

    mi_y = critical_uniaxial_load_y(
        mi.K11,
        **common,
    )

    mi_bi = critical_equal_biaxial_load(
        mi.K11,
        **common,
    )

    print()
    print("=" * 76)
    print("TPMS HOST buckling benchmark")
    print("=" * 76)

    print(
        "Topology              : {}".format(
            record.topology
        )
    )

    print(
        "RD actual             : {:.6f}".format(
            record.relative_density_actual
        )
    )

    print(
        "Orientation           : {}".format(
            record.orientation
        )
    )

    print(
        "Gxz / Gyz             : {:.6f} / {:.6f} MPa".format(
            record.core.G_xz,
            record.core.G_yz,
        )
    )

    print(
        "G anisotropy          : {:.6f}".format(
            record.core.shear_anisotropy_xz_yz
        )
    )

    print()
    print(
        "Units of Ncr          : N/mm"
    )

    print()
    print("Reddy HSDT")
    print("-" * 76)

    show_result(
        "Nx_cr",
        reddy_x,
    )

    show_result(
        "Ny_cr",
        reddy_y,
    )

    show_result(
        "Nbiax_cr",
        reddy_bi,
    )

    print()
    print("HOST material-informed")
    print("-" * 76)

    show_result(
        "Nx_cr",
        mi_x,
    )

    show_result(
        "Ny_cr",
        mi_y,
    )

    show_result(
        "Nbiax_cr",
        mi_bi,
    )

    print()
    print("HOST-MI / Reddy")
    print("-" * 76)

    print(
        "Nx ratio              : {:.8f}".format(
            mi_x.load_factor
            / reddy_x.load_factor
        )
    )

    print(
        "Ny ratio              : {:.8f}".format(
            mi_y.load_factor
            / reddy_y.load_factor
        )
    )

    print(
        "Biaxial ratio         : {:.8f}".format(
            mi_bi.load_factor
            / reddy_bi.load_factor
        )
    )


if __name__ == "__main__":
    main()