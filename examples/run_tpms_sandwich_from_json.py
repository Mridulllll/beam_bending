#!/usr/bin/env python3

"""
First real homogenized-TPMS HOST sandwich calculation.

Usage
-----

python examples/run_tpms_sandwich_from_json.py path/to/result.json

The same physical sandwich is evaluated using:

    1. global Reddy HSDT,
    2. material-informed HOST warping,
    3. energy-equivalent FSDT reference.

The COMPLETE homogenized TPMS stiffness tensor is retained in
the HSDT sandwich constitutive integration.
"""

from __future__ import annotations

import argparse

import numpy as np

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.fsdt_reference import (
    build_energy_equivalent_fsdt_section,
    center_deflection_uniform_pressure_fsdt,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
        help="Homogenized TPMS JSON file.",
    )

    parser.add_argument(
        "--face-E",
        type=float,
        default=70000.0,
        help="Face Young's modulus in MPa.",
    )

    parser.add_argument(
        "--face-nu",
        type=float,
        default=0.33,
        help="Face Poisson ratio.",
    )

    parser.add_argument(
        "--hf",
        type=float,
        default=0.5,
        help="Each face-sheet thickness in mm.",
    )

    parser.add_argument(
        "--hc",
        type=float,
        default=10.0,
        help="Core thickness in mm.",
    )

    parser.add_argument(
        "--a",
        type=float,
        default=100.0,
        help="Plate length a in mm.",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=100.0,
        help="Plate width b in mm.",
    )

    parser.add_argument(
        "--pressure",
        type=float,
        default=0.01,
        help="Uniform pressure in MPa (N/mm^2).",
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
    # Global Reddy baseline
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

    mi_warping = (
        material_informed_generalized_warping(
            h=h,
            hc=hc,
            G_face=G_face,
            G_core=G_core,
        )
    )

    material_informed = (
        build_three_layer_sandwich(
            C_bottom=C_face,
            C_core=record.C_hom,
            C_top=C_face,
            hf_bottom=hf,
            hc=hc,
            hf_top=hf,
            warping=mi_warping,
        )
    )

    # --------------------------------------------------------------
    # Energy-equivalent FSDT reference
    # --------------------------------------------------------------

    fsdt = (
        build_energy_equivalent_fsdt_section(
            reddy.layers
        )
    )

    common = dict(
        q0=float(
            args.pressure
        ),
        a=float(
            args.a
        ),
        b=float(
            args.b
        ),
        max_mode=21,
    )

    w_reddy = (
        center_deflection_uniform_pressure_hsdt(
            reddy.K11,
            **common,
        )
    )

    w_mi = (
        center_deflection_uniform_pressure_hsdt(
            material_informed.K11,
            **common,
        )
    )

    w_fsdt = (
        center_deflection_uniform_pressure_fsdt(
            fsdt,
            **common,
        )
    )

    print()
    print("=" * 72)
    print("TPMS HOST sandwich benchmark")
    print("=" * 72)

    print(
        "Topology              : {}".format(
            record.topology
        )
    )

    print(
        "RD target             : {:.6f}".format(
            record.relative_density_target
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

    print()

    print(
        "G_xz hom [MPa]        : {:.6f}".format(
            record.core.G_xz
        )
    )

    print(
        "G_yz hom [MPa]        : {:.6f}".format(
            record.core.G_yz
        )
    )

    print(
        "G_rep hom [MPa]       : {:.6f}".format(
            G_core
        )
    )

    print(
        "G_core/G_face         : {:.6e}".format(
            G_core / G_face
        )
    )

    print()

    print(
        "w FSDT [mm]           : {:.8e}".format(
            w_fsdt
        )
    )

    print(
        "w Reddy [mm]          : {:.8e}".format(
            w_reddy
        )
    )

    print(
        "w HOST-MI [mm]        : {:.8e}".format(
            w_mi
        )
    )

    print()

    print(
        "Reddy/FSDT            : {:.8f}".format(
            w_reddy / w_fsdt
        )
    )

    print(
        "HOST-MI/FSDT          : {:.8f}".format(
            w_mi / w_fsdt
        )
    )

    print(
        "HOST-MI/Reddy         : {:.8f}".format(
            w_mi / w_reddy
        )
    )


if __name__ == "__main__":
    main()