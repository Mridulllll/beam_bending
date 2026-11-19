#!/usr/bin/env python3

"""
Directional material-informed HOST benchmark for a homogenized TPMS core.

This example is intended specifically for cores for which

    G_xz != G_yz

and therefore a single scalar material-informed warping function is
not appropriate.

The same physical sandwich is evaluated using:

    1. global scalar Reddy HSDT,
    2. directional material-informed HOST with f_x(z) and f_y(z),
    3. energy-equivalent FSDT reference.

The complete homogenized TPMS stiffness tensor is retained in the
sandwich constitutive integration.
"""

from __future__ import annotations

import argparse

from host.constitutive import isotropic_stiffness_3d

from host.fsdt_reference import (
    build_energy_equivalent_fsdt_section,
    center_deflection_uniform_pressure_fsdt,
)

from host.generalized_stiffness import (
    assemble_directional_generalized_stiffness,
)

from host.hsdt import (
    extract_directional_hsdt_stiffness,
)

from host.navier import (
    center_deflection_uniform_pressure_hsdt,
    center_deflection_uniform_pressure_directional_hsdt,
)

from host.sandwich import build_three_layer_sandwich

from host.sandwich_warping import (
    material_informed_directional_generalized_warping,
)

from host.tpms_io import load_tpms_homogenization_json

from host.warping import reddy_baseline_generalized_warping


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

    parser.add_argument(
        "--max-mode",
        type=int,
        default=21,
        help="Maximum Navier mode number.",
    )

    args = parser.parse_args()

    record = load_tpms_homogenization_json(
        args.json_file
    )

    E_face = float(args.face_E)
    nu_face = float(args.face_nu)

    hf = float(args.hf)
    hc = float(args.hc)

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

    G_xz = float(
        record.core.G_xz
    )

    G_yz = float(
        record.core.G_yz
    )

    anisotropy = (
        max(G_xz, G_yz)
        / min(G_xz, G_yz)
    )

    relative_difference = (
        abs(G_xz - G_yz)
        / max(abs(G_xz), abs(G_yz))
    )

    # --------------------------------------------------------------
    # Scalar global Reddy baseline
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
    # Directional material-informed warping
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

    # The scalar sandwich builder has already constructed the exact
    # physical layer geometry and sigma_zz=0 reduced constitutive
    # matrices. Reuse those layers and only replace the generalized
    # through-thickness operator by the directional one.
    K15 = assemble_directional_generalized_stiffness(
        reddy.layers,
        directional_warping,
        quadrature_order=10,
    )

    K12 = extract_directional_hsdt_stiffness(
        K15
    )

    # --------------------------------------------------------------
    # Energy-equivalent FSDT reference
    # --------------------------------------------------------------

    fsdt = build_energy_equivalent_fsdt_section(
        reddy.layers
    )

    common = dict(
        q0=float(args.pressure),
        a=float(args.a),
        b=float(args.b),
        max_mode=int(args.max_mode),
    )

    # --------------------------------------------------------------
    # Solutions
    # --------------------------------------------------------------

    w_reddy = center_deflection_uniform_pressure_hsdt(
        reddy.K11,
        **common,
    )

    w_directional = (
        center_deflection_uniform_pressure_directional_hsdt(
            K12,
            **common,
        )
    )

    w_fsdt = center_deflection_uniform_pressure_fsdt(
        fsdt,
        **common,
    )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print()
    print("=" * 76)
    print("Directional TPMS HOST sandwich benchmark")
    print("=" * 76)

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
        "G_xz/G_yz anisotropy        : {:.8f}".format(
            anisotropy
        )
    )

    print(
        "relative shear difference   : {:.4f} %".format(
            100.0 * relative_difference
        )
    )

    print(
        "G_xz/G_face                 : {:.6e}".format(
            G_xz / G_face
        )
    )

    print(
        "G_yz/G_face                 : {:.6e}".format(
            G_yz / G_face
        )
    )

    print()

    print(
        "Directional K15 shape       : {}".format(
            K15.shape
        )
    )

    print(
        "Directional HSDT K12 shape  : {}".format(
            K12.shape
        )
    )

    print()

    print(
        "w FSDT [mm]                 : {:.8e}".format(
            w_fsdt
        )
    )

    print(
        "w scalar Reddy [mm]         : {:.8e}".format(
            w_reddy
        )
    )

    print(
        "w directional HOST [mm]     : {:.8e}".format(
            w_directional
        )
    )

    print()

    print(
        "Reddy/FSDT                  : {:.8f}".format(
            w_reddy / w_fsdt
        )
    )

    print(
        "Directional HOST/FSDT       : {:.8f}".format(
            w_directional / w_fsdt
        )
    )

    print(
        "Directional HOST/Reddy      : {:.8f}".format(
            w_directional / w_reddy
        )
    )

    print()

    if relative_difference > 0.05:
        print(
            "Scalar HOST-MI              : intentionally not evaluated"
        )
        print(
            "Reason                      : transverse-shear anisotropy "
            "exceeds 5%"
        )
    else:
        print(
            "Scalar HOST-MI              : admissible, but directional "
            "formulation used here"
        )


if __name__ == "__main__":
    main()