#!/usr/bin/env python3

"""
TPMS sandwich-beam bending benchmark.

Compares

    1. Euler-Bernoulli beam theory,
    2. Reddy higher-order beam theory,
    3. material-informed HOST beam theory.

The complete homogenized 6x6 TPMS stiffness tensor is reduced
specifically for a beam with free inactive stresses.

All specimen and fixture dimensions are independent parameters.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from host.beam import (
    SandwichBeamGeometry,
    build_three_layer_host_beam,
    free_beam_constitutive,
)

from host.beam_bending import (
    condense_zero_axial_force,
)

from host.beam_point_loads import (
    solve_simply_supported_point_loads,
    symmetric_four_point_bending_loads,
    symmetric_support_positions,
    three_point_bending_loads,
)

from host.constitutive import (
    isotropic_stiffness_3d,
)

from host.sandwich_warping import (
    material_informed_generalized_warping,
)

from host.tpms_io import (
    load_tpms_homogenization_json,
)

from host.warping import (
    reddy_baseline_generalized_warping,
)


def classical_bending_stiffness(
    K4,
):
    """
    Euler-Bernoulli bending stiffness after exact N=0
    axial condensation.

        D = K11 - K10*K01/K00
    """

    K4 = np.asarray(
        K4,
        dtype=float,
    )

    return float(
        K4[1, 1]
        - (
            K4[1, 0]
            * K4[0, 1]
            / K4[0, 0]
        )
    )


def eb_three_point_midspan(
    *,
    total_load,
    support_span,
    D,
):
    """
    Simply-supported central point load:

        delta = P S^3 / (48 D).
    """

    P = float(
        total_load
    )

    S = float(
        support_span
    )

    return (
        P
        * S**3
        / (
            48.0
            * D
        )
    )


def eb_four_point_midspan(
    *,
    total_load,
    support_span,
    load_span,
    D,
):
    """
    Symmetric four-point bending.

    Total applied load P is divided equally between both rollers.

    Distance from support to nearest loading roller:

        c = (S - S_L)/2.

    Midspan Euler-Bernoulli deflection:

        delta =
        P*c*(3*S^2 - 4*c^2)/(48*D).

    This reduces to the standard central point-load expression
    when S_L -> 0.
    """

    P = float(
        total_load
    )

    S = float(
        support_span
    )

    SL = float(
        load_span
    )

    c = 0.5 * (
        S - SL
    )

    return (
        P
        * c
        * (
            3.0 * S**2
            - 4.0 * c**2
        )
        / (
            48.0
            * D
        )
    )


def solve_beam(
    K3,
    *,
    specimen_length,
    support_span,
    total_load,
    test_type,
    load_span,
):
    L = float(
        specimen_length
    )

    xA, xB = symmetric_support_positions(
        specimen_length=L,
        support_span=support_span,
    )

    if test_type == "three-point":
        loads = three_point_bending_loads(
            load=total_load,
            load_position=0.5 * L,
        )

    elif test_type == "four-point":
        loads = symmetric_four_point_bending_loads(
            total_load=total_load,
            specimen_length=L,
            load_span=load_span,
        )

    else:
        raise ValueError(
            "Unsupported test type."
        )

    solution = solve_simply_supported_point_loads(
        K3,
        length=L,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
    )

    return (
        solution,
        loads,
        xA,
        xB,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_file",
    )

    parser.add_argument(
        "--axis",
        choices=(
            "x",
            "y",
        ),
        default="x",
    )

    parser.add_argument(
        "--test",
        choices=(
            "three-point",
            "four-point",
        ),
        default="four-point",
    )

    parser.add_argument(
        "--length",
        type=float,
        default=240.0,
    )

    parser.add_argument(
        "--width",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--core-height",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--face-bottom",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--face-top",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--support-span",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--load-span",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--load",
        type=float,
        default=1000.0,
        help="Total applied transverse load [N].",
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

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Geometry validation
    # --------------------------------------------------------------

    L = float(
        args.length
    )

    width = float(
        args.width
    )

    hc = float(
        args.core_height
    )

    hf_bottom = float(
        args.face_bottom
    )

    hf_top = float(
        args.face_top
    )

    support_span = float(
        args.support_span
    )

    load_span = float(
        args.load_span
    )

    total_load = float(
        args.load
    )

    if support_span > L:
        raise ValueError(
            "support span cannot exceed specimen length."
        )

    if args.test == "four-point":
        if load_span < 0.0:
            raise ValueError(
                "load span cannot be negative."
            )

        if load_span >= support_span:
            raise ValueError(
                "For standard four-point bending, "
                "load span must be smaller than support span."
            )

    # Current material-informed sandwich warping is symmetric.
    if not np.isclose(
        hf_bottom,
        hf_top,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "The current material-informed beam warping "
            "assumes equal top and bottom face thicknesses. "
            "Use equal faces for this benchmark."
        )

    geometry = SandwichBeamGeometry(
        length=L,
        width=width,
        core_height=hc,
        face_bottom=hf_bottom,
        face_top=hf_top,
    )

    h = geometry.total_height

    # --------------------------------------------------------------
    # Materials
    # --------------------------------------------------------------

    record = load_tpms_homogenization_json(
        args.json_file
    )

    C_core = np.asarray(
        record.C_hom,
        dtype=float,
    )

    C_face = isotropic_stiffness_3d(
        E=float(args.face_E),
        nu=float(args.face_nu),
    )

    # Beam-specific free-stress reduction.
    Cb_core = free_beam_constitutive(
        C_core,
        axis=args.axis,
    )

    Cb_face = free_beam_constitutive(
        C_face,
        axis=args.axis,
    )

    E_core_beam = float(
        Cb_core[0, 0]
    )

    G_core_beam = float(
        Cb_core[1, 1]
    )

    E_face_beam = float(
        Cb_face[0, 0]
    )

    G_face_beam = float(
        Cb_face[1, 1]
    )

    coupling = float(
        abs(
            Cb_core[0, 1]
        )
        / math.sqrt(
            Cb_core[0, 0]
            * Cb_core[1, 1]
        )
    )

    # --------------------------------------------------------------
    # Reddy beam
    # --------------------------------------------------------------

    reddy_warping = (
        reddy_baseline_generalized_warping(
            h
        )
    )

    reddy_beam = build_three_layer_host_beam(
        C_bottom=C_face,
        C_core=C_core,
        C_top=C_face,
        geometry=geometry,
        axis=args.axis,
        warping=reddy_warping,
        quadrature_order=12,
    )

    reddy_reduced = condense_zero_axial_force(
        reddy_beam.K4
    )

    # --------------------------------------------------------------
    # Material-informed HOST beam
    # --------------------------------------------------------------

    host_warping = (
        material_informed_generalized_warping(
            h=h,
            hc=hc,
            G_face=G_face_beam,
            G_core=G_core_beam,
        )
    )

    host_beam = build_three_layer_host_beam(
        C_bottom=C_face,
        C_core=C_core,
        C_top=C_face,
        geometry=geometry,
        axis=args.axis,
        warping=host_warping,
        quadrature_order=12,
    )

    host_reduced = condense_zero_axial_force(
        host_beam.K4
    )

    # --------------------------------------------------------------
    # Classical bending stiffness
    # --------------------------------------------------------------

    D = classical_bending_stiffness(
        reddy_beam.K4
    )

    D_host = classical_bending_stiffness(
        host_beam.K4
    )

    np.testing.assert_allclose(
        D_host,
        D,
        rtol=1.0e-12,
        atol=1.0e-8,
    )

    # --------------------------------------------------------------
    # Solve Reddy and HOST
    # --------------------------------------------------------------

    reddy_solution, loads, xA, xB = solve_beam(
        reddy_reduced.K3,
        specimen_length=L,
        support_span=support_span,
        total_load=total_load,
        test_type=args.test,
        load_span=load_span,
    )

    host_solution, _, _, _ = solve_beam(
        host_reduced.K3,
        specimen_length=L,
        support_span=support_span,
        total_load=total_load,
        test_type=args.test,
        load_span=load_span,
    )

    x_mid = 0.5 * L

    reddy_mid = reddy_solution.state(
        x_mid
    )

    host_mid = host_solution.state(
        x_mid
    )

    w_reddy = abs(
        float(
            reddy_mid[0]
        )
    )

    w_host = abs(
        float(
            host_mid[0]
        )
    )

    # --------------------------------------------------------------
    # Euler-Bernoulli reference
    # --------------------------------------------------------------

    if args.test == "three-point":
        w_eb = eb_three_point_midspan(
            total_load=total_load,
            support_span=support_span,
            D=D,
        )

    else:
        w_eb = eb_four_point_midspan(
            total_load=total_load,
            support_span=support_span,
            load_span=load_span,
            D=D,
        )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    print()
    print("=" * 82)
    print("TPMS sandwich-beam bending benchmark")
    print("=" * 82)

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
        "Beam axis                   : {}".format(
            args.axis
        )
    )

    print(
        "Test                        : {}".format(
            args.test
        )
    )

    print()

    print(
        "Specimen L x b              : {:.3f} x {:.3f} mm".format(
            L,
            width,
        )
    )

    print(
        "Core height                 : {:.3f} mm".format(
            hc
        )
    )

    print(
        "Face thicknesses            : {:.3f} / {:.3f} mm".format(
            hf_bottom,
            hf_top,
        )
    )

    print(
        "Total sandwich height       : {:.3f} mm".format(
            h
        )
    )

    print(
        "L / h                       : {:.5f}".format(
            L / h
        )
    )

    print(
        "Support span                : {:.3f} mm".format(
            support_span
        )
    )

    print(
        "Support span / h            : {:.5f}".format(
            support_span / h
        )
    )

    if args.test == "four-point":
        print(
            "Load span                   : {:.3f} mm".format(
                load_span
            )
        )

    print(
        "Total applied load          : {:.3f} N".format(
            total_load
        )
    )

    print()

    print("-" * 82)
    print("Beam-reduced material properties")
    print("-" * 82)

    print(
        "Core E_long [MPa]           : {:.6f}".format(
            E_core_beam
        )
    )

    print(
        "Core G_s [MPa]              : {:.6f}".format(
            G_core_beam
        )
    )

    print(
        "Face E_long [MPa]           : {:.6f}".format(
            E_face_beam
        )
    )

    print(
        "Face G_s [MPa]              : {:.6f}".format(
            G_face_beam
        )
    )

    print(
        "normalized normal/shear cpl : {:.8e}".format(
            coupling
        )
    )

    print()

    print("-" * 82)
    print("Fixture")
    print("-" * 82)

    print(
        "Supports                    : {:.3f}, {:.3f} mm".format(
            xA,
            xB,
        )
    )

    print(
        "Load positions              : {}".format(
            ", ".join(
                "{:.3f}".format(
                    load.x
                )
                for load in loads
            )
        )
    )

    print(
        "Reddy reactions             : {:.6f}, {:.6f} N".format(
            reddy_solution.reactions.left,
            reddy_solution.reactions.right,
        )
    )

    print()

    print("-" * 82)
    print("Midspan response")
    print("-" * 82)

    print(
        "Euler-Bernoulli w [mm]      : {:.10e}".format(
            w_eb
        )
    )

    print(
        "Reddy w [mm]                : {:.10e}".format(
            w_reddy
        )
    )

    print(
        "HOST-MI w [mm]              : {:.10e}".format(
            w_host
        )
    )

    print()

    print(
        "Reddy / EB                  : {:.8f}".format(
            w_reddy / w_eb
        )
    )

    print(
        "HOST / EB                   : {:.8f}".format(
            w_host / w_eb
        )
    )

    print(
        "HOST / Reddy                : {:.8f}".format(
            w_host / w_reddy
        )
    )

    print()

    print(
        "EB compliance [mm/N]        : {:.10e}".format(
            w_eb / total_load
        )
    )

    print(
        "Reddy compliance [mm/N]     : {:.10e}".format(
            w_reddy / total_load
        )
    )

    print(
        "HOST compliance [mm/N]      : {:.10e}".format(
            w_host / total_load
        )
    )

    print()

    print(
        "b * HOST compliance         : {:.10e} mm^2/N".format(
            width
            * w_host
            / total_load
        )
    )

    print()

    print(
        "HOST midspan M [Nmm]        : {:.8e}".format(
            host_mid[3]
        )
    )

    print(
        "HOST midspan V [N]          : {:.8e}".format(
            host_mid[4]
        )
    )

    print(
        "HOST midspan H              : {:.8e}".format(
            host_mid[5]
        )
    )


if __name__ == "__main__":
    main()