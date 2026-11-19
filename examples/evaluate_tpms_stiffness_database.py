#!/usr/bin/env python3

"""
Evaluate all homogenized TPMS query JSON files through the HOST
sandwich model.

For every state this script reports:

    topology
    load case
    iteration tag
    relative density
    homogenized engineering constants
    transverse-shear anisotropy
    FSDT deflection
    Reddy-HSDT deflection
    material-informed HOST deflection

The complete homogenized C tensor is used in the HOST stiffness
integration.
"""

from pathlib import Path
import json

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


QUERY_DIR = Path(
    r"C:\AbaqusScratch\UCs\Shells\TPMS_mapped_thickness"
    r"\_stiffness_database\queries"
)


def main():
    # --------------------------------------------------------------
    # Sandwich benchmark
    # --------------------------------------------------------------

    E_face = 70000.0
    nu_face = 0.33

    hf = 0.5
    hc = 10.0
    h = 2.0 * hf + hc

    a = 100.0
    b = 100.0

    q0 = 0.01

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

    rows = []

    files = sorted(
        QUERY_DIR.glob("*.json")
    )

    if not files:
        raise RuntimeError(
            "No JSON files found in {}".format(
                QUERY_DIR
            )
        )

    print()
    print(
        "{:>9s} {:>5s} {:>11s} {:>8s} "
        "{:>9s} {:>9s} {:>8s} "
        "{:>11s} {:>11s} {:>11s}".format(
            "Topology",
            "RD",
            "load",
            "itag",
            "Gxz",
            "Gyz",
            "G-ratio",
            "FSDT [um]",
            "Reddy [um]",
            "HOST [um]",
        )
    )

    print("-" * 110)

    for path in files:
        record = (
            load_tpms_homogenization_json(
                path
            )
        )

        data = record.metadata

        load_case = str(
            data.get(
                "optimization_load_case",
                "unknown",
            )
        )

        iteration = int(
            data.get(
                "iteration",
                -1,
            )
        )

        try:
            G_core = (
                representative_transverse_shear_modulus(
                    record.core,
                    relative_tolerance=0.05,
                )
            )

            scalar_warping_valid = True

        except ValueError:
            # Still record the state. We simply do not run the current
            # single-f(z) material-informed formulation.
            G_core = 0.5 * (
                record.core.G_xz
                + record.core.G_yz
            )

            scalar_warping_valid = False

        # ----------------------------------------------------------
        # Reddy
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # FSDT
        # ----------------------------------------------------------

        fsdt = (
            build_energy_equivalent_fsdt_section(
                reddy.layers
            )
        )

        # ----------------------------------------------------------
        # HOST material-informed
        # ----------------------------------------------------------

        host = None

        if scalar_warping_valid:
            warping = (
                material_informed_generalized_warping(
                    h=h,
                    hc=hc,
                    G_face=G_face,
                    G_core=G_core,
                )
            )

            host = build_three_layer_sandwich(
                C_bottom=C_face,
                C_core=record.C_hom,
                C_top=C_face,
                hf_bottom=hf,
                hc=hc,
                hf_top=hf,
                warping=warping,
            )

        common = dict(
            q0=q0,
            a=a,
            b=b,
            max_mode=21,
        )

        w_fsdt = (
            center_deflection_uniform_pressure_fsdt(
                fsdt,
                **common,
            )
        )

        w_reddy = (
            center_deflection_uniform_pressure_hsdt(
                reddy.K11,
                **common,
            )
        )

        if host is not None:
            w_host = (
                center_deflection_uniform_pressure_hsdt(
                    host.K11,
                    **common,
                )
            )
        else:
            w_host = np.nan

        shear_ratio = (
            max(
                record.core.G_xz,
                record.core.G_yz,
            )
            / min(
                record.core.G_xz,
                record.core.G_yz,
            )
        )

        eng = data.get(
            "engineering_constants",
            {}
        )

        rows.append(
            [
                record.topology,
                record.relative_density_actual,
                load_case,
                iteration,
                float(
                    eng.get(
                        "Ex",
                        np.nan,
                    )
                ),
                float(
                    eng.get(
                        "Ey",
                        np.nan,
                    )
                ),
                float(
                    eng.get(
                        "Ez",
                        np.nan,
                    )
                ),
                record.core.G_xy,
                record.core.G_xz,
                record.core.G_yz,
                shear_ratio,
                w_fsdt,
                w_reddy,
                w_host,
                scalar_warping_valid,
                path.name,
            ]
        )

        host_text = (
            "{:.6f}".format(
                w_host * 1000.0
            )
            if np.isfinite(w_host)
            else "anisotropic"
        )

        print(
            "{:>9s} {:5.3f} {:>11s} {:8d} "
            "{:9.2f} {:9.2f} {:8.4f} "
            "{:11.6f} {:11.6f} {:>11s}".format(
                record.topology,
                record.relative_density_actual,
                load_case,
                iteration,
                record.core.G_xz,
                record.core.G_yz,
                shear_ratio,
                w_fsdt * 1000.0,
                w_reddy * 1000.0,
                host_text,
            )
        )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "tpms_database"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "tpms_sandwich_database.csv"
    )

    header = (
        "topology,"
        "relative_density,"
        "optimization_load_case,"
        "iteration,"
        "Ex_MPa,"
        "Ey_MPa,"
        "Ez_MPa,"
        "Gxy_MPa,"
        "Gxz_MPa,"
        "Gyz_MPa,"
        "Gxz_Gyz_anisotropy,"
        "w_fsdt_mm,"
        "w_reddy_mm,"
        "w_host_mm,"
        "scalar_warping_valid,"
        "source_json"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        handle.write(
            header + "\n"
        )

        for row in rows:
            handle.write(
                ",".join(
                    str(value)
                    for value in row
                )
                + "\n"
            )

    print()
    print(
        "CSV:",
        csv_path,
    )


if __name__ == "__main__":
    main()