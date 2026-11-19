#!/usr/bin/env python3

"""
Batch HOST buckling evaluation for all TPMS homogenization JSON files.

For every state, the script evaluates:

    - uniaxial x compression
    - uniaxial y compression
    - equal biaxial compression

using:

    1. global Reddy HSDT
    2. material-informed HOST warping, where scalar transverse
       shear warping is admissible

The complete homogenized 6x6 TPMS stiffness tensor is retained.

Outputs:
    - console summary
    - CSV database
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


QUERY_DIR = Path(
    r"C:\AbaqusScratch\UCs\Shells\TPMS_mapped_thickness"
    r"\_stiffness_database\queries"
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


def mode_text(result):
    return "({},{})".format(
        result.m,
        result.n,
    )


def main():
    # --------------------------------------------------------------
    # Benchmark geometry/material
    # --------------------------------------------------------------

    E_face = 70000.0
    nu_face = 0.33

    hf = 0.5
    hc = 10.0
    h = 2.0 * hf + hc

    a = 100.0
    b = 100.0

    max_m = 10
    max_n = 10

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

    files = sorted(
        QUERY_DIR.glob("*.json")
    )

    if not files:
        raise RuntimeError(
            "No JSON files found in {}".format(
                QUERY_DIR
            )
        )

    rows = []

    print()
    print(
        "{:>9s} {:>5s} {:>11s} {:>5s} "
        "{:>10s} {:>7s} "
        "{:>10s} {:>7s} "
        "{:>10s} {:>7s} "
        "{:>10s} {:>10s} {:>10s}".format(
            "Topology",
            "RD",
            "load",
            "itag",
            "Nx_R",
            "mode",
            "Ny_R",
            "mode",
            "Nbi_R",
            "mode",
            "Nx_HOST",
            "Ny_HOST",
            "Nbi_HOST",
        )
    )

    print("-" * 140)

    for path in files:
        record = load_tpms_homogenization_json(
            path
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

        # ----------------------------------------------------------
        # Reddy model
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

        common = dict(
            a=a,
            b=b,
            max_m=max_m,
            max_n=max_n,
        )

        r_x = critical_uniaxial_load_x(
            reddy.K11,
            **common,
        )

        r_y = critical_uniaxial_load_y(
            reddy.K11,
            **common,
        )

        r_bi = critical_equal_biaxial_load(
            reddy.K11,
            **common,
        )

        # ----------------------------------------------------------
        # Material-informed HOST
        # ----------------------------------------------------------

        scalar_warping_valid = True

        try:
            G_core = (
                representative_transverse_shear_modulus(
                    record.core,
                    relative_tolerance=0.05,
                )
            )

        except ValueError:
            scalar_warping_valid = False
            G_core = np.nan

        if scalar_warping_valid:
            host = build_three_layer_sandwich(
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

            h_x = critical_uniaxial_load_x(
                host.K11,
                **common,
            )

            h_y = critical_uniaxial_load_y(
                host.K11,
                **common,
            )

            h_bi = critical_equal_biaxial_load(
                host.K11,
                **common,
            )

            host_x = h_x.load_factor
            host_y = h_y.load_factor
            host_bi = h_bi.load_factor

            host_mode_x = mode_text(
                h_x
            )

            host_mode_y = mode_text(
                h_y
            )

            host_mode_bi = mode_text(
                h_bi
            )

        else:
            host_x = np.nan
            host_y = np.nan
            host_bi = np.nan

            host_mode_x = "anis"
            host_mode_y = "anis"
            host_mode_bi = "anis"

        shear_ratio = (
            record.core.shear_anisotropy_xz_yz
        )

        rows.append(
            [
                record.topology,
                record.relative_density_actual,
                load_case,
                iteration,

                record.core.G_xz,
                record.core.G_yz,
                shear_ratio,

                r_x.load_factor,
                r_x.m,
                r_x.n,

                r_y.load_factor,
                r_y.m,
                r_y.n,

                r_bi.load_factor,
                r_bi.m,
                r_bi.n,

                host_x,
                host_mode_x,

                host_y,
                host_mode_y,

                host_bi,
                host_mode_bi,

                scalar_warping_valid,

                path.name,
            ]
        )

        host_x_text = (
            "{:.3f}".format(
                host_x
            )
            if np.isfinite(host_x)
            else "anis"
        )

        host_y_text = (
            "{:.3f}".format(
                host_y
            )
            if np.isfinite(host_y)
            else "anis"
        )

        host_bi_text = (
            "{:.3f}".format(
                host_bi
            )
            if np.isfinite(host_bi)
            else "anis"
        )

        print(
            "{:>9s} {:5.3f} {:>11s} {:5d} "
            "{:10.3f} {:>7s} "
            "{:10.3f} {:>7s} "
            "{:10.3f} {:>7s} "
            "{:>10s} {:>10s} {:>10s}".format(
                record.topology,
                record.relative_density_actual,
                load_case,
                iteration,

                r_x.load_factor,
                mode_text(r_x),

                r_y.load_factor,
                mode_text(r_y),

                r_bi.load_factor,
                mode_text(r_bi),

                host_x_text,
                host_y_text,
                host_bi_text,
            )
        )

    # --------------------------------------------------------------
    # CSV output
    # --------------------------------------------------------------

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "verification"
        / "tpms_buckling_database"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "tpms_buckling_database.csv"
    )

    header = (
        "topology,"
        "relative_density,"
        "optimization_load_case,"
        "iteration,"
        "Gxz_MPa,"
        "Gyz_MPa,"
        "Gxz_Gyz_anisotropy,"
        "Nx_cr_reddy_N_per_mm,"
        "Nx_mode_m_reddy,"
        "Nx_mode_n_reddy,"
        "Ny_cr_reddy_N_per_mm,"
        "Ny_mode_m_reddy,"
        "Ny_mode_n_reddy,"
        "Nbiax_cr_reddy_N_per_mm,"
        "Nbiax_mode_m_reddy,"
        "Nbiax_mode_n_reddy,"
        "Nx_cr_host_N_per_mm,"
        "Nx_mode_host,"
        "Ny_cr_host_N_per_mm,"
        "Ny_mode_host,"
        "Nbiax_cr_host_N_per_mm,"
        "Nbiax_mode_host,"
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