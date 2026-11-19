"""
Input/output utilities for homogenized TPMS stiffness data.

Expected JSON fields include:

    topology
    relative_density_target
    relative_density_actual
    units
    orientation
    shear_convention
    voigt_order
    C_hom

The HOST permanent Voigt ordering is

    [xx, yy, zz, xy, xz, yz].
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from host.tpms_core import (
    HomogenizedTPMSCore,
    build_homogenized_tpms_core,
)


HOST_VOIGT_ORDER = (
    "xx",
    "yy",
    "zz",
    "xy",
    "xz",
    "yz",
)


@dataclass(frozen=True)
class TPMSHomogenizationRecord:
    topology: str
    relative_density_target: float
    relative_density_actual: float
    units: str
    orientation: str
    shear_convention: str
    voigt_order: tuple
    C_hom: np.ndarray
    core: HomogenizedTPMSCore
    metadata: dict


def load_tpms_homogenization_json(
    path,
) -> TPMSHomogenizationRecord:
    """
    Load and validate one homogenized TPMS JSON record.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    required = (
        "topology",
        "relative_density_target",
        "relative_density_actual",
        "units",
        "orientation",
        "shear_convention",
        "voigt_order",
        "C_hom",
    )

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        raise ValueError(
            "Missing required TPMS fields: {}".format(
                ", ".join(missing)
            )
        )

    units = str(
        data["units"]
    )

    if units.lower() != "mpa":
        raise ValueError(
            "Current HOST benchmark expects stiffness units in MPa; "
            "received '{}'.".format(units)
        )

    shear_convention = str(
        data["shear_convention"]
    ).lower()

    if shear_convention != "engineering":
        raise ValueError(
            "HOST currently expects engineering shear convention."
        )

    voigt_order = tuple(
        data["voigt_order"]
    )

    if voigt_order != HOST_VOIGT_ORDER:
        raise ValueError(
            "TPMS tensor ordering {} does not match HOST ordering {}."
            .format(
                voigt_order,
                HOST_VOIGT_ORDER,
            )
        )

    C_hom = np.asarray(
        data["C_hom"],
        dtype=float,
    )

    if C_hom.shape != (6, 6):
        raise ValueError(
            "C_hom must have shape (6, 6)."
        )

    core = build_homogenized_tpms_core(
        C_hom
    )

    return TPMSHomogenizationRecord(
        topology=str(
            data["topology"]
        ),
        relative_density_target=float(
            data["relative_density_target"]
        ),
        relative_density_actual=float(
            data["relative_density_actual"]
        ),
        units=units,
        orientation=str(
            data["orientation"]
        ),
        shear_convention=shear_convention,
        voigt_order=voigt_order,
        C_hom=C_hom.copy(),
        core=core,
        metadata=data,
    )