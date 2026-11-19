#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameterized Abaqus/Standard four-point-bending builder for a sandwich beam
with an isotropic AlSi10Mg face material and a homogenized anisotropic TPMS core.

Primary purpose
---------------
Create a clean 3-D continuum reference model for comparison with:
    - the current HOST beam theory,
    - Euler-Bernoulli / classical bending,
    - FSDT / Timoshenko-type references,
    - Reddy-type higher-order theory,
    - later laminate, layerwise, and special sandwich theories.

The model is deliberately linear elastic and geometrically linear. Loads and
supports are idealized as lines on the outer face-sheet surfaces. Contact and
roller indentation are NOT included in this verification model.

Coordinate convention
---------------------
Global Abaqus coordinates are chosen to match the structural coordinates:

    X = beam longitudinal direction
    Y = specimen width direction
    Z = sandwich thickness direction

The part is created with z in [0,h] and the assembly instance is translated by
-h/2, so the final GLOBAL Abaqus coordinates have the sandwich mid-plane at
Z = 0.

TPMS database
-------------
The script queries the SQLite stiffness database directly through
"query_tpms_stiffness_database_rotated.py".

Default database state:
    STATE = "initial"

Thus the default core is the unoptimized homogenized baseline state.

The current database uses:
    compression -> target metric Ey
    shear       -> target metric Gxy

The default orientation preset maps

    x_UC -> +X_model
    y_UC -> +Z_model
    z_UC -> -Y_model

which maps both optimization targets into structurally relevant beam-core
directions:

    Ey_UC  -> Ez_model      (through-thickness normal stiffness)
    Gxy_UC -> Gxz_model     (longitudinal-thickness shear stiffness)

The SAME orientation should be used for the initial and optimized states when
they are compared.

Units
-----
Consistent mm-N-MPa units are used.

Compatibility
-------------
Written with Python 3.8-compatible syntax for Abaqus 2024/2025 workflows.

Typical execution
-----------------
    abaqus cae noGUI=build_host_4pb_homogenized_beam.py

Files generated
---------------
Each deterministic case directory under the configured
Abaqus scratch root / 4pb_homogenized contains:

    <job>.cae
    <job>.inp
    manifest.json

If the exact same result-relevant configuration is requested again, the
builder stops rather than overwriting the previous case unless overwrite is
explicitly enabled.

Optionally, the job can also be submitted by setting SUBMIT_JOB = True.
"""

from __future__ import print_function

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Database / core state
# -----------------------------------------------------------------------------

# Machine-specific paths come from config/paths.local.json.
#
# Important Abaqus/CAE detail:
# "abaqus cae noGUI=script.py" executes the script in a context where
# __file__ is not guaranteed to be defined. Resolve the builder path once at
# startup and use that canonical path everywhere in this script.
def _resolve_builder_script_path():
    try:
        return Path(__file__).resolve()
    except NameError:
        pass

    launch_cwd = Path.cwd().resolve()

    # First inspect argv because some Abaqus launchers retain the noGUI script
    # path there.
    for argument in sys.argv:
        text = str(argument).strip().strip('"').strip("'")
        if not text.lower().endswith(".py"):
            continue

        candidate = Path(text)

        if not candidate.is_absolute():
            candidate = launch_cwd / candidate

        if candidate.is_file():
            return candidate.resolve()

    # Standard HOST invocation from the repository root:
    #
    #   abaqus cae noGUI="fem\\abaqus\\builders\\build_4pb_homogenized_beam.py"
    candidate = (
        launch_cwd
        / "fem"
        / "abaqus"
        / "builders"
        / "build_4pb_homogenized_beam.py"
    )

    if candidate.is_file():
        return candidate.resolve()

    # Also support launching while already inside the builders directory.
    candidate = launch_cwd / "build_4pb_homogenized_beam.py"

    if candidate.is_file():
        return candidate.resolve()

    raise RuntimeError(
        "Could not resolve the Abaqus builder script path. "
        "Launch from the HOST repository root with:\n"
        '  abaqus cae noGUI="fem\\abaqus\\builders\\'
        'build_4pb_homogenized_beam.py"'
    )


BUILDER_SCRIPT_PATH = _resolve_builder_script_path()
ABAQUS_FEM_ROOT = BUILDER_SCRIPT_PATH.parents[1]

if str(ABAQUS_FEM_ROOT) not in sys.path:
    sys.path.insert(0, str(ABAQUS_FEM_ROOT))

from common.path_config import load_path_config

_PATHS = load_path_config(BUILDER_SCRIPT_PATH)
REPO_ROOT = _PATHS["repo_root"]
LOCAL_PATH_CONFIG = _PATHS["config_path"]
ABAQUS_SCRATCH_ROOT = _PATHS["abaqus_scratch_root"]
DATABASE_PATH = _PATHS["tpms_stiffness_database"]

# -----------------------------------------------------------------------------
# Optional runtime overrides
# -----------------------------------------------------------------------------
#
# Selected campaign parameters can be overridden without editing the tracked
# builder source. The resolved result-relevant values are already part of the
# deterministic case configuration/hash; the raw environment overrides are
# additionally recorded in manifest.json for provenance.
#
# Example (PowerShell):
#   $env:HOST_4PB_NE_CELL = "4"
#   $env:HOST_4PB_SUBMIT_JOB = "1"
#   abaqus cae noGUI="fem\\abaqus\\builders\\build_4pb_homogenized_beam.py"

def _env_int(name, default):
    value = os.environ.get(name)

    if value is None or not str(value).strip():
        return int(default)

    return int(str(value).strip())


def _env_float(name, default):
    value = os.environ.get(name)

    if value is None or not str(value).strip():
        return float(default)

    return float(str(value).strip())


def _env_text(name, default):
    value = os.environ.get(name)

    if value is None or not str(value).strip():
        return str(default)

    return str(value).strip()


def _env_bool(name, default):
    value = os.environ.get(name)

    if value is None or not str(value).strip():
        return bool(default)

    normalized = str(value).strip().lower()

    if normalized in ("1", "true", "yes", "on"):
        return True

    if normalized in ("0", "false", "no", "off"):
        return False

    raise ValueError(
        "{} must be one of 1/0, true/false, yes/no, on/off; got {!r}."
        .format(
            name,
            value,
        )
    )


RUNTIME_OVERRIDES = {
    key: os.environ[key]
    for key in (
        "HOST_4PB_CORE_CAMPAIGN_TYPE",
        "HOST_4PB_CORE_TOPOLOGY",
        "HOST_4PB_CORE_RELATIVE_DENSITY",
        "HOST_4PB_CORE_LOAD_CASE",
        "HOST_4PB_CORE_STATE",
        "HOST_4PB_N_CORE_HEIGHT",
        "HOST_4PB_N_SHEAR",
        "HOST_4PB_FACE_BOTTOM_MM",
        "HOST_4PB_FACE_TOP_MM",
        "HOST_4PB_NE_CELL",
        "HOST_4PB_FACE_NE_Z",
        "HOST_4PB_ELEMENT_TYPE",
        "HOST_4PB_FIXTURE_MODE",
        "HOST_4PB_DOMAIN_MODE",
        "HOST_4PB_SUBMIT_JOB",
        "HOST_4PB_NUM_CPUS",
    )
    if key in os.environ
}

# The stiffness query/rotation helper is a repository file, not a local path.
QUERY_MODULE_DIRECTORY = ABAQUS_FEM_ROOT / "materials"

CORE_CAMPAIGN_TYPE = _env_text(
    "HOST_4PB_CORE_CAMPAIGN_TYPE", "sizing_shape"
)  # "sizing_shape" / "sizing_only"
CORE_TOPOLOGY = _env_text(
    "HOST_4PB_CORE_TOPOLOGY", "Gyroid"
)  # Primitive / Gyroid / Diamond
CORE_RELATIVE_DENSITY = _env_text(
    "HOST_4PB_CORE_RELATIVE_DENSITY", "0.12"
)  # 0.12, 12, "12%", or "rd12"
CORE_LOAD_CASE = _env_text(
    "HOST_4PB_CORE_LOAD_CASE", "compression"
)  # compression / shear
CORE_STATE = _env_text(
    "HOST_4PB_CORE_STATE", "initial"
)  # initial / best / final / integer-like text

# -----------------------------------------------------------------------------
# Unit-cell-to-beam orientation
# -----------------------------------------------------------------------------

# "target_to_beam_xz":
#     model X <- +UC x
#     model Y <- -UC z
#     model Z <- +UC y
#
# This maps the current database target metrics
#     Ey  -> model Ez
#     Gxy -> model Gxz
#
# "identity":
#     model X <- +UC x
#     model Y <- +UC y
#     model Z <- +UC z
#
# "axis_map":
#     use the three explicit settings below.
CORE_ORIENTATION_PRESET = "target_to_beam_xz"

MODEL_X_FROM_UC = "+x"
MODEL_Y_FROM_UC = "-z"
MODEL_Z_FROM_UC = "+y"

# The database matrices are already symmetric in the current production data.
# Keep False unless deliberately cleaning tiny numerical asymmetry.
SYMMETRIZE_CORE_MATRIX = False

# -----------------------------------------------------------------------------
# Four-point-bending geometry: cell-count parameterization
# -----------------------------------------------------------------------------

CELL_SIZE_MM = 10.0

# Core and specimen dimensions in nominal unit-cell counts.
#
# HOST_4PB_N_CORE_HEIGHT enables deterministic core-height transfer sweeps.
# The resolved core height enters the derived geometry, case hash, and manifest.
N_CORE_HEIGHT = _env_int("HOST_4PB_N_CORE_HEIGHT", 6)
N_WIDTH = 6

# Four-point-bending fixture:
# support-to-nearest-load distance = N_SHEAR * CELL_SIZE_MM.
#
# The runtime override is used for deterministic deep-to-slender 4PB sweeps:
#
#     HOST_4PB_N_SHEAR=6,9,12,...
#
# N_SHEAR enters the derived geometry and therefore the case hash/manifest.
N_SHEAR = _env_int("HOST_4PB_N_SHEAR", 6)

# Distance between the two loading lines.
N_MOMENT = 8

# Specimen overhang outside each support.
N_OVERHANG = 2

# Face-sheet thicknesses.
#
# Runtime overrides enable deterministic sandwich-geometry transfer sweeps
# without editing this tracked builder. The resolved thicknesses are part of
# geometry and therefore automatically enter the case hash and manifest.
FACE_BOTTOM_MM = _env_float("HOST_4PB_FACE_BOTTOM_MM", 1.0)
FACE_TOP_MM = _env_float("HOST_4PB_FACE_TOP_MM", 1.0)

# -----------------------------------------------------------------------------
# Face-sheet material
# -----------------------------------------------------------------------------

FACE_E_MPA = 70000.0
FACE_NU = 0.33

# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------

# 2 -> 2 x 2 x 2 = 8 linear hex elements per nominal core-cell volume.
# This is the requested baseline. It must later be checked against 1, 2, 4...
CORE_ELEMENTS_PER_CELL_EDGE = _env_int("HOST_4PB_NE_CELL", 2)

# Number of solid elements through EACH face sheet.
FACE_ELEMENTS_THROUGH_THICKNESS = _env_int("HOST_4PB_FACE_NE_Z", 2)

# "C3D8I" is the default bending-oriented first-order solid.
# Other supported choices here: "C3D8R", "C3D20R".
ELEMENT_TYPE = _env_text("HOST_4PB_ELEMENT_TYPE", "C3D8I").upper()

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

TOTAL_APPLIED_LOAD_N = 1000.0

# Line loads are applied by consistent nodal weighting along the specimen width:
# endpoint nodes receive half the interior nodal line load.
USE_CONSISTENT_LINE_LOAD_WEIGHTS = True

# Fixture/load-introduction mode:
#
#   outer_surface_lines
#       Physical-test-like idealization used by the existing reference model:
#       supports on the bottom outer surface and loads on the top outer surface.
#
#   midplane_lines
#       Diagnostic only: supports and loads act on the internal z=0 mesh lines.
#       This removes the top/bottom eccentricity but is still a local line-load
#       model and is not equivalent to a generalized beam-section resultant.
#
#   section_distributing
#       Theory-compatibility reference: a centroidal reference point at each
#       support/load station is continuum-distributing coupled to every mesh
#       node on the corresponding full cross section. Only translations are
#       coupled; all RP rotations are released. Loads/support conditions are
#       therefore imposed as section-average translational resultants while the
#       cross section remains free to deform and warp.
#
# The midplane-line option is deliberately retained as a diagnostic. The
# section-distributing option is the intended reduced-theory comparison model.
FIXTURE_MODE = _env_text(
    "HOST_4PB_FIXTURE_MODE",
    "outer_surface_lines",
).strip().lower()

if FIXTURE_MODE not in (
    "outer_surface_lines",
    "midplane_lines",
    "section_distributing",
):
    raise ValueError(
        "HOST_4PB_FIXTURE_MODE must be one of "
        "'outer_surface_lines', 'midplane_lines', or "
        "'section_distributing'; got {!r}.".format(
            FIXTURE_MODE
        )
    )

# Computational-domain mode:
#
# full:
#     full beam length and full specimen width.
#
# quarter_xy:
#     x in [0,L/2], y in [0,b/2], with U1=0 on x=L/2 and U2=0 on
#     y=0. The retained load is one quarter of the full 4PB resultant.
#     This is an imposed-symmetry approximation if the homogenized tensor is
#     not exactly reflection-symmetric.
DOMAIN_MODE = _env_text(
    "HOST_4PB_DOMAIN_MODE",
    "full",
).strip().lower()

if DOMAIN_MODE not in (
    "full",
    "quarter_xy",
):
    raise ValueError(
        "HOST_4PB_DOMAIN_MODE must be 'full' or 'quarter_xy'; got {!r}."
        .format(DOMAIN_MODE)
    )

if (
    DOMAIN_MODE == "quarter_xy"
    and FIXTURE_MODE != "section_distributing"
):
    raise ValueError(
        "quarter_xy is currently implemented only for "
        "HOST_4PB_FIXTURE_MODE='section_distributing'."
    )

# -----------------------------------------------------------------------------
# Analysis / output
# -----------------------------------------------------------------------------

MODEL_NAME = "HOST_4PB_HOMOGENIZED"
STEP_NAME = "BENDING"

OUTPUT_ROOT = ABAQUS_SCRATCH_ROOT / "4pb_homogenized"

# None -> generated automatically from the material/query/mesh settings.
JOB_NAME = None

# Data-hygiene protection. The deterministic case directory contains a hash of
# all result-relevant inputs. Existing cases are never replaced accidentally.
ALLOW_OVERWRITE_EXISTING_CASE = False
CASE_HASH_LENGTH = 10

# Result-relevant implementation revision. Bump this deliberately whenever a
# builder change can alter the generated FE model, node/element sets, loads,
# boundary conditions, output semantics, or other scientifically relevant
# contents of a case. Including it in the case hash prevents different model
# implementations from silently sharing the same run directory.
CASE_DEFINITION_SCHEMA = "host_4pb_homogenized_case_v3"
MODEL_IMPLEMENTATION_REVISION = 6

SAVE_CAE = True
WRITE_INPUT = True
SUBMIT_JOB = _env_bool("HOST_4PB_SUBMIT_JOB", False)
WAIT_FOR_COMPLETION = True

NUM_CPUS = _env_int("HOST_4PB_NUM_CPUS", 8)
MEMORY_PERCENT = 90

# Coordinate-selection tolerance. If None, it is derived from the mesh size.
NODE_SELECTION_TOL_MM = None

# Relative matrix-symmetry tolerance before Abaqus receives the 21 constants.
MATRIX_SYMMETRY_REL_TOL = 1.0e-8

# Field variables retained for later theory/FE comparison and publication.
FIELD_OUTPUT_VARIABLES = (
    "U",
    "RF",
    "S",
    "E",
    "SENER",
    "IVOL",
    "COORD",
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())


def state_label(value):
    if isinstance(value, int):
        return "itag{:02d}".format(int(value))

    text = str(value).strip().lower()

    match = re.fullmatch(r"(?:itag|iter|iteration)?\s*(\d+)", text)
    if match:
        return "itag{:02d}".format(int(match.group(1)))

    return safe_name(text)


def as_float_matrix(matrix):
    return [
        [float(value) for value in row]
        for row in matrix
    ]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def max_abs_matrix(matrix):
    maximum = 0.0
    for row in matrix:
        for value in row:
            maximum = max(maximum, abs(float(value)))
    return maximum


def max_symmetry_error(matrix):
    matrix_t = transpose(matrix)
    maximum = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            maximum = max(
                maximum,
                abs(float(matrix[i][j]) - float(matrix_t[i][j])),
            )
    return maximum


def symmetrize(matrix):
    n = len(matrix)
    out = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            out[i][j] = 0.5 * (
                float(matrix[i][j])
                + float(matrix[j][i])
            )

    return out


def row_value(row, key, default=None):
    try:
        value = row[key]
    except Exception:
        return default

    return default if value is None else value


def derive_geometry():
    positive_ints = {
        "N_CORE_HEIGHT": N_CORE_HEIGHT,
        "N_WIDTH": N_WIDTH,
        "N_SHEAR": N_SHEAR,
        "N_MOMENT": N_MOMENT,
    }

    for name, value in positive_ints.items():
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError("{} must be a positive integer.".format(name))

    if isinstance(N_OVERHANG, bool) or int(N_OVERHANG) != N_OVERHANG:
        raise ValueError("N_OVERHANG must be a non-negative integer.")

    if int(N_OVERHANG) < 0:
        raise ValueError("N_OVERHANG must be non-negative.")

    lc = float(CELL_SIZE_MM)
    tf_bottom = float(FACE_BOTTOM_MM)
    tf_top = float(FACE_TOP_MM)

    if lc <= 0.0:
        raise ValueError("CELL_SIZE_MM must be positive.")
    if tf_bottom <= 0.0 or tf_top <= 0.0:
        raise ValueError("Face thicknesses must be positive.")

    nh = int(N_CORE_HEIGHT)
    nb = int(N_WIDTH)
    na = int(N_SHEAR)
    nm = int(N_MOMENT)
    no = int(N_OVERHANG)

    hc = nh * lc
    b = nb * lc

    shear_span = na * lc
    load_span = nm * lc
    support_span = (2 * na + nm) * lc
    overhang = no * lc
    length = (2 * no + 2 * na + nm) * lc

    h = tf_bottom + hc + tf_top

    support_left = overhang
    support_right = length - overhang

    load_left = overhang + shear_span
    load_right = load_left + load_span

    midspan = 0.5 * length

    return {
        "cell_size": lc,
        "n_height": nh,
        "n_width": nb,
        "n_shear": na,
        "n_moment": nm,
        "n_overhang": no,
        "core_height": hc,
        "width": b,
        "face_bottom": tf_bottom,
        "face_top": tf_top,
        "total_height": h,
        "shear_span": shear_span,
        "load_span": load_span,
        "support_span": support_span,
        "overhang": overhang,
        "length": length,
        "support_left": support_left,
        "support_right": support_right,
        "load_left": load_left,
        "load_right": load_right,
        "midspan": midspan,
        "z_bottom": -0.5 * h,
        "z_top": 0.5 * h,
        "z_core_bottom": -0.5 * h + tf_bottom,
        "z_core_top": 0.5 * h - tf_top,
    }


def build_domain_definition(geometry):
    if DOMAIN_MODE == "full":
        return {
            "mode": "full",
            "x_min_mm": 0.0,
            "x_max_mm": float(geometry["length"]),
            "y_min_mm": -0.5 * float(geometry["width"]),
            "y_max_mm": 0.5 * float(geometry["width"]),
            "model_volume_fraction_of_full": 1.0,
            "applied_load_fraction_of_full": 1.0,
            "energy_reconstruction_factor": 1.0,
            "section_resultant_reconstruction_factor": 1.0,
            "symmetry_planes": [],
        }

    if DOMAIN_MODE == "quarter_xy":
        return {
            "mode": "quarter_xy",
            "x_min_mm": 0.0,
            "x_max_mm": float(geometry["midspan"]),
            "y_min_mm": 0.0,
            "y_max_mm": 0.5 * float(geometry["width"]),
            "model_volume_fraction_of_full": 0.25,
            "applied_load_fraction_of_full": 0.25,
            "energy_reconstruction_factor": 4.0,
            "section_resultant_reconstruction_factor": 2.0,
            "symmetry_planes": [
                {
                    "plane": "x=L/2",
                    "normal": "X",
                    "constraint": "U1=0",
                },
                {
                    "plane": "y=0",
                    "normal": "Y",
                    "constraint": "U2=0",
                },
            ],
        }

    raise RuntimeError(
        "Unsupported DOMAIN_MODE {!r}.".format(DOMAIN_MODE)
    )


def constitutive_reflection_relative_error(C, normal_axis):
    """Return max-norm violation of C = T C T for an axis reflection."""
    axis = str(normal_axis).strip().lower()

    if axis == "x":
        signs = (1.0, 1.0, 1.0, -1.0, -1.0, 1.0)
    elif axis == "y":
        signs = (1.0, 1.0, 1.0, -1.0, 1.0, -1.0)
    elif axis == "z":
        signs = (1.0, 1.0, 1.0, 1.0, -1.0, -1.0)
    else:
        raise ValueError("normal_axis must be x, y, or z.")

    max_c = max(
        abs(float(C[i][j]))
        for i in range(6)
        for j in range(6)
    )
    max_diff = max(
        abs(
            float(C[i][j])
            - signs[i] * float(C[i][j]) * signs[j]
        )
        for i in range(6)
        for j in range(6)
    )

    return max_diff / max(max_c, 1.0e-30)


def build_orientation_map(target_metric):
    preset = str(CORE_ORIENTATION_PRESET).strip().lower()

    if preset == "identity":
        return "+x", "+y", "+z"

    if preset == "axis_map":
        return (
            str(MODEL_X_FROM_UC),
            str(MODEL_Y_FROM_UC),
            str(MODEL_Z_FROM_UC),
        )

    if preset == "target_to_beam_xz":
        # Current database target metrics:
        # compression -> Ey
        # shear       -> Gxy
        #
        # The same proper rotation maps:
        #     Ey  -> Ez
        #     Gxy -> Gxz
        supported = {"ey", "gxy"}

        metric = str(target_metric).strip().lower()

        if metric not in supported:
            raise ValueError(
                "CORE_ORIENTATION_PRESET='target_to_beam_xz' was designed "
                "for current database target metrics Ey and Gxy, but the "
                "queried row reports {!r}. Use CORE_ORIENTATION_PRESET="
                "'axis_map' and specify the mapping explicitly.".format(
                    target_metric
                )
            )

        return "+x", "-z", "+y"

    raise ValueError(
        "CORE_ORIENTATION_PRESET must be 'target_to_beam_xz', "
        "'identity', or 'axis_map'; got {!r}.".format(
            CORE_ORIENTATION_PRESET
        )
    )


def abaqus_anisotropic_constants_from_canonical(C):
    """
    Convert canonical 6x6 stiffness

        [xx, yy, zz, xy, xz, yz]

    into the 21-value Abaqus fully anisotropic elasticity table order:

        D1111,
        D1122, D2222,
        D1133, D2233, D3333,
        D1112, D2212, D3312, D1212,
        D1113, D2213, D3313, D1213, D1313,
        D1123, D2223, D3323, D1223, D1323, D2323
    """
    if len(C) != 6 or any(len(row) != 6 for row in C):
        raise ValueError("Canonical stiffness matrix must be 6x6.")

    c = C

    return (
        float(c[0][0]),  # D1111

        float(c[0][1]),  # D1122
        float(c[1][1]),  # D2222

        float(c[0][2]),  # D1133
        float(c[1][2]),  # D2233
        float(c[2][2]),  # D3333

        float(c[0][3]),  # D1112
        float(c[1][3]),  # D2212
        float(c[2][3]),  # D3312
        float(c[3][3]),  # D1212

        float(c[0][4]),  # D1113
        float(c[1][4]),  # D2213
        float(c[2][4]),  # D3313
        float(c[3][4]),  # D1213
        float(c[4][4]),  # D1313

        float(c[0][5]),  # D1123
        float(c[1][5]),  # D2223
        float(c[2][5]),  # D3323
        float(c[3][5]),  # D1223
        float(c[4][5]),  # D1323
        float(c[5][5]),  # D2323
    )


def compact_number(value):
    text = "{:g}".format(float(value))
    return text.replace("-", "m").replace(".", "p")


def short_load_case(value):
    text = str(value).strip().lower()

    aliases = {
        "compression": "ca",
        "compressive": "ca",
        "ca": "ca",
        "shear": "sa",
        "pure_shear": "sa",
        "sa": "sa",
    }

    return aliases.get(text, safe_name(text))


def build_case_configuration(core_data, geometry):
    """
    Build the canonical result-relevant case definition used for hashing.

    If any quantity in this dictionary changes, the case hash changes and a
    new run directory is created.
    """
    row = core_data["row"]

    return {
        "schema": CASE_DEFINITION_SCHEMA,
        "model_implementation_revision": int(
            MODEL_IMPLEMENTATION_REVISION
        ),
        "core": {
            "campaign_type": row_value(row, "campaign_type"),
            "campaign_name": row_value(row, "campaign_name"),
            "eval_id": row_value(row, "eval_id"),
            "topology": row_value(row, "topology", CORE_TOPOLOGY),
            "rd_target_fraction": row_value(
                row,
                "rd_target_fraction",
                CORE_RELATIVE_DENSITY,
            ),
            "rd_actual_fraction": row_value(
                row,
                "rd_actual_fraction",
            ),
            "load_case": row_value(
                row,
                "load_case",
                CORE_LOAD_CASE,
            ),
            "target_metric": row_value(
                row,
                "target_metric_name",
            ),
            "iteration": row_value(row, "iteration"),
            "requested_state": CORE_STATE,
            "C_model_MPa": core_data["C_model_symmetric"],
        },
        "orientation": {
            "preset": CORE_ORIENTATION_PRESET,
            "model_x_from_uc": core_data["axis_map"][
                "model_x_from_uc"
            ],
            "model_y_from_uc": core_data["axis_map"][
                "model_y_from_uc"
            ],
            "model_z_from_uc": core_data["axis_map"][
                "model_z_from_uc"
            ],
            "Q_source_to_model": core_data["Q"],
        },
        "geometry": geometry,
        "domain": build_domain_definition(geometry),
        "face_material": {
            "E_MPa": float(FACE_E_MPA),
            "nu": float(FACE_NU),
        },
        "mesh": {
            "core_elements_per_cell_edge": int(
                CORE_ELEMENTS_PER_CELL_EDGE
            ),
            "face_elements_through_thickness": int(
                FACE_ELEMENTS_THROUGH_THICKNESS
            ),
            "element_type": str(ELEMENT_TYPE).strip().upper(),
        },
        "loading": {
            "fixture_mode": FIXTURE_MODE,
            "total_applied_load_N": float(TOTAL_APPLIED_LOAD_N),
            "consistent_line_load_weights": bool(
                USE_CONSISTENT_LINE_LOAD_WEIGHTS
            ),
        },
        "analysis": {
            "solver": "Abaqus/Standard",
            "linear_elastic": True,
            "nlgeom": False,
        },
    }


def build_case_identity(core_data, geometry):
    configuration = build_case_configuration(
        core_data,
        geometry,
    )

    canonical_json = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    full_hash = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    hash_length = int(CASE_HASH_LENGTH)

    if hash_length < 8 or hash_length > 64:
        raise ValueError(
            "CASE_HASH_LENGTH must lie between 8 and 64."
        )

    short_hash = full_hash[:hash_length]

    row = core_data["row"]

    topology = safe_name(
        row_value(row, "topology", CORE_TOPOLOGY)
    )

    rd_percent = 100.0 * float(
        row_value(
            row,
            "rd_target_fraction",
            CORE_RELATIVE_DENSITY,
        )
    )

    load_short = short_load_case(
        row_value(row, "load_case", CORE_LOAD_CASE)
    )

    state_short = state_label(CORE_STATE)

    descriptor = (
        "{topology}_rd{rd}_{load}_{state}"
        "_nh{nh}_nb{nb}_na{na}_nm{nm}_no{no}"
        "_tfb{tfb}_tft{tft}_ne{ne}_{element}_{fixture}_{domain}"
    ).format(
        topology=topology,
        rd=compact_number(rd_percent),
        load=load_short,
        state=state_short,
        nh=geometry["n_height"],
        nb=geometry["n_width"],
        na=geometry["n_shear"],
        nm=geometry["n_moment"],
        no=geometry["n_overhang"],
        tfb=compact_number(geometry["face_bottom"]),
        tft=compact_number(geometry["face_top"]),
        ne=int(CORE_ELEMENTS_PER_CELL_EDGE),
        element=safe_name(str(ELEMENT_TYPE).upper()),
        fixture={
            "outer_surface_lines": "fxsurf",
            "midplane_lines": "fxmid",
            "section_distributing": "fxsec",
        }[FIXTURE_MODE],
        domain={
            "full": "domfull",
            "quarter_xy": "domqxy",
        }[DOMAIN_MODE],
    )

    case_id = "{}_{}".format(
        safe_name(descriptor),
        short_hash,
    )

    if JOB_NAME is None:
        job_name = (
            "4PB_{topology}_rd{rd}_{load}_{state}"
            "_ne{ne}_{hash}"
        ).format(
            topology=topology[:12],
            rd=compact_number(rd_percent),
            load=load_short,
            state=state_short[:10],
            ne=int(CORE_ELEMENTS_PER_CELL_EDGE),
            hash=short_hash,
        )
    else:
        job_name = safe_name(JOB_NAME)

    job_name = safe_name(job_name)[:80]

    return {
        "case_id": case_id,
        "case_hash": short_hash,
        "case_hash_full_sha256": full_hash,
        "job_name": job_name,
        "configuration": configuration,
    }


def prepare_case_directory(output_root, case_id):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    case_directory = output_root / str(case_id)

    if case_directory.exists():
        existing_items = list(case_directory.iterdir())

        if existing_items and not ALLOW_OVERWRITE_EXISTING_CASE:
            raise RuntimeError(
                "Simulation case already exists and is not empty:\n"
                "{}\n\n"
                "This case ID is deterministic, so the current model inputs "
                "match an existing run.\n"
                "Use the existing results, change the case configuration, "
                "or deliberately set ALLOW_OVERWRITE_EXISTING_CASE=True."
                .format(case_directory)
            )

        if existing_items and ALLOW_OVERWRITE_EXISTING_CASE:
            print(
                "[WARNING] Removing existing case because "
                "ALLOW_OVERWRITE_EXISTING_CASE=True:"
            )
            print("          {}".format(case_directory))
            shutil.rmtree(str(case_directory))

    case_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return case_directory


def sha256_file(path):
    path = Path(path)

    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def git_provenance(repo_root):
    result = {
        "available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "status_porcelain": None,
    }

    try:
        commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "HEAD",
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

        branch = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

        result.update(
            {
                "available": True,
                "commit": commit,
                "branch": branch,
                "dirty": bool(status),
                "status_porcelain": status.splitlines(),
            }
        )

    except Exception as exc:
        result["error"] = str(exc)

    return result


def normalization_data(geometry):
    P = float(TOTAL_APPLIED_LOAD_N)
    b = float(geometry["width"])
    h = float(geometry["total_height"])
    S = float(geometry["support_span"])
    a = float(geometry["shear_span"])
    lc = float(geometry["cell_size"])

    if P <= 0.0:
        raise ValueError("TOTAL_APPLIED_LOAD_N must be positive.")

    reaction_nominal = 0.5 * P
    max_moment = reaction_nominal * a

    bending_stress_scale = max_moment / (b * h * h)
    shear_stress_scale = reaction_nominal / (b * h)

    deflection_scale = (
        P * S**3
        / (
            float(FACE_E_MPA)
            * b
            * h**3
        )
    )

    return {
        "recommended_dimensionless_coordinates": {
            "xi_support": "(x - support_left) / support_span",
            "xi_specimen": "x / specimen_length",
            "zeta": "z / total_height",
        },
        "dimensionless_geometry": {
            "L_over_h": geometry["length"] / h,
            "S_over_h": S / h,
            "a_over_h": a / h,
            "load_span_over_h": geometry["load_span"] / h,
            "b_over_h": b / h,
            "cell_size_over_h": lc / h,
            "core_height_over_h": geometry["core_height"] / h,
            "face_bottom_over_h": geometry["face_bottom"] / h,
            "face_top_over_h": geometry["face_top"] / h,
        },
        "load_scales": {
            "total_load_N": P,
            "nominal_support_reaction_N": reaction_nominal,
            "max_constant_moment_Nmm": max_moment,
        },
        "response_scales": {
            "deflection_scale_mm": deflection_scale,
            "deflection_normalization": (
                "w_bar = w / deflection_scale = "
                "E_face*b*h^3*w/(P*S^3)"
            ),
            "bending_stress_scale_MPa": bending_stress_scale,
            "bending_stress_normalization": (
                "sigma_bar = sigma_xx / "
                "[M_max/(b*h^2)]"
            ),
            "shear_stress_scale_MPa": shear_stress_scale,
            "shear_stress_normalization": (
                "tau_bar = tau_xz / [(P/2)/(b*h)]"
            ),
            "moment_normalization": "M_bar = M / M_max",
            "shear_resultant_normalization": "Q_bar = Q / (P/2)",
        },
    }


# =============================================================================
# DATABASE QUERY
# =============================================================================

def load_core_material():
    if QUERY_MODULE_DIRECTORY is None:
        script_dir = BUILDER_SCRIPT_PATH.parent
    else:
        script_dir = Path(QUERY_MODULE_DIRECTORY)

    module_dir = str(script_dir)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        from query_tpms_stiffness_database_rotated import (
            get_single_stiffness_record,
            oriented_stiffness_from_record,
            rotation_matrix_from_axis_map,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Could not import query_tpms_stiffness_database_rotated.py.\n"
            "Place that file next to this Abaqus builder or set "
            "QUERY_MODULE_DIRECTORY explicitly.\n"
            "Import error: {}".format(exc)
        )

    row, convention = get_single_stiffness_record(
        database_path=DATABASE_PATH,
        campaign_type=CORE_CAMPAIGN_TYPE,
        topology=CORE_TOPOLOGY,
        relative_density=CORE_RELATIVE_DENSITY,
        load_case=CORE_LOAD_CASE,
        state=CORE_STATE,
    )

    target_metric = row_value(row, "target_metric_name", "")
    model_x_from_uc, model_y_from_uc, model_z_from_uc = (
        build_orientation_map(target_metric)
    )

    Q = rotation_matrix_from_axis_map(
        model_x_from_uc,
        model_y_from_uc,
        model_z_from_uc,
    )

    oriented = oriented_stiffness_from_record(
        row=row,
        convention=convention,
        q_source_to_model=Q,
        symmetrize_before_rotation=bool(SYMMETRIZE_CORE_MATRIX),
    )

    C_model = as_float_matrix(oriented["C_model"])

    symmetry_error = max_symmetry_error(C_model)
    scale = max(1.0, max_abs_matrix(C_model))
    relative_symmetry_error = symmetry_error / scale

    if relative_symmetry_error > float(MATRIX_SYMMETRY_REL_TOL):
        raise ValueError(
            "Rotated core stiffness is not sufficiently symmetric for "
            "Abaqus anisotropic elasticity. max|C-C^T|/max|C| = {:.6e}"
            .format(relative_symmetry_error)
        )

    # Abaqus expects the 21 independent constants of a symmetric matrix.
    # Enforce exact numerical major symmetry after checking that the correction
    # is negligible.
    C_model_symmetric = symmetrize(C_model)

    positive_definite = row_value(
        row,
        "positive_definite_C_sym",
        None,
    )

    if positive_definite is not None and int(positive_definite) != 1:
        raise ValueError(
            "Selected database state does not pass the stored "
            "positive-definiteness QA flag."
        )

    return {
        "row": row,
        "convention": convention,
        "oriented": oriented,
        "Q": Q,
        "axis_map": {
            "model_x_from_uc": model_x_from_uc,
            "model_y_from_uc": model_y_from_uc,
            "model_z_from_uc": model_z_from_uc,
        },
        "C_model_symmetric": C_model_symmetric,
        "abaqus_constants": (
            abaqus_anisotropic_constants_from_canonical(
                C_model_symmetric
            )
        ),
        "relative_symmetry_error_before_cleanup": relative_symmetry_error,
    }


# =============================================================================
# ABAQUS-SPECIFIC HELPERS
# =============================================================================

def edge_endpoints(part, edge):
    vertex_indices = edge.getVertices()

    if len(vertex_indices) != 2:
        raise RuntimeError(
            "Expected a straight two-vertex edge; got {} vertices.".format(
                len(vertex_indices)
            )
        )

    p0 = part.vertices[vertex_indices[0]].pointOn[0]
    p1 = part.vertices[vertex_indices[1]].pointOn[0]

    return (
        (float(p0[0]), float(p0[1]), float(p0[2])),
        (float(p1[0]), float(p1[1]), float(p1[2])),
    )


def classify_axis_aligned_edges(part, geometry):
    x_edges = []
    y_edges = []
    z_bottom_edges = []
    z_core_edges = []
    z_top_edges = []

    tf_bottom = geometry["face_bottom"]
    hc = geometry["core_height"]
    tf_top = geometry["face_top"]

    z1 = tf_bottom
    z2 = tf_bottom + hc
    h = tf_bottom + hc + tf_top

    geom_tol = 1.0e-8 * max(
        1.0,
        geometry["length"],
        geometry["width"],
        h,
    )

    for edge in part.edges:
        p0, p1 = edge_endpoints(part, edge)

        delta = (
            abs(p1[0] - p0[0]),
            abs(p1[1] - p0[1]),
            abs(p1[2] - p0[2]),
        )

        axis = max(range(3), key=lambda index: delta[index])

        other = [
            delta[index]
            for index in range(3)
            if index != axis
        ]

        if max(other) > geom_tol:
            raise RuntimeError(
                "Encountered a non-axis-aligned edge; current structured "
                "meshing logic assumes a rectangular prismatic partition."
            )

        if axis == 0:
            x_edges.append(edge)

        elif axis == 1:
            y_edges.append(edge)

        else:
            z_mid = 0.5 * (p0[2] + p1[2])

            if z_mid < z1 - geom_tol:
                z_bottom_edges.append(edge)
            elif z_mid > z2 + geom_tol:
                z_top_edges.append(edge)
            else:
                z_core_edges.append(edge)

    if not x_edges or not y_edges or not z_core_edges:
        raise RuntimeError(
            "Failed to classify the structured mesh edges."
        )

    return {
        "x": tuple(x_edges),
        "y": tuple(y_edges),
        "z_bottom": tuple(z_bottom_edges),
        "z_core": tuple(z_core_edges),
        "z_top": tuple(z_top_edges),
    }


def labels_near_coordinates(nodes, x=None, y=None, z=None, tol=1.0e-6):
    labels = []

    for node in nodes:
        coordinate = node.coordinates

        if x is not None and abs(float(coordinate[0]) - float(x)) > tol:
            continue
        if y is not None and abs(float(coordinate[1]) - float(y)) > tol:
            continue
        if z is not None and abs(float(coordinate[2]) - float(z)) > tol:
            continue

        labels.append(int(node.label))

    return labels


def make_node_set_from_labels(assembly, instance, name, labels):
    labels = tuple(sorted(set(int(label) for label in labels)))

    if not labels:
        raise RuntimeError(
            "Cannot create node set {} because no nodes were selected.".format(
                name
            )
        )

    node_array = instance.nodes.sequenceFromLabels(
        labels=labels
    )

    assembly.Set(
        name=name,
        nodes=node_array,
    )

    return labels


def nearest_line_node_label(instance, line_labels, target_y):
    label_set = set(int(label) for label in line_labels)

    candidates = [
        node
        for node in instance.nodes
        if int(node.label) in label_set
    ]

    if not candidates:
        raise RuntimeError("No candidate nodes for nearest-line selection.")

    node = min(
        candidates,
        key=lambda item: abs(
            float(item.coordinates[1]) - float(target_y)
        ),
    )

    return int(node.label)


def split_line_load_labels(instance, labels):
    label_set = set(int(label) for label in labels)

    nodes = [
        node
        for node in instance.nodes
        if int(node.label) in label_set
    ]

    nodes.sort(
        key=lambda node: float(node.coordinates[1])
    )

    if len(nodes) < 2:
        raise RuntimeError(
            "A loading line must contain at least two nodes."
        )

    endpoint_labels = (
        int(nodes[0].label),
        int(nodes[-1].label),
    )

    interior_labels = tuple(
        int(node.label)
        for node in nodes[1:-1]
    )

    return endpoint_labels, interior_labels


def apply_line_load(
    model,
    assembly,
    instance,
    line_name,
    labels,
    line_resultant,
    step_name,
):
    endpoint_labels, interior_labels = split_line_load_labels(
        instance,
        labels,
    )

    n_segments = len(labels) - 1

    if n_segments <= 0:
        raise RuntimeError(
            "Loading line {} has no line segments.".format(line_name)
        )

    if USE_CONSISTENT_LINE_LOAD_WEIGHTS:
        interior_force = (
            -float(line_resultant)
            / float(n_segments)
        )

        endpoint_force = 0.5 * interior_force

    else:
        force_per_node = (
            -float(line_resultant)
            / float(len(labels))
        )
        interior_force = force_per_node
        endpoint_force = force_per_node

    end_set_name = "{}_ENDS".format(line_name)
    make_node_set_from_labels(
        assembly,
        instance,
        end_set_name,
        endpoint_labels,
    )

    model.ConcentratedForce(
        name="LOAD_{}_ENDS".format(line_name),
        createStepName=step_name,
        region=assembly.sets[end_set_name],
        cf3=endpoint_force,
    )

    if interior_labels:
        interior_set_name = "{}_INTERIOR".format(line_name)

        make_node_set_from_labels(
            assembly,
            instance,
            interior_set_name,
            interior_labels,
        )

        model.ConcentratedForce(
            name="LOAD_{}_INTERIOR".format(line_name),
            createStepName=step_name,
            region=assembly.sets[interior_set_name],
            cf3=interior_force,
        )

    applied_resultant = (
        2.0 * endpoint_force
        + len(interior_labels) * interior_force
    )

    return {
        "n_nodes": len(labels),
        "n_segments": n_segments,
        "endpoint_force_per_node_N": endpoint_force,
        "interior_force_per_node_N": interior_force,
        "resultant_N": applied_resultant,
    }


# =============================================================================
# ABAQUS MODEL BUILD
# =============================================================================

def build_abaqus_model(core_data, geometry, job_name):
    try:
        from abaqus import mdb
        # Field/history output request factories belong to the Abaqus Step
        # kernel module. Importing it is required in noGUI scripts so the
        # corresponding Model API is registered.
        import step
        import interaction
        import regionToolset
        from abaqusConstants import (
            ANALYSIS,
            ANISOTROPIC,
            AXIS_3,
            CARTESIAN,
            C3D20R,
            C3D8I,
            C3D8R,
            DEFAULT,
            DEFORMABLE_BODY,
            DISTRIBUTING,
            FIXED,
            GLOBAL,
            HEX,
            LAST_INCREMENT,
            OFF,
            ON,
            PERCENTAGE,
            ROTATION_NONE,
            STACK_3,
            STANDARD,
            STRUCTURED,
            THREE_D,
            UNIFORM,
            WHOLE_SURFACE,
            XYPLANE,
        )
        import mesh
    except ImportError as exc:
        raise RuntimeError(
            "Abaqus Python modules are unavailable. Run this script with "
            "'abaqus cae noGUI=<script.py>'. Import error: {}".format(exc)
        )

    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]

    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    mdb.Model(name=MODEL_NAME)
    model = mdb.models[MODEL_NAME]

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    L = geometry["length"]
    b = geometry["width"]
    h = geometry["total_height"]

    domain = build_domain_definition(geometry)

    x_min = float(domain["x_min_mm"])
    x_max = float(domain["x_max_mm"])
    y_min = float(domain["y_min_mm"])
    y_max = float(domain["y_max_mm"])

    tf_bottom = geometry["face_bottom"]
    hc = geometry["core_height"]

    sketch_size = 2.5 * max(L, b, h)

    sketch = model.ConstrainedSketch(
        name="__profile__",
        sheetSize=sketch_size,
    )

    sketch.rectangle(
        point1=(x_min, y_min),
        point2=(x_max, y_max),
    )

    part = model.Part(
        name="SANDWICH_BEAM",
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY,
    )

    part.BaseSolidExtrude(
        sketch=sketch,
        depth=h,
    )

    del model.sketches["__profile__"]

    # Layer partitions in the part coordinate frame: z in [0,h].
    datum_bottom_interface = part.DatumPlaneByPrincipalPlane(
        principalPlane=XYPLANE,
        offset=tf_bottom,
    )

    part.PartitionCellByDatumPlane(
        datumPlane=part.datums[datum_bottom_interface.id],
        cells=part.cells,
    )

    datum_top_interface = part.DatumPlaneByPrincipalPlane(
        principalPlane=XYPLANE,
        offset=tf_bottom + hc,
    )

    part.PartitionCellByDatumPlane(
        datumPlane=part.datums[datum_top_interface.id],
        cells=part.cells,
    )

    # Identify cells using a point strictly inside the retained domain.
    x_probe = 0.5 * (x_min + x_max)
    y_probe = 0.5 * (y_min + y_max)

    bottom_cells = part.cells.findAt(
        ((x_probe, y_probe, 0.5 * tf_bottom),)
    )

    core_cells = part.cells.findAt(
        ((x_probe, y_probe, tf_bottom + 0.5 * hc),)
    )

    top_cells = part.cells.findAt(
        (
            (
                x_probe,
                y_probe,
                tf_bottom + hc + 0.5 * geometry["face_top"],
            ),
        )
    )

    part.Set(
        name="SET_FACE_BOTTOM",
        cells=bottom_cells,
    )

    part.Set(
        name="SET_CORE",
        cells=core_cells,
    )

    part.Set(
        name="SET_FACE_TOP",
        cells=top_cells,
    )

    # Outer surfaces are useful later for postprocessing and experiment-like
    # fixture extensions.
    bottom_surface_faces = part.faces.findAt(
        ((x_probe, y_probe, 0.0),)
    )
    top_surface_faces = part.faces.findAt(
        ((x_probe, y_probe, h),)
    )

    part.Surface(
        name="SURF_BOTTOM",
        side1Faces=bottom_surface_faces,
    )

    part.Surface(
        name="SURF_TOP",
        side1Faces=top_surface_faces,
    )

    # -------------------------------------------------------------------------
    # Materials and sections
    # -------------------------------------------------------------------------

    face_material = model.Material(
        name="MAT_FACE_ALSI10MG"
    )

    face_material.Elastic(
        table=((float(FACE_E_MPA), float(FACE_NU)),)
    )

    core_material = model.Material(
        name="MAT_CORE_TPMS_HOM"
    )

    core_material.Elastic(
        type=ANISOTROPIC,
        table=(tuple(core_data["abaqus_constants"]),),
    )

    model.HomogeneousSolidSection(
        name="SEC_FACE",
        material="MAT_FACE_ALSI10MG",
        thickness=None,
    )

    model.HomogeneousSolidSection(
        name="SEC_CORE",
        material="MAT_CORE_TPMS_HOM",
        thickness=None,
    )

    part.SectionAssignment(
        region=part.sets["SET_FACE_BOTTOM"],
        sectionName="SEC_FACE",
    )

    part.SectionAssignment(
        region=part.sets["SET_CORE"],
        sectionName="SEC_CORE",
    )

    # Abaqus requires an explicit material orientation for anisotropic solid
    # elasticity. C_model has already been rotated from the unit-cell/database
    # frame into the structural beam frame (global X/Y/Z), so the correct
    # Abaqus material orientation is the GLOBAL system with zero additional
    # rotation. Do not apply Q_source_to_model a second time here.
    part.MaterialOrientation(
        region=part.sets["SET_CORE"],
        orientationType=GLOBAL,
        axis=AXIS_3,
        additionalRotationType=ROTATION_NONE,
        angle=0.0,
        localCsys=None,
        fieldName="",
        stackDirection=STACK_3,
    )

    part.SectionAssignment(
        region=part.sets["SET_FACE_TOP"],
        sectionName="SEC_FACE",
    )

    # -------------------------------------------------------------------------
    # Structured mesh
    # -------------------------------------------------------------------------

    ne_cell = int(CORE_ELEMENTS_PER_CELL_EDGE)
    ne_face = int(FACE_ELEMENTS_THROUGH_THICKNESS)

    if ne_cell <= 0:
        raise ValueError(
            "CORE_ELEMENTS_PER_CELL_EDGE must be positive."
        )

    if ne_face <= 0:
        raise ValueError(
            "FACE_ELEMENTS_THROUGH_THICKNESS must be positive."
        )

    nx_full = (
        2 * geometry["n_overhang"]
        + 2 * geometry["n_shear"]
        + geometry["n_moment"]
    ) * ne_cell
    ny_full = geometry["n_width"] * ne_cell

    if DOMAIN_MODE == "quarter_xy":
        if nx_full % 2 != 0 or ny_full % 2 != 0:
            raise ValueError(
                "quarter_xy requires even full-model X and Y mesh counts; "
                "got nx_full={}, ny_full={}.".format(nx_full, ny_full)
            )
        nx = nx_full // 2
        ny = ny_full // 2
    else:
        nx = nx_full
        ny = ny_full

    nz_core = geometry["n_height"] * ne_cell

    edge_groups = classify_axis_aligned_edges(
        part,
        geometry,
    )

    part.seedEdgeByNumber(
        edges=edge_groups["x"],
        number=nx,
        constraint=FIXED,
    )

    part.seedEdgeByNumber(
        edges=edge_groups["y"],
        number=ny,
        constraint=FIXED,
    )

    if edge_groups["z_bottom"]:
        part.seedEdgeByNumber(
            edges=edge_groups["z_bottom"],
            number=ne_face,
            constraint=FIXED,
        )

    part.seedEdgeByNumber(
        edges=edge_groups["z_core"],
        number=nz_core,
        constraint=FIXED,
    )

    if edge_groups["z_top"]:
        part.seedEdgeByNumber(
            edges=edge_groups["z_top"],
            number=ne_face,
            constraint=FIXED,
        )

    part.setMeshControls(
        regions=part.cells,
        elemShape=HEX,
        technique=STRUCTURED,
    )

    element_key = str(ELEMENT_TYPE).strip().upper()

    element_codes = {
        "C3D8I": C3D8I,
        "C3D8R": C3D8R,
        "C3D20R": C3D20R,
    }

    if element_key not in element_codes:
        raise ValueError(
            "ELEMENT_TYPE must be one of {}; got {!r}.".format(
                sorted(element_codes),
                ELEMENT_TYPE,
            )
        )

    elem_type = mesh.ElemType(
        elemCode=element_codes[element_key],
        elemLibrary=STANDARD,
    )

    part.setElementType(
        regions=(part.cells,),
        elemTypes=(elem_type,),
    )

    part.generateMesh()

    actual_node_count = len(part.nodes)
    actual_element_count = len(part.elements)

    expected_element_count = (
        nx
        * ny
        * (
            nz_core
            + 2 * ne_face
        )
    )

    if actual_element_count != expected_element_count:
        raise RuntimeError(
            "Unexpected structured element count: generated {}, expected {} "
            "from nx*ny*(nz_core+2*nz_face).".format(
                actual_element_count,
                expected_element_count,
            )
        )

    # -------------------------------------------------------------------------
    # Assembly and centered structural coordinate system
    # -------------------------------------------------------------------------

    assembly = model.rootAssembly

    assembly.DatumCsysByDefault(
        CARTESIAN
    )

    instance = assembly.Instance(
        name="SANDWICH_BEAM-1",
        part=part,
        dependent=ON,
    )

    # Put the physical sandwich mid-plane at global Z = 0.
    assembly.translate(
        instanceList=(instance.name,),
        vector=(0.0, 0.0, -0.5 * h),
    )

    assembly.regenerate()

    # -------------------------------------------------------------------------
    # Node sets for supports, loads, and comparison lines
    # -------------------------------------------------------------------------

    element_size = geometry["cell_size"] / float(ne_cell)

    if NODE_SELECTION_TOL_MM is None:
        selection_tol = max(
            1.0e-7,
            1.0e-5 * min(
                element_size,
                geometry["face_bottom"] / float(ne_face),
                geometry["face_top"] / float(ne_face),
            ),
        )
    else:
        selection_tol = float(NODE_SELECTION_TOL_MM)

    symmetry_data = {
        "mode": DOMAIN_MODE,
        "x_midspan_node_count": 0,
        "y_centerplane_node_count": 0,
    }

    if DOMAIN_MODE == "quarter_xy":
        symmetry_x_labels = labels_near_coordinates(
            instance.nodes,
            x=geometry["midspan"],
            tol=selection_tol,
        )
        symmetry_y_labels = labels_near_coordinates(
            instance.nodes,
            y=0.0,
            tol=selection_tol,
        )

        make_node_set_from_labels(
            assembly,
            instance,
            "SET_SYMMETRY_X_MIDSPAN",
            symmetry_x_labels,
        )
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_SYMMETRY_Y_CENTER",
            symmetry_y_labels,
        )

        symmetry_data["x_midspan_node_count"] = len(symmetry_x_labels)
        symmetry_data["y_centerplane_node_count"] = len(symmetry_y_labels)

    z_bottom = geometry["z_bottom"]
    z_top = geometry["z_top"]

    section_coupling_data = None

    if FIXTURE_MODE in (
        "outer_surface_lines",
        "midplane_lines",
    ):
        if FIXTURE_MODE == "outer_surface_lines":
            support_z = z_bottom
            load_z = z_top
        else:
            support_z = 0.0
            load_z = 0.0

        support_left_labels = labels_near_coordinates(
            instance.nodes,
            x=geometry["support_left"],
            z=support_z,
            tol=selection_tol,
        )
        support_right_labels = labels_near_coordinates(
            instance.nodes,
            x=geometry["support_right"],
            z=support_z,
            tol=selection_tol,
        )
        load_left_labels = labels_near_coordinates(
            instance.nodes,
            x=geometry["load_left"],
            z=load_z,
            tol=selection_tol,
        )
        load_right_labels = labels_near_coordinates(
            instance.nodes,
            x=geometry["load_right"],
            z=load_z,
            tol=selection_tol,
        )

        make_node_set_from_labels(
            assembly,
            instance,
            "SET_SUPPORT_LEFT_LINE",
            support_left_labels,
        )
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_SUPPORT_RIGHT_LINE",
            support_right_labels,
        )
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_LOAD_LEFT_LINE",
            load_left_labels,
        )
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_LOAD_RIGHT_LINE",
            load_right_labels,
        )

    elif FIXTURE_MODE == "section_distributing":
        support_z = None
        load_z = None

        support_left_cloud = labels_near_coordinates(
            instance.nodes,
            x=geometry["support_left"],
            tol=selection_tol,
        )
        load_left_cloud = labels_near_coordinates(
            instance.nodes,
            x=geometry["load_left"],
            tol=selection_tol,
        )

        if DOMAIN_MODE == "quarter_xy":
            support_right_cloud = ()
            load_right_cloud = ()
            cloud_definitions = (
                ("SUPPORT_LEFT", geometry["support_left"], support_left_cloud),
                ("LOAD_LEFT", geometry["load_left"], load_left_cloud),
            )
        else:
            support_right_cloud = labels_near_coordinates(
                instance.nodes,
                x=geometry["support_right"],
                tol=selection_tol,
            )
            load_right_cloud = labels_near_coordinates(
                instance.nodes,
                x=geometry["load_right"],
                tol=selection_tol,
            )
            cloud_definitions = (
                ("SUPPORT_LEFT", geometry["support_left"], support_left_cloud),
                ("SUPPORT_RIGHT", geometry["support_right"], support_right_cloud),
                ("LOAD_LEFT", geometry["load_left"], load_left_cloud),
                ("LOAD_RIGHT", geometry["load_right"], load_right_cloud),
            )

        section_coupling_data = {}

        for fixture_name, x_coord, cloud_labels in cloud_definitions:
            cloud_set_name = "SET_{}_SECTION".format(fixture_name)

            make_node_set_from_labels(
                assembly,
                instance,
                cloud_set_name,
                cloud_labels,
            )

            rp_feature = assembly.ReferencePoint(
                point=(
                    float(x_coord),
                    0.0,
                    0.0,
                )
            )
            rp = assembly.referencePoints[
                rp_feature.id
            ]

            rp_set_name = "SET_{}_RP".format(fixture_name)
            assembly.Set(
                name=rp_set_name,
                referencePoints=(rp,),
            )

            control_region = regionToolset.Region(
                referencePoints=(rp,),
            )
            cloud_nodes = instance.nodes.sequenceFromLabels(
                labels=tuple(
                    sorted(
                        set(
                            int(label)
                            for label in cloud_labels
                        )
                    )
                )
            )
            cloud_region = regionToolset.Region(
                nodes=cloud_nodes,
            )

            coupling_name = "COUPLE_{}_SECTION".format(
                fixture_name
            )

            model.Coupling(
                name=coupling_name,
                controlPoint=control_region,
                surface=cloud_region,
                influenceRadius=WHOLE_SURFACE,
                couplingType=DISTRIBUTING,
                weightingMethod=UNIFORM,
                localCsys=None,
                u1=ON,
                u2=ON,
                u3=ON,
                ur1=OFF,
                ur2=OFF,
                ur3=OFF,
            )

            section_coupling_data[
                fixture_name.lower()
            ] = {
                "x_mm": float(x_coord),
                "cloud_set": cloud_set_name,
                "cloud_node_count": len(cloud_labels),
                "rp_set": rp_set_name,
                "reference_point_id": int(rp_feature.id),
                "coupling_name": coupling_name,
                "coupling_type": "continuum_distributing",
                "translational_dofs": [1, 2, 3],
                "rotational_dofs": [],
                "weighting_method": "UNIFORM",
            }

        # Preserve the postprocessor's support reaction set names. In fxsec the
        # reaction-carrying regions are the two support reference points.
        assembly.Set(
            name="SET_SUPPORT_LEFT_LINE",
            referencePoints=(
                assembly.referencePoints[
                    section_coupling_data[
                        "support_left"
                    ]["reference_point_id"]
                ],
            ),
        )
        if DOMAIN_MODE == "full":
            assembly.Set(
                name="SET_SUPPORT_RIGHT_LINE",
                referencePoints=(
                    assembly.referencePoints[
                        section_coupling_data[
                            "support_right"
                        ]["reference_point_id"]
                    ],
                ),
            )

        support_left_labels = ()
        support_right_labels = ()
        load_left_labels = ()
        load_right_labels = ()

    else:
        raise RuntimeError(
            "Unsupported FIXTURE_MODE {!r}.".format(
                FIXTURE_MODE
            )
        )

    # Midspan OUTER lines for convenient displacement / face-strain extraction.
    #
    # The midspan location is exactly aligned with the structured X mesh for
    # the cell-count parameterization used here, so the same tight coordinate
    # tolerance used for the load/support lines is appropriate. Do NOT use a
    # tolerance based on the 5 mm core element size here: the face sheets are
    # only 1 mm thick and such a tolerance can accidentally include interface
    # nodes together with the outer-surface nodes.
    mid_top_labels = labels_near_coordinates(
        instance.nodes,
        x=geometry["midspan"],
        z=z_top,
        tol=selection_tol,
    )

    mid_bottom_labels = labels_near_coordinates(
        instance.nodes,
        x=geometry["midspan"],
        z=z_bottom,
        tol=selection_tol,
    )

    if mid_top_labels:
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_MIDSPAN_TOP",
            mid_top_labels,
        )

    if mid_bottom_labels:
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_MIDSPAN_BOTTOM",
            mid_bottom_labels,
        )

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    model.StaticStep(
        name=STEP_NAME,
        previous="Initial",
        timePeriod=1.0,
        nlgeom=OFF,
        initialInc=1.0,
        minInc=1.0e-8,
        maxInc=1.0,
        maxNumInc=20,
    )

    # -------------------------------------------------------------------------
    # Computational symmetry
    # -------------------------------------------------------------------------

    if DOMAIN_MODE == "quarter_xy":
        model.DisplacementBC(
            name="BC_SYMMETRY_X_MIDSPAN",
            createStepName="Initial",
            region=assembly.sets["SET_SYMMETRY_X_MIDSPAN"],
            u1=0.0,
        )
        model.DisplacementBC(
            name="BC_SYMMETRY_Y_CENTER",
            createStepName="Initial",
            region=assembly.sets["SET_SYMMETRY_Y_CENTER"],
            u2=0.0,
        )

    # -------------------------------------------------------------------------
    # Simply-supported fixture with minimal in-plane stabilization
    # -------------------------------------------------------------------------

    if FIXTURE_MODE == "section_distributing":
        support_left_region = assembly.sets[
            "SET_SUPPORT_LEFT_RP"
        ]

        model.DisplacementBC(
            name="BC_SUPPORT_LEFT_Z",
            createStepName="Initial",
            region=support_left_region,
            u3=0.0,
        )

        if DOMAIN_MODE == "full":
            support_right_region = assembly.sets[
                "SET_SUPPORT_RIGHT_RP"
            ]
            model.DisplacementBC(
                name="BC_SUPPORT_RIGHT_Z",
                createStepName="Initial",
                region=support_right_region,
                u3=0.0,
            )
            model.DisplacementBC(
                name="BC_INPLANE_PIN",
                createStepName="Initial",
                region=support_left_region,
                u1=0.0,
                u2=0.0,
            )
            model.DisplacementBC(
                name="BC_INPLANE_GUIDE",
                createStepName="Initial",
                region=support_right_region,
                u2=0.0,
            )
        else:
            support_right_region = None

        left_center_label = None
        right_center_label = None

    else:
        model.DisplacementBC(
            name="BC_SUPPORT_LEFT_Z",
            createStepName="Initial",
            region=assembly.sets["SET_SUPPORT_LEFT_LINE"],
            u3=0.0,
        )
        model.DisplacementBC(
            name="BC_SUPPORT_RIGHT_Z",
            createStepName="Initial",
            region=assembly.sets["SET_SUPPORT_RIGHT_LINE"],
            u3=0.0,
        )

        left_center_label = nearest_line_node_label(
            instance,
            support_left_labels,
            target_y=0.0,
        )
        right_center_label = nearest_line_node_label(
            instance,
            support_right_labels,
            target_y=0.0,
        )

        make_node_set_from_labels(
            assembly,
            instance,
            "SET_SUPPORT_LEFT_CENTER",
            (left_center_label,),
        )
        make_node_set_from_labels(
            assembly,
            instance,
            "SET_SUPPORT_RIGHT_CENTER",
            (right_center_label,),
        )

        model.DisplacementBC(
            name="BC_INPLANE_PIN",
            createStepName="Initial",
            region=assembly.sets["SET_SUPPORT_LEFT_CENTER"],
            u1=0.0,
            u2=0.0,
        )
        model.DisplacementBC(
            name="BC_INPLANE_GUIDE",
            createStepName="Initial",
            region=assembly.sets["SET_SUPPORT_RIGHT_CENTER"],
            u2=0.0,
        )

    # -------------------------------------------------------------------------
    # Symmetric four-point loading
    # -------------------------------------------------------------------------

    half_total = 0.5 * float(TOTAL_APPLIED_LOAD_N)

    if FIXTURE_MODE == "section_distributing":
        if DOMAIN_MODE == "quarter_xy":
            left_resultant = (
                -float(TOTAL_APPLIED_LOAD_N)
                * float(domain["applied_load_fraction_of_full"])
            )
        else:
            left_resultant = -half_total

        model.ConcentratedForce(
            name="LOAD_LEFT_SECTION_RESULTANT",
            createStepName=STEP_NAME,
            region=assembly.sets["SET_LOAD_LEFT_RP"],
            cf3=left_resultant,
        )

        left_load_info = {
            "type": "concentrated_force_at_reference_point",
            "set_name": "SET_LOAD_LEFT_RP",
            "resultant_N": left_resultant,
            "cf3_N": left_resultant,
        }

        if DOMAIN_MODE == "full":
            model.ConcentratedForce(
                name="LOAD_RIGHT_SECTION_RESULTANT",
                createStepName=STEP_NAME,
                region=assembly.sets["SET_LOAD_RIGHT_RP"],
                cf3=-half_total,
            )
            right_load_info = {
                "type": "concentrated_force_at_reference_point",
                "set_name": "SET_LOAD_RIGHT_RP",
                "resultant_N": -half_total,
                "cf3_N": -half_total,
            }
        else:
            right_load_info = None

    else:
        left_load_info = apply_line_load(
            model=model,
            assembly=assembly,
            instance=instance,
            line_name="SET_LOAD_LEFT_LINE",
            labels=load_left_labels,
            line_resultant=half_total,
            step_name=STEP_NAME,
        )
        right_load_info = apply_line_load(
            model=model,
            assembly=assembly,
            instance=instance,
            line_name="SET_LOAD_RIGHT_LINE",
            labels=load_right_labels,
            line_resultant=half_total,
            step_name=STEP_NAME,
        )

    total_discrete_load = float(
        left_load_info["resultant_N"]
    )
    if right_load_info is not None:
        total_discrete_load += float(
            right_load_info["resultant_N"]
        )

    model_applied_load_magnitude = (
        float(TOTAL_APPLIED_LOAD_N)
        * float(domain["applied_load_fraction_of_full"])
    )

    if abs(
        total_discrete_load + model_applied_load_magnitude
    ) > 1.0e-10 * max(
        1.0,
        abs(model_applied_load_magnitude),
    ):
        raise RuntimeError(
            "Discrete load resultant does not equal the computational-domain "
            "applied load."
        )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    # Abaqus normally exposes the field-output repository as
    # model.fieldOutputRequests. Some CAE/noGUI execution contexts, however,
    # return a Model proxy without that repository attribute even though the
    # FieldOutputRequest factory method itself is available.
    #
    # Prefer modifying the default request when the repository is accessible.
    # Otherwise create a deliberately unique HOST request so that we never
    # collide with an inaccessible default F-Output-1 request.
    field_output_repository = getattr(
        model,
        "fieldOutputRequests",
        None,
    )

    if (
        field_output_repository is not None
        and "F-Output-1" in field_output_repository
    ):
        field_output_repository["F-Output-1"].setValues(
            variables=FIELD_OUTPUT_VARIABLES,
            frequency=LAST_INCREMENT,
        )
        field_output_request_name = "F-Output-1"
        field_output_request_mode = "modified_default"
    else:
        field_output_request_name = "F-Output-HOST"

        model.FieldOutputRequest(
            name=field_output_request_name,
            createStepName=STEP_NAME,
            variables=FIELD_OUTPUT_VARIABLES,
            frequency=LAST_INCREMENT,
        )

        field_output_request_mode = "created_explicit"

    # -------------------------------------------------------------------------
    # Job
    # -------------------------------------------------------------------------

    # Abaqus' Job object defaults to one Explicit domain. With numCpus > 1,
    # some Abaqus/CAE kernels validate the DOMAIN relationship already when
    # the Job object is created, even for this Abaqus/Standard model. Keep the
    # domain count consistent with the CPU count so the generic Job definition
    # is valid. numDomains is only relevant to Abaqus/Explicit execution.
    job_scratch_directory = ABAQUS_SCRATCH_ROOT / "tmp"
    job_scratch_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    job = mdb.Job(
        name=job_name,
        model=MODEL_NAME,
        type=ANALYSIS,
        memory=int(MEMORY_PERCENT),
        memoryUnits=PERCENTAGE,
        numCpus=int(NUM_CPUS),
        numDomains=int(NUM_CPUS),
        multiprocessingMode=DEFAULT,
        scratch=str(job_scratch_directory),
    )

    return {
        "mdb": mdb,
        "model": model,
        "part": part,
        "assembly": assembly,
        "instance": instance,
        "job": job,
        "job_scratch_directory": str(job_scratch_directory),
        "selection_tol": selection_tol,
        "mesh_counts": {
            "nx": nx,
            "ny": ny,
            "nx_full_equivalent": nx_full,
            "ny_full_equivalent": ny_full,
            "nz_core": nz_core,
            "nz_face_bottom": ne_face,
            "nz_face_top": ne_face,
            "core_elements_per_nominal_cell_volume": ne_cell**3,
            "actual_node_count": actual_node_count,
            "actual_element_count": actual_element_count,
            "expected_structured_element_count": expected_element_count,
        },
        "fixture": {
            "mode": FIXTURE_MODE,
            "support_z_mm": (
                None
                if support_z is None
                else float(support_z)
            ),
            "load_z_mm": (
                None
                if load_z is None
                else float(load_z)
            ),
            "support_left_node_count": (
                1
                if FIXTURE_MODE == "section_distributing"
                else len(support_left_labels)
            ),
            "support_right_node_count": (
                (
                    0
                    if DOMAIN_MODE == "quarter_xy"
                    else 1
                )
                if FIXTURE_MODE == "section_distributing"
                else len(support_right_labels)
            ),
            "load_left_node_count": (
                1
                if FIXTURE_MODE == "section_distributing"
                else len(load_left_labels)
            ),
            "load_right_node_count": (
                (
                    0
                    if DOMAIN_MODE == "quarter_xy"
                    else 1
                )
                if FIXTURE_MODE == "section_distributing"
                else len(load_right_labels)
            ),
            "left_load": left_load_info,
            "right_load": right_load_info,
            "model_applied_load_N": model_applied_load_magnitude,
            "left_center_stabilization_node": left_center_label,
            "right_center_stabilization_node": right_center_label,
            "section_couplings": section_coupling_data,
            "midspan_top_node_count": len(mid_top_labels),
            "midspan_bottom_node_count": len(mid_bottom_labels),
        },
        "domain": domain,
        "symmetry": symmetry_data,
        "output_request": {
            "name": field_output_request_name,
            "mode": field_output_request_mode,
            "variables": list(FIELD_OUTPUT_VARIABLES),
            "frequency": "LAST_INCREMENT",
        },
    }


# =============================================================================
# MANIFEST
# =============================================================================

def build_manifest(
    core_data,
    geometry,
    abaqus_data,
    case_identity,
    output_directory,
):
    row = core_data["row"]
    convention = core_data["convention"]
    oriented = core_data["oriented"]
    job_name = case_identity["job_name"]

    state_roles = []
    if row_value(row, "is_initial", 0):
        state_roles.append("initial")
    if row_value(row, "is_best", 0):
        state_roles.append("best")
    if row_value(row, "is_final", 0):
        state_roles.append("final")
    if not state_roles:
        state_roles.append("iteration")

    profile_stations = [
        {
            "name": "support_left",
            "x_mm": geometry["support_left"],
            "purpose": "support boundary/local transverse-normal-stress zone",
        },
        {
            "name": "left_shear_mid",
            "x_mm": 0.5 * (
                geometry["support_left"] + geometry["load_left"]
            ),
            "purpose": "bulk shear-span comparison",
        },
        {
            "name": "load_left",
            "x_mm": geometry["load_left"],
            "purpose": "load-introduction/local transverse-normal-stress zone",
        },
        {
            "name": "midspan",
            "x_mm": geometry["midspan"],
            "purpose": "constant-moment/global bending comparison",
        },
        {
            "name": "load_right",
            "x_mm": geometry["load_right"],
            "purpose": "load-introduction/local transverse-normal-stress zone",
        },
        {
            "name": "right_shear_mid",
            "x_mm": 0.5 * (
                geometry["load_right"] + geometry["support_right"]
            ),
            "purpose": "bulk shear-span comparison",
        },
        {
            "name": "support_right",
            "x_mm": geometry["support_right"],
            "purpose": "support boundary/local transverse-normal-stress zone",
        },
    ]

    domain = dict(
        abaqus_data["domain"]
    )

    if DOMAIN_MODE == "quarter_xy":
        x_max_domain = float(
            domain["x_max_mm"]
        )
        profile_stations = [
            station
            for station in profile_stations
            if float(station["x_mm"]) <= x_max_domain + 1.0e-12
        ]

    reflection_error_x = constitutive_reflection_relative_error(
        core_data["C_model_symmetric"],
        "x",
    )
    reflection_error_y = constitutive_reflection_relative_error(
        core_data["C_model_symmetric"],
        "y",
    )

    domain["constitutive_reflection_relative_error_x"] = reflection_error_x
    domain["constitutive_reflection_relative_error_y"] = reflection_error_y
    domain["exact_material_symmetry"] = (
        reflection_error_x <= 1.0e-12
        and reflection_error_y <= 1.0e-12
    )
    domain["interpretation"] = (
        "quarter_xy is an imposed symmetry approximation when the "
        "constitutive reflection errors are nonzero."
        if DOMAIN_MODE == "quarter_xy"
        else "No computational symmetry reduction is imposed."
    )

    comparison_metrics = {
        "primary_global": [
            "transverse deflection w(x), especially midspan",
            "longitudinal displacement u_x(x,z)",
            "face-sheet longitudinal strain epsilon_xx(x)",
            "face-sheet longitudinal stress sigma_xx(x)",
            "bending resultant M(x) from FE section integration",
            "transverse shear resultant Q_x(x) from FE section integration",
            "support reactions and global equilibrium",
        ],
        "primary_through_thickness": [
            "epsilon_xx(z) and sigma_xx(z) at selected x stations",
            "gamma_xz(z) and tau_xz(z) in the shear span",
            "epsilon_zz(z) and sigma_zz(z), especially near load/support lines",
            "interface values at both face/core interfaces",
        ],
        "diagnostic_3d": [
            "width-averaged response versus width-center response",
            "sigma_yy, tau_xy, tau_yz as measures of departure from 1-D assumptions",
            "lateral displacement u_y and anticlastic/anisotropic effects",
            "local boundary-layer decay away from supports and load lines",
        ],
        "energetic": [
            "total elastic strain energy",
            "normal/bending contribution based on sigma_xx*epsilon_xx",
            "transverse-shear contribution based on tau_xz*gamma_xz",
            "transverse-normal contribution based on sigma_zz*epsilon_zz",
            "energy fractions for theory-regime classification",
        ],
        "numerical_quality": [
            "mesh-convergence of w_mid, face strain, tau_xz and sigma_zz",
            "reaction-force imbalance",
            "internal-versus-external work consistency",
            "DOF count, wall time and memory",
        ],
    }

    query_module_path = (
        Path(QUERY_MODULE_DIRECTORY)
        / "query_tpms_stiffness_database_rotated.py"
    )

    manifest = {
        "schema": "host_4pb_homogenized_beam_v3",
        "model_implementation_revision": int(
            MODEL_IMPLEMENTATION_REVISION
        ),
        "case": {
            "case_id": case_identity["case_id"],
            "case_hash": case_identity["case_hash"],
            "case_hash_full_sha256": case_identity[
                "case_hash_full_sha256"
            ],
            "canonical_configuration": case_identity[
                "configuration"
            ],
        },
        "job_name": job_name,
        "model_name": MODEL_NAME,
        "step_name": STEP_NAME,
        "runtime_overrides": dict(RUNTIME_OVERRIDES),
        "paths": {
            "repository_root": str(REPO_ROOT),
            "local_path_config": str(LOCAL_PATH_CONFIG),
            "abaqus_scratch_root": str(ABAQUS_SCRATCH_ROOT),
            "simulation_family_root": str(OUTPUT_ROOT),
            "case_directory": str(output_directory),
            "tpms_stiffness_database": str(DATABASE_PATH),
        },
        "provenance": {
            "git": git_provenance(REPO_ROOT),
            "builder_script": str(BUILDER_SCRIPT_PATH),
            "builder_script_sha256": sha256_file(
                BUILDER_SCRIPT_PATH
            ),
            "query_module": str(query_module_path),
            "query_module_sha256": sha256_file(
                query_module_path
            ),
        },
        "units": {
            "length": "mm",
            "force": "N",
            "stress_stiffness": "MPa",
        },
        "coordinates": {
            "X": "beam longitudinal",
            "Y": "specimen width",
            "Z": "sandwich thickness",
            "sandwich_midplane_global_Z_mm": 0.0,
        },
        "database_query": {
            "database_path": str(DATABASE_PATH),
            "campaign_type": row_value(
                row, "campaign_type", CORE_CAMPAIGN_TYPE
            ),
            "campaign_name": row_value(row, "campaign_name"),
            "topology": row_value(row, "topology", CORE_TOPOLOGY),
            "relative_density_target": row_value(
                row, "rd_target_fraction"
            ),
            "relative_density_actual": row_value(
                row, "rd_actual_fraction"
            ),
            "optimization_load_case": row_value(
                row, "load_case", CORE_LOAD_CASE
            ),
            "target_metric": row_value(row, "target_metric_name"),
            "requested_state": CORE_STATE,
            "iteration": row_value(row, "iteration"),
            "state_role": state_roles,
            "eval_id": row_value(row, "eval_id"),
        },
        "orientation": {
            "preset": CORE_ORIENTATION_PRESET,
            "convention": (
                "v_model = Q_source_to_model * v_source"
            ),
            "abaqus_material_orientation": {
                "region": "SET_CORE",
                "system": "GLOBAL",
                "additional_rotation_deg": 0.0,
                "reason": (
                    "C_hom_model is already expressed in structural "
                    "global X/Y/Z; the Abaqus material axes therefore "
                    "coincide with global X/Y/Z."
                ),
            },
            "model_x_from_unit_cell": core_data["axis_map"][
                "model_x_from_uc"
            ],
            "model_y_from_unit_cell": core_data["axis_map"][
                "model_y_from_uc"
            ],
            "model_z_from_unit_cell": core_data["axis_map"][
                "model_z_from_uc"
            ],
            "Q_source_to_model": core_data["Q"],
        },
        "constitutive": {
            "canonical_voigt_order": oriented[
                "canonical_voigt_order"
            ],
            "source_voigt_order": oriented[
                "source_voigt_order"
            ],
            "shear_convention": row_value(
                convention,
                "shear_convention",
                "engineering",
            ),
            "C_hom_source_raw_MPa": oriented["C_source_raw"],
            "C_hom_model_MPa": core_data["C_model_symmetric"],
            "abaqus_anisotropic_21_constants_MPa": list(
                core_data["abaqus_constants"]
            ),
            "rotated_engineering_constants_MPa": oriented[
                "engineering_constants_model"
            ],
            "relative_symmetry_error_before_cleanup": core_data[
                "relative_symmetry_error_before_cleanup"
            ],
            "face_material": {
                "E_MPa": float(FACE_E_MPA),
                "nu": float(FACE_NU),
            },
        },
        "geometry": geometry,
        "domain": domain,
        "mesh": {
            "element_type": str(ELEMENT_TYPE).upper(),
            "core_elements_per_cell_edge": int(
                CORE_ELEMENTS_PER_CELL_EDGE
            ),
            "face_elements_through_thickness": int(
                FACE_ELEMENTS_THROUGH_THICKNESS
            ),
            "derived_counts": abaqus_data["mesh_counts"],
            "node_selection_tolerance_mm": abaqus_data[
                "selection_tol"
            ],
        },
        "fixture": {
            "type": {
                "outer_surface_lines": (
                    "idealized_outer_surface_line_4PB"
                ),
                "midplane_lines": (
                    "idealized_reference_midplane_line_4PB"
                ),
                "section_distributing": (
                    "idealized_full_section_distributing_4PB"
                ),
            }[FIXTURE_MODE],
            "mode": FIXTURE_MODE,
            "total_applied_load_N": float(TOTAL_APPLIED_LOAD_N),
            "model_applied_load_N": float(
                abaqus_data["fixture"]["model_applied_load_N"]
            ),
            "load_direction": "-Z",
            "consistent_line_load_weights": bool(
                USE_CONSISTENT_LINE_LOAD_WEIGHTS
            ),
            "details": abaqus_data["fixture"],
            "note": {
                "outer_surface_lines": (
                    "Physical-test-like idealization without rigid rollers/"
                    "contact. Local stress fields at the outer-surface support/"
                    "load lines contain through-thickness indentation boundary "
                    "layers and should be separated from bulk theory "
                    "comparisons."
                ),
                "midplane_lines": (
                    "Diagnostic only: support/load node lines are moved to "
                    "z=0, but the loading remains local and is not equivalent "
                    "to a generalized cross-sectional beam resultant."
                ),
                "section_distributing": (
                    "Theory-compatibility reference: centroidal reference "
                    "points are continuum-distributing coupled to complete "
                    "cross-sectional node clouds. Only translations are "
                    "coupled; rotations are released so sections remain free "
                    "to deform and warp."
                ),
            }[FIXTURE_MODE],
        },
        "normalization": normalization_data(geometry),
        "recommended_profile_stations": profile_stations,
        "comparison_metrics": comparison_metrics,
        "field_outputs": list(FIELD_OUTPUT_VARIABLES),
        "field_output_request": abaqus_data["output_request"],
        "analysis": {
            "linear_elastic": True,
            "geometrically_linear": True,
            "nlgeom": False,
            "solver": "Abaqus/Standard",
            "num_cpus": int(NUM_CPUS),
            "num_domains_job_definition": int(NUM_CPUS),
            "multiprocessing_mode": "DEFAULT",
            "job_scratch_directory": abaqus_data[
                "job_scratch_directory"
            ],
        },
    }

    return manifest


# =============================================================================
# MAIN
# =============================================================================

def main():
    geometry = derive_geometry()

    print("=" * 88)
    print("HOST / ABAQUS PARAMETERIZED FOUR-POINT-BENDING BUILDER")
    print("=" * 88)
    print()
    print("[1/5] Querying homogenized TPMS stiffness database...")

    core_data = load_core_material()

    row = core_data["row"]

    print(
        "  topology        : {}".format(
            row_value(row, "topology", CORE_TOPOLOGY)
        )
    )
    print(
        "  RD target       : {}".format(
            row_value(row, "rd_target_fraction")
        )
    )
    print(
        "  load/campaign   : {}".format(
            row_value(row, "load_case", CORE_LOAD_CASE)
        )
    )
    print(
        "  target metric   : {}".format(
            row_value(row, "target_metric_name")
        )
    )
    print(
        "  requested state : {}".format(CORE_STATE)
    )
    print(
        "  iteration       : {}".format(
            row_value(row, "iteration")
        )
    )
    print()
    print(
        "  orientation     : X<-{}, Y<-{}, Z<-{}".format(
            core_data["axis_map"]["model_x_from_uc"],
            core_data["axis_map"]["model_y_from_uc"],
            core_data["axis_map"]["model_z_from_uc"],
        )
    )

    rotated_constants = core_data["oriented"][
        "engineering_constants_model"
    ]

    print()
    print("  rotated model-frame constants [MPa]")
    print(
        "    Ex={:.6f}, Ey={:.6f}, Ez={:.6f}".format(
            rotated_constants["Ex"],
            rotated_constants["Ey"],
            rotated_constants["Ez"],
        )
    )
    print(
        "    Gxy={:.6f}, Gxz={:.6f}, Gyz={:.6f}".format(
            rotated_constants["Gxy"],
            rotated_constants["Gxz"],
            rotated_constants["Gyz"],
        )
    )

    print()
    print("[2/5] Derived four-point-bending geometry...")
    print(
        "  L x b x h       : {:.3f} x {:.3f} x {:.3f} mm".format(
            geometry["length"],
            geometry["width"],
            geometry["total_height"],
        )
    )
    print(
        "  core height      : {:.3f} mm".format(
            geometry["core_height"]
        )
    )
    print(
        "  support span S   : {:.3f} mm".format(
            geometry["support_span"]
        )
    )
    print(
        "  shear span a     : {:.3f} mm".format(
            geometry["shear_span"]
        )
    )
    print(
        "  load span        : {:.3f} mm".format(
            geometry["load_span"]
        )
    )

    case_identity = build_case_identity(
        core_data,
        geometry,
    )

    job_name = case_identity["job_name"]

    output_directory = prepare_case_directory(
        OUTPUT_ROOT,
        case_identity["case_id"],
    )

    print()
    print("  case ID          : {}".format(
        case_identity["case_id"]
    ))
    print("  case hash        : {}".format(
        case_identity["case_hash"]
    ))
    print("  output directory : {}".format(
        output_directory
    ))

    # Abaqus writes the .inp/.odb/etc. to the current working directory.
    os.chdir(str(output_directory))

    print()
    print("[3/5] Building Abaqus model...")
    abaqus_data = build_abaqus_model(
        core_data=core_data,
        geometry=geometry,
        job_name=job_name,
    )

    manifest = build_manifest(
        core_data=core_data,
        geometry=geometry,
        abaqus_data=abaqus_data,
        case_identity=case_identity,
        output_directory=output_directory,
    )

    manifest_path = output_directory / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print()
    print("[4/5] Writing model files...")

    if SAVE_CAE:
        cae_path = output_directory / (
            "{}.cae".format(job_name)
        )

        abaqus_data["mdb"].saveAs(
            pathName=str(cae_path)
        )

        print("  CAE      : {}".format(cae_path))

    if WRITE_INPUT:
        abaqus_data["job"].writeInput()

        inp_path = output_directory / (
            "{}.inp".format(job_name)
        )

        print("  INP      : {}".format(inp_path))

    print("  manifest : {}".format(manifest_path))

    print()
    print("[5/5] Job action...")

    if SUBMIT_JOB:
        print("  submitting {} ...".format(job_name))

        abaqus_data["job"].submit()

        if WAIT_FOR_COMPLETION:
            abaqus_data["job"].waitForCompletion()
            print("  completed.")
        else:
            print("  submitted without waiting.")
    else:
        print(
            "  SUBMIT_JOB=False -> model built but not submitted."
        )

    print()
    print("=" * 88)
    print("BUILD SUMMARY")
    print("=" * 88)
    print("Job name                  : {}".format(job_name))
    print(
        "Case ID                   : {}".format(
            case_identity["case_id"]
        )
    )
    print(
        "Case directory            : {}".format(
            output_directory
        )
    )
    print(
        "Core state                : {} / itag {}".format(
            " | ".join(manifest["database_query"]["state_role"]),
            manifest["database_query"]["iteration"],
        )
    )
    print(
        "Core elements/cell volume : {}".format(
            manifest["mesh"]["derived_counts"][
                "core_elements_per_nominal_cell_volume"
            ]
        )
    )
    print(
        "Total applied load        : {:.6f} N".format(
            float(TOTAL_APPLIED_LOAD_N)
        )
    )
    print(
        "Model coordinates         : X=beam, Y=width, Z=thickness"
    )
    print()
    print(
        "The manifest already records the response quantities intended "
        "for HOST / sandwich-theory / FE comparison."
    )


if __name__ == "__main__":
    main()


