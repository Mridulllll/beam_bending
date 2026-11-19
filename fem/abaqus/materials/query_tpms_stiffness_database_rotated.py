#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query the TPMS stiffness SQLite database and optionally rotate the homogenized
stiffness tensor from the unit-cell frame into the structural/model frame.

Designed for the TPMS stiffness database used in the HOST/Abaqus workflow.

Compatible with Python 3.8+ and uses only the Python standard library.

Tensor convention
-----------------
Stress Voigt vector:
    [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_xz, tau_yz]

Engineering-strain Voigt vector:
    [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_xz, gamma_yz]

Canonical Voigt ordering:
    ["xx", "yy", "zz", "xy", "xz", "yz"]

Rotation convention
-------------------
The rotation matrix Q is defined by

    v_model = Q * v_source

where "source" is the unit-cell/database frame and "model" is the desired
HOST/Abaqus structural frame.

For axis-map mode, each MODEL axis is expressed in terms of a signed SOURCE
(unit-cell) axis.

Example:
    MODEL_X_FROM_UC = "+x"
    MODEL_Y_FROM_UC = "-z"
    MODEL_Z_FROM_UC = "+y"

means
    x_model = +x_UC
    y_model = -z_UC
    z_model = +y_UC

which maps the unit-cell y direction onto the model z direction while
preserving a proper right-handed rotation (det(Q) = +1).

Important:
A spatial translation of the coordinate origin does NOT alter a constitutive
stiffness tensor. Only the orientation/rotation of the basis is relevant here.
"""

import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Machine-specific external paths are read from config/paths.local.json.
ABAQUS_FEM_ROOT = Path(__file__).resolve().parents[1]

if str(ABAQUS_FEM_ROOT) not in sys.path:
    sys.path.insert(0, str(ABAQUS_FEM_ROOT))

from common.path_config import load_path_config

_PATHS = load_path_config(__file__)
DATABASE_PATH = _PATHS["tpms_stiffness_database"]

# "sizing_shape", "sizing_only", or None to return all matching campaigns.
CAMPAIGN_TYPE = "sizing_shape"

TOPOLOGY = "Primitive"              # Primitive / Gyroid / Diamond
RELATIVE_DENSITY = 0.12             # 0.12, 12, "12%", or "rd12"
LOAD_CASE = "compression"           # shear / sa / compression / ca

# "initial", "best", "final", or an integer iteration such as 7.
STATE = "initial"

# Matrix print precision.
MATRIX_DECIMALS = 6

# Optional JSON export of every returned matrix.
EXPORT_JSON = True
OUTPUT_DIRECTORY = DATABASE_PATH.parent / "queries"


# =============================================================================
# ORIENTATION / ROTATION CONFIGURATION
# =============================================================================

# "none"     : keep the database/unit-cell orientation unchanged
# "axis_map" : define model axes using signed unit-cell axes
# "matrix"   : provide an explicit 3x3 source-to-model rotation matrix
ORIENTATION_MODE = "axis_map"

# Recommended HOST/Abaqus structural frame:
#   model x = beam longitudinal direction
#   model y = beam width direction
#   model z = sandwich thickness direction
#
# Example below maps unit-cell y -> model z:
MODEL_X_FROM_UC = "+x"
MODEL_Y_FROM_UC = "-z"
MODEL_Z_FROM_UC = "+y"

# Used only when ORIENTATION_MODE == "matrix".
#
# Convention:
#   v_model = Q * v_source
#
# The matrix must be orthonormal and must represent a proper rotation:
#   Q^T Q = I
#   det(Q) = +1
ROTATION_MATRIX_SOURCE_TO_MODEL = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

# Numerical tolerances for validating the rotation matrix.
ROTATION_TOL = 1.0e-10

# Optional constitutive QA.
# The database already stores symmetry/positive-definiteness diagnostics.
# If True, the matrix used for rotation is replaced by 0.5*(C + C^T).
# The original raw matrix is still retained in the exported JSON.
SYMMETRIZE_BEFORE_ROTATION = False

# If True, fail if more than one database row matches the query.
# Useful later when this module is imported directly by an Abaqus builder.
REQUIRE_UNIQUE_MATCH = False


# =============================================================================
# BASIC HELPERS
# =============================================================================

VOIGT_PAIR_BY_LABEL = {
    "xx": (0, 0),
    "yy": (1, 1),
    "zz": (2, 2),
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def canonical_topology(value: Any) -> str:
    value_text = str(value).strip()
    aliases = {
        "primitive": "Primitive",
        "gyroid": "Gyroid",
        "diamond": "Diamond",
    }
    return aliases.get(value_text.lower(), value_text)


def canonical_load_case(value: Any) -> str:
    value_text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ca": "compression",
        "compression": "compression",
        "compressive": "compression",
        "sa": "shear",
        "shear": "shear",
        "pure_shear": "shear",
        "xy_shear": "shear",
    }
    return aliases.get(value_text, value_text)


def relative_density_fraction(value: Any) -> float:
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        value_text = str(value).strip().lower()
        match = re.search(r"[-+]?\d*\.?\d+", value_text)
        if not match:
            raise ValueError("Could not parse relative density: {!r}".format(value))
        numeric = float(match.group(0))

    if numeric > 1.0:
        numeric /= 100.0

    if numeric <= 0.0 or numeric >= 1.0:
        raise ValueError(
            "Relative density must resolve to a fraction between 0 and 1; got {}".format(
                numeric
            )
        )

    return numeric


def state_filter(state: Any) -> Tuple[str, List[Any]]:
    if isinstance(state, int):
        return "iteration = ?", [state]

    state_text = str(state).strip().lower()

    if state_text in {"initial", "baseline", "pure", "unoptimized"}:
        return "is_initial = 1", []
    if state_text in {"best", "optimal", "best_target"}:
        return "is_best = 1", []
    if state_text in {"final", "last"}:
        return "is_final = 1", []

    match = re.fullmatch(r"(?:itag|iter|iteration)?\s*(\d+)", state_text)
    if match:
        return "iteration = ?", [int(match.group(1))]

    raise ValueError(
        "STATE must be initial, best, final, or an iteration number; got {!r}".format(
            state
        )
    )


def format_matrix(matrix: Sequence[Sequence[float]], decimals: int) -> str:
    width = 0
    formatted = []

    for row in matrix:
        formatted_row = []
        for value in row:
            item = "{:.{}f}".format(float(value), decimals)
            formatted_row.append(item)
            width = max(width, len(item))
        formatted.append(formatted_row)

    lines = ["["]
    for row in formatted:
        lines.append(
            "  [" + ", ".join(item.rjust(width) for item in row) + "],"
        )
    lines.append("]")
    return "\n".join(lines)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())


def copy_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(value) for value in row] for row in matrix]


def transpose_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*matrix)]


def symmetrize_matrix(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(matrix)
    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = 0.5 * (
                float(matrix[i][j]) + float(matrix[j][i])
            )
    return result


def max_abs_matrix_difference(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> float:
    maximum = 0.0
    for row_a, row_b in zip(a, b):
        for value_a, value_b in zip(row_a, row_b):
            maximum = max(maximum, abs(float(value_a) - float(value_b)))
    return maximum


# =============================================================================
# ROTATION MATRIX HELPERS
# =============================================================================

def _parse_signed_axis(token: str) -> Tuple[int, float]:
    text = str(token).strip().lower().replace(" ", "")

    aliases = {
        "x": "+x",
        "y": "+y",
        "z": "+z",
    }
    text = aliases.get(text, text)

    if text not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(
            "Axis token must be one of +x, -x, +y, -y, +z, -z; got {!r}".format(
                token
            )
        )

    sign = -1.0 if text.startswith("-") else 1.0
    axis = text[-1]
    index = {"x": 0, "y": 1, "z": 2}[axis]

    return index, sign


def rotation_matrix_from_axis_map(
    model_x_from_source: str,
    model_y_from_source: str,
    model_z_from_source: str,
) -> List[List[float]]:
    """
    Build Q such that v_model = Q * v_source.

    Each row tells how one model basis coordinate is obtained from the
    source/unit-cell coordinates.
    """
    tokens = [
        model_x_from_source,
        model_y_from_source,
        model_z_from_source,
    ]

    q = [[0.0, 0.0, 0.0] for _ in range(3)]
    used_source_axes = set()

    for model_axis_index, token in enumerate(tokens):
        source_axis_index, sign = _parse_signed_axis(token)

        if source_axis_index in used_source_axes:
            raise ValueError(
                "Each source axis must be used exactly once in axis-map mode; "
                "duplicate source axis in {!r}".format(tokens)
            )

        used_source_axes.add(source_axis_index)
        q[model_axis_index][source_axis_index] = sign

    validate_rotation_matrix(q, ROTATION_TOL)
    return q


def determinant_3x3(q: Sequence[Sequence[float]]) -> float:
    return (
        q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
        - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
        + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0])
    )


def validate_rotation_matrix(
    q: Sequence[Sequence[float]],
    tolerance: float = 1.0e-10,
) -> None:
    if len(q) != 3 or any(len(row) != 3 for row in q):
        raise ValueError("Rotation matrix must be 3x3.")

    # Check Q * Q^T = I.
    for i in range(3):
        for j in range(3):
            dot = 0.0
            for k in range(3):
                dot += float(q[i][k]) * float(q[j][k])

            target = 1.0 if i == j else 0.0
            if abs(dot - target) > tolerance:
                raise ValueError(
                    "Rotation matrix is not orthonormal: "
                    "(Q Q^T)[{},{}] = {:.16g}, expected {:.16g}".format(
                        i, j, dot, target
                    )
                )

    det_q = determinant_3x3(q)
    if abs(det_q - 1.0) > tolerance:
        raise ValueError(
            "Rotation matrix must be a proper right-handed rotation with "
            "det(Q)=+1. Got det(Q)={:.16g}. "
            "If you only swapped two axes, change the sign of one mapped axis."
            .format(det_q)
        )


def configured_rotation_matrix() -> List[List[float]]:
    mode = str(ORIENTATION_MODE).strip().lower()

    if mode == "none":
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    if mode == "axis_map":
        return rotation_matrix_from_axis_map(
            MODEL_X_FROM_UC,
            MODEL_Y_FROM_UC,
            MODEL_Z_FROM_UC,
        )

    if mode == "matrix":
        q = copy_matrix(ROTATION_MATRIX_SOURCE_TO_MODEL)
        validate_rotation_matrix(q, ROTATION_TOL)
        return q

    raise ValueError(
        "ORIENTATION_MODE must be 'none', 'axis_map', or 'matrix'; got {!r}".format(
            ORIENTATION_MODE
        )
    )


# =============================================================================
# VOIGT <-> FOURTH-ORDER TENSOR CONVERSION
# =============================================================================

def validate_voigt_order(voigt_order: Sequence[str]) -> None:
    normalized = [str(label).strip().lower() for label in voigt_order]
    expected = set(VOIGT_PAIR_BY_LABEL)

    if len(normalized) != 6 or set(normalized) != expected:
        raise ValueError(
            "Voigt order must contain exactly {}. Got {}".format(
                sorted(expected),
                list(voigt_order),
            )
        )


def _pair_permutations(i: int, j: int) -> List[Tuple[int, int]]:
    if i == j:
        return [(i, j)]
    return [(i, j), (j, i)]


def stiffness_6x6_to_4th_order(
    c6: Sequence[Sequence[float]],
    voigt_order: Sequence[str],
) -> List[List[List[List[float]]]]:
    """
    Convert engineering-shear 6x6 stiffness to C_ijkl.

    No factor-of-two is inserted in the stiffness components themselves.
    For example, C_xyxy = G. The two minor-symmetric tensor terms then act on
    eps_xy = gamma_xy/2 and recover tau_xy = G * gamma_xy.
    """
    validate_voigt_order(voigt_order)

    if len(c6) != 6 or any(len(row) != 6 for row in c6):
        raise ValueError("Stiffness matrix must be 6x6.")

    labels = [str(label).strip().lower() for label in voigt_order]

    c4 = [
        [
            [
                [0.0 for _ in range(3)]
                for _ in range(3)
            ]
            for _ in range(3)
        ]
        for _ in range(3)
    ]

    for a, label_a in enumerate(labels):
        i, j = VOIGT_PAIR_BY_LABEL[label_a]

        for b, label_b in enumerate(labels):
            k, l = VOIGT_PAIR_BY_LABEL[label_b]
            value = float(c6[a][b])

            for ii, jj in _pair_permutations(i, j):
                for kk, ll in _pair_permutations(k, l):
                    c4[ii][jj][kk][ll] = value

    return c4


def stiffness_4th_order_to_6x6(
    c4: Sequence[Sequence[Sequence[Sequence[float]]]],
    voigt_order: Sequence[str],
) -> List[List[float]]:
    validate_voigt_order(voigt_order)

    labels = [str(label).strip().lower() for label in voigt_order]
    c6 = [[0.0 for _ in range(6)] for _ in range(6)]

    for a, label_a in enumerate(labels):
        i, j = VOIGT_PAIR_BY_LABEL[label_a]

        for b, label_b in enumerate(labels):
            k, l = VOIGT_PAIR_BY_LABEL[label_b]
            c6[a][b] = float(c4[i][j][k][l])

    return c6


def rotate_stiffness_6x6(
    c_source: Sequence[Sequence[float]],
    q_source_to_model: Sequence[Sequence[float]],
    voigt_order: Sequence[str],
) -> List[List[float]]:
    """
    Rotate a homogenized stiffness tensor from source to model coordinates.

    Convention:
        v_model = Q * v_source

    Fourth-order transformation:
        C_model_ijkl =
            Q_ip Q_jq Q_kr Q_ls C_source_pqrs
    """
    validate_rotation_matrix(q_source_to_model, ROTATION_TOL)

    c_source_4 = stiffness_6x6_to_4th_order(c_source, voigt_order)

    c_model_4 = [
        [
            [
                [0.0 for _ in range(3)]
                for _ in range(3)
            ]
            for _ in range(3)
        ]
        for _ in range(3)
    ]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    total = 0.0

                    for p in range(3):
                        q_ip = float(q_source_to_model[i][p])
                        if q_ip == 0.0:
                            continue

                        for q_idx in range(3):
                            q_jq = float(q_source_to_model[j][q_idx])
                            if q_jq == 0.0:
                                continue

                            for r in range(3):
                                q_kr = float(q_source_to_model[k][r])
                                if q_kr == 0.0:
                                    continue

                                for s in range(3):
                                    q_ls = float(q_source_to_model[l][s])
                                    if q_ls == 0.0:
                                        continue

                                    total += (
                                        q_ip
                                        * q_jq
                                        * q_kr
                                        * q_ls
                                        * float(c_source_4[p][q_idx][r][s])
                                    )

                    c_model_4[i][j][k][l] = total

    return stiffness_4th_order_to_6x6(c_model_4, voigt_order)


# =============================================================================
# ENGINEERING CONSTANTS FROM A GENERAL 6x6 STIFFNESS
# =============================================================================

def invert_square_matrix(
    matrix: Sequence[Sequence[float]],
    pivot_tolerance: float = 1.0e-18,
) -> List[List[float]]:
    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be non-empty and square.")

    augmented = []

    for i in range(n):
        row = [float(value) for value in matrix[i]]
        row.extend(1.0 if i == j else 0.0 for j in range(n))
        augmented.append(row)

    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row_index: abs(augmented[row_index][column]),
        )

        pivot_value = augmented[pivot_row][column]

        if abs(pivot_value) <= pivot_tolerance:
            raise ValueError(
                "Matrix is singular or numerically singular at column {}.".format(
                    column
                )
            )

        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )

        pivot_value = augmented[column][column]
        for j in range(2 * n):
            augmented[column][j] /= pivot_value

        for row_index in range(n):
            if row_index == column:
                continue

            factor = augmented[row_index][column]
            if factor == 0.0:
                continue

            for j in range(2 * n):
                augmented[row_index][j] -= factor * augmented[column][j]

    return [row[n:] for row in augmented]


def engineering_constants_from_stiffness(
    c6: Sequence[Sequence[float]],
    voigt_order: Sequence[str],
) -> Dict[str, float]:
    """
    Compute directional engineering constants from the compliance matrix.

    This does not assume orthotropy. The reported constants are the standard
    uniaxial/shear compliances in the current coordinate frame.
    """
    validate_voigt_order(voigt_order)

    labels = [str(label).strip().lower() for label in voigt_order]
    index = {label: labels.index(label) for label in labels}

    s6 = invert_square_matrix(c6)

    i_xx = index["xx"]
    i_yy = index["yy"]
    i_zz = index["zz"]
    i_xy = index["xy"]
    i_xz = index["xz"]
    i_yz = index["yz"]

    ex = 1.0 / s6[i_xx][i_xx]
    ey = 1.0 / s6[i_yy][i_yy]
    ez = 1.0 / s6[i_zz][i_zz]

    gxy = 1.0 / s6[i_xy][i_xy]
    gxz = 1.0 / s6[i_xz][i_xz]
    gyz = 1.0 / s6[i_yz][i_yz]

    # nu_ab = - epsilon_b / epsilon_a under uniaxial sigma_a.
    nuxy = -s6[i_yy][i_xx] / s6[i_xx][i_xx]
    nuxz = -s6[i_zz][i_xx] / s6[i_xx][i_xx]

    nuyx = -s6[i_xx][i_yy] / s6[i_yy][i_yy]
    nuyz = -s6[i_zz][i_yy] / s6[i_yy][i_yy]

    nuzx = -s6[i_xx][i_zz] / s6[i_zz][i_zz]
    nuzy = -s6[i_yy][i_zz] / s6[i_zz][i_zz]

    return {
        "Ex": ex,
        "Ey": ey,
        "Ez": ez,
        "Gxy": gxy,
        "Gxz": gxz,
        "Gyz": gyz,
        "nuxy": nuxy,
        "nuxz": nuxz,
        "nuyx": nuyx,
        "nuyz": nuyz,
        "nuzx": nuzx,
        "nuzy": nuzy,
    }


# =============================================================================
# DATABASE QUERY API
# =============================================================================

def query_stiffness_database(
    database_path: Path,
    campaign_type: Optional[str],
    topology: Any,
    relative_density: Any,
    load_case: Any,
    state: Any,
) -> Tuple[List[sqlite3.Row], sqlite3.Row]:
    """
    Query the stiffness database.

    This function is intentionally reusable so that a future Abaqus builder
    can import this module and call the same query logic directly.
    """
    database_path = Path(database_path)

    if not database_path.is_file():
        raise FileNotFoundError(
            "Database does not exist:\n{}\n\n"
            "Run build_tpms_stiffness_database.py first.".format(database_path)
        )

    topology_canonical = canonical_topology(topology)
    load_case_canonical = canonical_load_case(load_case)
    rd_fraction = relative_density_fraction(relative_density)

    state_sql, state_parameters = state_filter(state)

    where = [
        "topology = ?",
        "load_case = ?",
        "ABS(rd_target_fraction - ?) < 1.0e-9",
        state_sql,
    ]

    parameters = [
        topology_canonical,
        load_case_canonical,
        rd_fraction,
    ]
    parameters.extend(state_parameters)

    if campaign_type is not None:
        where.append("campaign_type = ?")
        parameters.append(str(campaign_type).strip())

    query = """
        SELECT *
        FROM stiffness_catalog
        WHERE {}
        ORDER BY campaign_type, campaign_name, iteration, eval_id
    """.format(" AND ".join(where))

    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(query, parameters).fetchall()

        convention = connection.execute(
            "SELECT * FROM conventions WHERE convention_id = 1"
        ).fetchone()

    finally:
        connection.close()

    if convention is None:
        raise RuntimeError(
            "No convention row with convention_id = 1 was found in the database."
        )

    return rows, convention


def get_single_stiffness_record(
    database_path: Path,
    campaign_type: Optional[str],
    topology: Any,
    relative_density: Any,
    load_case: Any,
    state: Any,
) -> Tuple[sqlite3.Row, sqlite3.Row]:
    """
    Convenience API for scripts that require exactly one material state.
    """
    rows, convention = query_stiffness_database(
        database_path=database_path,
        campaign_type=campaign_type,
        topology=topology,
        relative_density=relative_density,
        load_case=load_case,
        state=state,
    )

    if not rows:
        raise LookupError(
            "No matching stiffness record found for topology={!r}, RD={!r}, "
            "load_case={!r}, state={!r}, campaign_type={!r}.".format(
                topology,
                relative_density,
                load_case,
                state,
                campaign_type,
            )
        )

    if len(rows) != 1:
        raise LookupError(
            "Expected exactly one stiffness record but found {}. "
            "Narrow the campaign/query before using this record in a model build."
            .format(len(rows))
        )

    return rows[0], convention


def oriented_stiffness_from_record(
    row: sqlite3.Row,
    convention: sqlite3.Row,
    q_source_to_model: Sequence[Sequence[float]],
    symmetrize_before_rotation: bool = False,
) -> Dict[str, Any]:
    """
    Convert a database row into source- and model-frame constitutive data.
    """
    canonical_order = json.loads(convention["canonical_voigt_order"])
    source_order = json.loads(convention["source_voigt_order"])

    c_source_raw = json.loads(row["matrix_json"])
    c_source_raw = copy_matrix(c_source_raw)

    if symmetrize_before_rotation:
        c_source_used = symmetrize_matrix(c_source_raw)
    else:
        c_source_used = copy_matrix(c_source_raw)

    c_model = rotate_stiffness_6x6(
        c_source_used,
        q_source_to_model,
        canonical_order,
    )

    source_constants_recomputed = engineering_constants_from_stiffness(
        c_source_used,
        canonical_order,
    )
    model_constants = engineering_constants_from_stiffness(
        c_model,
        canonical_order,
    )

    return {
        "canonical_voigt_order": canonical_order,
        "source_voigt_order": source_order,
        "C_source_raw": c_source_raw,
        "C_source_used": c_source_used,
        "C_model": c_model,
        "engineering_constants_source_recomputed": source_constants_recomputed,
        "engineering_constants_model": model_constants,
    }


# =============================================================================
# PRINTING
# =============================================================================

def _print_engineering_constants(constants: Dict[str, float]) -> None:
    print("  Ex   = {}".format(constants["Ex"]))
    print("  Ey   = {}".format(constants["Ey"]))
    print("  Ez   = {}".format(constants["Ez"]))
    print("  Gxy  = {}".format(constants["Gxy"]))
    print("  Gxz  = {}".format(constants["Gxz"]))
    print("  Gyz  = {}".format(constants["Gyz"]))
    print("  nuxy = {}".format(constants["nuxy"]))
    print("  nuxz = {}".format(constants["nuxz"]))
    print("  nuyx = {}".format(constants["nuyx"]))
    print("  nuyz = {}".format(constants["nuyz"]))
    print("  nuzx = {}".format(constants["nuzx"]))
    print("  nuzy = {}".format(constants["nuzy"]))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    topology = canonical_topology(TOPOLOGY)
    load_case = canonical_load_case(LOAD_CASE)
    rd_fraction = relative_density_fraction(RELATIVE_DENSITY)

    q_source_to_model = configured_rotation_matrix()

    try:
        rows, convention = query_stiffness_database(
            database_path=DATABASE_PATH,
            campaign_type=CAMPAIGN_TYPE,
            topology=topology,
            relative_density=rd_fraction,
            load_case=load_case,
            state=STATE,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    if not rows:
        campaign_text = CAMPAIGN_TYPE if CAMPAIGN_TYPE is not None else "ANY"
        raise SystemExit(
            "No matching matrix found.\n"
            "  campaign : {}\n"
            "  topology : {}\n"
            "  RD       : {:.6f}\n"
            "  load     : {}\n"
            "  state    : {}".format(
                campaign_text,
                topology,
                rd_fraction,
                load_case,
                STATE,
            )
        )

    if REQUIRE_UNIQUE_MATCH and len(rows) != 1:
        raise SystemExit(
            "Expected exactly one database match, but found {}.\n"
            "Narrow CAMPAIGN_TYPE / STATE / query settings before continuing."
            .format(len(rows))
        )

    canonical_order = json.loads(convention["canonical_voigt_order"])
    source_order = json.loads(convention["source_voigt_order"])

    print("=" * 80)
    print("TPMS HOMOGENIZED STIFFNESS QUERY")
    print("=" * 80)
    print("Matches: {}".format(len(rows)))
    print()

    print("Coordinate/orientation transformation")
    print("-------------------------------------")
    print("Mode:                      {}".format(ORIENTATION_MODE))

    if str(ORIENTATION_MODE).strip().lower() == "axis_map":
        print("model x from unit cell:    {}".format(MODEL_X_FROM_UC))
        print("model y from unit cell:    {}".format(MODEL_Y_FROM_UC))
        print("model z from unit cell:    {}".format(MODEL_Z_FROM_UC))

    print("Q_source_to_model =")
    print(format_matrix(q_source_to_model, MATRIX_DECIMALS))
    print("det(Q) = {:.12g}".format(determinant_3x3(q_source_to_model)))
    print()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for match_index, row in enumerate(rows, start=1):
        oriented = oriented_stiffness_from_record(
            row=row,
            convention=convention,
            q_source_to_model=q_source_to_model,
            symmetrize_before_rotation=SYMMETRIZE_BEFORE_ROTATION,
        )

        c_source_raw = oriented["C_source_raw"]
        c_source_used = oriented["C_source_used"]
        c_model = oriented["C_model"]

        labels = []
        if row["is_initial"]:
            labels.append("initial")
        if row["is_best"]:
            labels.append("best")
        if row["is_final"]:
            labels.append("final")
        if not labels:
            labels.append("iteration")

        print("-" * 80)
        print("MATCH {} / {}".format(match_index, len(rows)))
        print("-" * 80)
        print("Campaign:                 {} ({})".format(
            row["campaign_name"], row["campaign_type"]
        ))
        print("Topology:                 {}".format(row["topology"]))
        print("Relative density target:  {:.6f} ({:.3f} %)".format(
            row["rd_target_fraction"], row["rd_target_percent"]
        ))
        if row["rd_actual_fraction"] is not None:
            print("Relative density actual:  {:.6f} ({:.3f} %)".format(
                row["rd_actual_fraction"], row["rd_actual_percent"]
            ))
        print("Optimization load case:   {}".format(row["load_case"]))
        print("Target metric:             {}".format(row["target_metric_name"]))
        print("Iteration:                 {}".format(row["iteration"]))
        print("State role:                {}".format(" | ".join(labels)))
        print("Units:                     {}".format(convention["units"]))
        print("Database orientation:      {}".format(
            convention["coordinate_orientation"]
        ))
        print("Shear convention:          {}".format(
            convention["shear_convention"]
        ))
        print("Canonical Voigt ordering:  {}".format(canonical_order))
        print("Source Voigt ordering:     {}".format(source_order))
        print()

        print("C_hom SOURCE / database frame =")
        print(format_matrix(c_source_raw, MATRIX_DECIMALS))
        print()

        if SYMMETRIZE_BEFORE_ROTATION:
            symmetry_adjustment = max_abs_matrix_difference(
                c_source_raw,
                c_source_used,
            )
            print("C_hom SOURCE used after symmetrization =")
            print(format_matrix(c_source_used, MATRIX_DECIMALS))
            print()
            print(
                "Max abs source-matrix adjustment due to symmetrization: {}"
                .format(symmetry_adjustment)
            )
            print()

        print("C_hom MODEL / rotated frame =")
        print(format_matrix(c_model, MATRIX_DECIMALS))
        print()

        print("Database engineering constants in SOURCE frame [MPa]:")
        print("  Ex  = {}".format(row["Ex"]))
        print("  Ey  = {}".format(row["Ey"]))
        print("  Ez  = {}".format(row["Ez"]))
        print("  Gxy = {}".format(row["Gxy"]))
        print("  Gxz = {}".format(row["Gxz"]))
        print("  Gyz = {}".format(row["Gyz"]))
        print()

        print("Recomputed engineering constants in MODEL frame [MPa]:")
        _print_engineering_constants(
            oriented["engineering_constants_model"]
        )
        print()

        print("QA from database:")
        print("  positive definite = {}".format(
            row["positive_definite_C_sym"]
        ))
        print("  min eigenvalue    = {}".format(
            row["min_eigenvalue_C_sym"]
        ))
        print("  condition number  = {}".format(
            row["condition_number_C"]
        ))
        print("  symmetry error    = {}".format(
            row["matrix_symmetry_error"]
        ))
        print()

        print("Source:")
        print("  eval_id               = {}".format(row["eval_id"]))
        print("  input_mesh            = {}".format(row["input_mesh"]))
        print("  applied_thickness_csv = {}".format(
            row["applied_thickness_csv"]
        ))
        print("  state_out_dir         = {}".format(row["state_out_dir"]))

        if EXPORT_JSON:
            output = {
                "campaign_name": row["campaign_name"],
                "campaign_type": row["campaign_type"],
                "topology": row["topology"],
                "relative_density_target": row["rd_target_fraction"],
                "relative_density_actual": row["rd_actual_fraction"],
                "optimization_load_case": row["load_case"],
                "target_metric": row["target_metric_name"],
                "iteration": row["iteration"],
                "state_role": labels,
                "units": convention["units"],
                "database_orientation": convention["coordinate_orientation"],
                "shear_convention": convention["shear_convention"],
                "voigt_order": canonical_order,

                # Backward-compatible database/source tensor.
                "C_hom": c_source_raw,

                # Explicitly named tensors for the HOST/Abaqus workflow.
                "C_hom_source_raw": c_source_raw,
                "C_hom_source_used": c_source_used,
                "C_hom_model": c_model,

                "orientation_transform": {
                    "mode": ORIENTATION_MODE,
                    "convention": "v_model = Q_source_to_model * v_source",
                    "Q_source_to_model": q_source_to_model,
                    "det_Q": determinant_3x3(q_source_to_model),
                    "model_x_from_unit_cell": (
                        MODEL_X_FROM_UC
                        if str(ORIENTATION_MODE).strip().lower() == "axis_map"
                        else None
                    ),
                    "model_y_from_unit_cell": (
                        MODEL_Y_FROM_UC
                        if str(ORIENTATION_MODE).strip().lower() == "axis_map"
                        else None
                    ),
                    "model_z_from_unit_cell": (
                        MODEL_Z_FROM_UC
                        if str(ORIENTATION_MODE).strip().lower() == "axis_map"
                        else None
                    ),
                    "symmetrize_before_rotation": SYMMETRIZE_BEFORE_ROTATION,
                },

                "engineering_constants_source_database": {
                    "Ex": row["Ex"],
                    "Ey": row["Ey"],
                    "Ez": row["Ez"],
                    "Gxy": row["Gxy"],
                    "Gxz": row["Gxz"],
                    "Gyz": row["Gyz"],
                    "nuxy": row["nuxy"],
                    "nuxz": row["nuxz"],
                    "nuyx": row["nuyx"],
                    "nuyz": row["nuyz"],
                    "nuzx": row["nuzx"],
                    "nuzy": row["nuzy"],
                },

                "engineering_constants_source_recomputed": oriented[
                    "engineering_constants_source_recomputed"
                ],

                "engineering_constants_model": oriented[
                    "engineering_constants_model"
                ],

                "qa": {
                    "positive_definite": row["positive_definite_C_sym"],
                    "min_eigenvalue": row["min_eigenvalue_C_sym"],
                    "condition_number": row["condition_number_C"],
                    "symmetry_error": row["matrix_symmetry_error"],
                },

                "source": {
                    "eval_id": row["eval_id"],
                    "input_mesh": row["input_mesh"],
                    "applied_thickness_csv": row["applied_thickness_csv"],
                    "state_out_dir": row["state_out_dir"],
                    "source_voigt_order": source_order,
                },
            }

            rd_percent = 100.0 * rd_fraction

            orientation_suffix = (
                "identity"
                if str(ORIENTATION_MODE).strip().lower() == "none"
                else "rotated"
            )

            filename = (
                "{}_{}_rd{:g}_{}_itag{:02d}_{}.json".format(
                    safe_name(row["campaign_type"]),
                    safe_name(row["topology"]),
                    rd_percent,
                    safe_name(row["load_case"]),
                    int(row["iteration"]),
                    orientation_suffix,
                )
            )

            output_path = OUTPUT_DIRECTORY / filename
            output_path.write_text(
                json.dumps(output, indent=2),
                encoding="utf-8",
            )

            print()
            print("[EXPORTED] {}".format(output_path))

        print()


if __name__ == "__main__":
    main()
