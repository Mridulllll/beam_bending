"""
Point-load boundary-value solver for the two-mode HOST beam.

The solver mirrors the existing one-mode HOST multiple-shooting strategy but
uses the eight-state two-mode system

    [w, theta, beta1, beta2, M, V, H1, H2].

Concentrated transverse forces produce jumps only in V.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import expm

from host.beam_point_loads import (
    PointForce,
    SupportReactions,
    simply_supported_reactions,
)

from host.beam_two_mode import (
    two_mode_state_matrix,
)


@dataclass(frozen=True)
class TwoModePointLoadSolution:
    K5: np.ndarray
    length: float
    support_left: float
    support_right: float
    applied_forces: tuple
    reactions: SupportReactions
    knot_positions: np.ndarray
    knot_states_right: np.ndarray
    combined_forces: tuple

    def all_point_forces(self) -> tuple:
        return self.combined_forces

    def state(
        self,
        x: float,
        *,
        side: str = "right",
    ) -> np.ndarray:
        return _evaluate_state(
            self.K5,
            self.knot_positions,
            self.knot_states_right,
            self.combined_forces,
            x,
            length=self.length,
            side=side,
        )


def _combine_point_forces(
    point_forces: Iterable[PointForce],
    *,
    tolerance: float = 1.0e-12,
) -> tuple:
    loads = sorted(
        (
            PointForce(
                float(load.x),
                float(load.force),
            )
            for load in point_forces
        ),
        key=lambda load: load.x,
    )

    combined = []

    for load in loads:
        if (
            combined
            and abs(load.x - combined[-1].x)
            <= tolerance * max(1.0, abs(load.x))
        ):
            previous = combined.pop()

            combined.append(
                PointForce(
                    previous.x,
                    previous.force + load.force,
                )
            )
        else:
            combined.append(load)

    return tuple(combined)


def _force_at(
    point_forces,
    x: float,
    *,
    length: float,
) -> float:
    tolerance = 1.0e-12 * max(
        1.0,
        abs(float(x)),
        abs(float(length)),
    )

    return float(
        sum(
            load.force
            for load in point_forces
            if abs(float(load.x) - float(x)) <= tolerance
        )
    )


def _build_knots(
    *,
    length: float,
    point_forces,
    max_segment_length: float,
) -> np.ndarray:
    base = sorted(
        {
            0.0,
            float(length),
            *(
                float(load.x)
                for load in point_forces
            ),
        }
    )

    knots = [base[0]]

    for x0, x1 in zip(base[:-1], base[1:]):
        dx = float(x1 - x0)

        count = max(
            1,
            int(
                np.ceil(
                    dx / float(max_segment_length)
                )
            ),
        )

        for j in range(1, count + 1):
            knots.append(
                x0 + dx * float(j) / float(count)
            )

    return np.asarray(knots, dtype=float)


def _knot_index(
    knots: np.ndarray,
    x: float,
) -> int:
    indices = np.where(
        np.abs(knots - float(x))
        <= 1.0e-10 * max(1.0, abs(float(x)))
    )[0]

    if len(indices) != 1:
        raise RuntimeError(
            "Required beam coordinate is not represented by one knot."
        )

    return int(indices[0])


def _equilibrated_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)

    row_norms = np.linalg.norm(A, axis=1)

    if np.any(row_norms == 0.0):
        raise RuntimeError("Multiple-shooting system has a zero row.")

    row_scale = 1.0 / row_norms
    Ar = row_scale[:, None] * A
    br = row_scale * b

    column_norms = np.linalg.norm(Ar, axis=0)

    if np.any(column_norms == 0.0):
        raise RuntimeError(
            "Multiple-shooting system has an unconstrained variable."
        )

    column_scale = 1.0 / column_norms
    As = Ar * column_scale[None, :]

    solution_scaled = np.linalg.solve(As, br)

    return column_scale * solution_scaled


def _solve_knot_states(
    K5: np.ndarray,
    *,
    length: float,
    support_left: float,
    support_right: float,
    point_forces,
    max_segment_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    K5 = np.asarray(K5, dtype=float)
    length = float(length)

    knots = _build_knots(
        length=length,
        point_forces=point_forces,
        max_segment_length=max_segment_length,
    )

    n = len(knots)
    size = 8 * n

    system = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    A = two_mode_state_matrix(K5)

    row = 0

    for j in range(n - 1):
        x0 = float(knots[j])
        x1 = float(knots[j + 1])
        dx = x1 - x0

        transfer = expm(A * dx)

        left = slice(8 * j, 8 * j + 8)
        right = slice(8 * (j + 1), 8 * (j + 1) + 8)

        system[row:row + 8, left] = -transfer
        system[row:row + 8, right] = np.eye(8)

        jump = np.zeros(8, dtype=float)
        jump[5] = -_force_at(
            point_forces,
            x1,
            length=length,
        )

        rhs[row:row + 8] = jump
        row += 8

    force_at_zero = _force_at(
        point_forces,
        0.0,
        length=length,
    )

    # Left physical end: M(0-)=V(0-)=H1(0-)=H2(0-)=0.
    # Unknown state at x=0 is right of a possible boundary force.
    for component, value in (
        (4, 0.0),
        (5, -force_at_zero),
        (6, 0.0),
        (7, 0.0),
    ):
        system[row, component] = 1.0
        rhs[row] = value
        row += 1

    iA = _knot_index(knots, support_left)
    iB = _knot_index(knots, support_right)

    system[row, 8 * iA] = 1.0
    row += 1

    system[row, 8 * iB] = 1.0
    row += 1

    # Right higher-order natural conditions.
    system[row, 8 * (n - 1) + 6] = 1.0
    row += 1

    system[row, 8 * (n - 1) + 7] = 1.0
    row += 1

    if row != size:
        raise RuntimeError(
            "Two-mode multiple-shooting equation count mismatch."
        )

    states = _equilibrated_solve(
        system,
        rhs,
    ).reshape((n, 8))

    return knots, states


def _evaluate_state(
    K5: np.ndarray,
    knots: np.ndarray,
    states_right: np.ndarray,
    point_forces,
    x: float,
    *,
    length: float,
    side: str,
) -> np.ndarray:
    x = float(x)
    side_text = str(side).strip().lower()

    if side_text not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    if not 0.0 <= x <= float(length):
        raise ValueError("x lies outside the beam.")

    matches = np.where(
        np.abs(knots - x)
        <= 1.0e-10 * max(1.0, abs(x), abs(float(length)))
    )[0]

    if len(matches) == 1:
        state = states_right[int(matches[0])].copy()

        if side_text == "left":
            force = _force_at(
                point_forces,
                x,
                length=length,
            )
            state[5] += force

        return state

    index = int(
        np.searchsorted(
            knots,
            x,
            side="right",
        )
        - 1
    )

    if index < 0 or index >= len(knots) - 1:
        raise RuntimeError("Could not identify local beam segment.")

    A = two_mode_state_matrix(K5)
    dx = x - float(knots[index])

    return expm(A * dx) @ states_right[index]


def solve_simply_supported_two_mode_point_loads(
    K5: np.ndarray,
    *,
    length: float,
    support_left: float,
    support_right: float,
    applied_forces,
    max_segment_length: float = 10.0,
) -> TwoModePointLoadSolution:
    """
    Solve a uniform two-mode HOST beam with two simple supports.
    """

    K5 = np.asarray(K5, dtype=float)

    if K5.shape != (5, 5):
        raise ValueError("K5 must have shape (5, 5).")

    length = float(length)
    xA = float(support_left)
    xB = float(support_right)

    if length <= 0.0:
        raise ValueError("length must be positive.")

    if not 0.0 <= xA < xB <= length:
        raise ValueError(
            "Supports must satisfy 0 <= left < right <= length."
        )

    if float(max_segment_length) <= 0.0:
        raise ValueError("max_segment_length must be positive.")

    loads = tuple(
        PointForce(
            float(load.x),
            float(load.force),
        )
        for load in applied_forces
    )

    if not loads:
        raise ValueError("At least one applied point force is required.")

    for load in loads:
        if not 0.0 <= load.x <= length:
            raise ValueError("Point force lies outside the beam.")

    reactions = simply_supported_reactions(
        loads,
        support_left=xA,
        support_right=xB,
    )

    combined = _combine_point_forces(
        loads
        + (
            PointForce(xA, reactions.left),
            PointForce(xB, reactions.right),
        )
    )

    knots, states = _solve_knot_states(
        K5,
        length=length,
        support_left=xA,
        support_right=xB,
        point_forces=combined,
        max_segment_length=max_segment_length,
    )

    solution = TwoModePointLoadSolution(
        K5=K5.copy(),
        length=length,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
        reactions=reactions,
        knot_positions=knots.copy(),
        knot_states_right=states.copy(),
        combined_forces=combined,
    )

    # Cheap consistency checks.
    right = solution.state(length, side="right")

    scale_force = max(
        1.0,
        sum(abs(load.force) for load in loads),
    )

    if abs(right[4]) > 1.0e-7 * scale_force * max(1.0, length):
        raise RuntimeError("Right-end bending moment equilibrium failed.")

    if abs(right[5]) > 1.0e-8 * scale_force:
        raise RuntimeError("Right-end shear equilibrium failed.")

    return solution
