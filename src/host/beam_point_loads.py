"""
Exact piecewise solution of a uniform HOST beam under concentrated
transverse forces.

The physical specimen may extend beyond the supports.

Force sign convention
---------------------
A concentrated force P follows the same sign convention as the
distributed load q in beam_bending.py.

Since

    V' = -q,

a point force produces

    V(x_i+) - V(x_i-) = -P_i.

The support reactions are obtained from global static equilibrium.

Natural conditions at free physical specimen ends:

    M = 0,
    V = 0,
    H = 0.

Simple supports impose

    w(x_A) = 0,
    w(x_B) = 0.

The remaining higher-order natural condition is

    H(L) = 0.

No longitudinal mesh is introduced. Between point-force locations the
uniform state is propagated exactly with the matrix-exponential HOST
beam propagator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from host.beam_bending import (
    beam_state_space,
    propagate_uniform_segment,
)


@dataclass(frozen=True)
class PointForce:
    """
    Concentrated transverse force.

    Parameters
    ----------
    x
        Position measured from the left physical specimen end.

    force
        Concentrated force following the q sign convention.
    """

    x: float
    force: float


@dataclass(frozen=True)
class SupportReactions:
    left: float
    right: float


@dataclass(frozen=True)
class HOSTPointLoadSolution:
    """
    Exact uniform-section point-load HOST beam solution.

    The boundary-value problem is solved with multiple shooting.

    State values at the load/support knots are retained so arbitrary
    x evaluations require propagation over only one local segment.
    This avoids catastrophic amplification of the exponentially
    growing higher-order state mode in deep sandwich beams.
    """

    K3: np.ndarray
    length: float

    support_left: float
    support_right: float

    applied_forces: tuple
    reactions: SupportReactions

    # Right-side state at x = 0, retained for backward compatibility.
    initial_state: np.ndarray

    # Multiple-shooting representation.
    knot_positions: np.ndarray
    knot_states_right: np.ndarray
    combined_forces: tuple

    def all_point_forces(self) -> tuple:
        """
        Applied loads plus both support reactions.
        """

        loads = list(
            self.applied_forces
        )

        loads.append(
            PointForce(
                self.support_left,
                self.reactions.left,
            )
        )

        loads.append(
            PointForce(
                self.support_right,
                self.reactions.right,
            )
        )

        return tuple(
            sorted(
                loads,
                key=lambda item: item.x,
            )
        )

    def state(
        self,
        x: float,
        *,
        side: str = "right",
    ) -> np.ndarray:
        """
        Evaluate

            [w, theta, beta, M, V, H]

        using the nearest multiple-shooting knot.

        At a concentrated force:

            side="left"
                state immediately before the force jump;

            side="right"
                state immediately after the force jump.
        """

        return _evaluate_multiple_shooting_state(
            self.K3,
            self.knot_positions,
            self.knot_states_right,
            self.combined_forces,
            x,
            length=self.length,
            side=side,
        )


def _validate_point_force(
    load: PointForce,
    *,
    length: float,
) -> PointForce:
    x = float(
        load.x
    )

    force = float(
        load.force
    )

    if x < 0.0 or x > length:
        raise ValueError(
            "Point-force position must lie within the specimen."
        )

    if not np.isfinite(
        force
    ):
        raise ValueError(
            "Point force must be finite."
        )

    return PointForce(
        x=x,
        force=force,
    )


def combine_point_forces(
    point_forces,
    *,
    tolerance: float = 1.0e-12,
) -> tuple:
    """
    Combine forces applied at the same position.

    Positions are assumed to use a common consistent length unit.
    """

    forces = sorted(
        (
            PointForce(
                float(load.x),
                float(load.force),
            )
            for load in point_forces
        ),
        key=lambda item: item.x,
    )

    if not forces:
        return tuple()

    combined = []

    current_x = forces[0].x
    current_force = forces[0].force

    for load in forces[1:]:
        scale = max(
            1.0,
            abs(current_x),
            abs(load.x),
        )

        if abs(
            load.x - current_x
        ) <= tolerance * scale:
            current_force += load.force

        else:
            combined.append(
                PointForce(
                    current_x,
                    current_force,
                )
            )

            current_x = load.x
            current_force = load.force

    combined.append(
        PointForce(
            current_x,
            current_force,
        )
    )

    return tuple(
        combined
    )


def simply_supported_reactions(
    applied_forces,
    *,
    support_left: float,
    support_right: float,
) -> SupportReactions:
    """
    Compute statically determinate reactions for two simple supports.

    Equilibrium:

        R_A + R_B + sum(P_i) = 0

        R_B (x_B - x_A)
        + sum(P_i (x_i - x_A)) = 0
    """

    xA = float(
        support_left
    )

    xB = float(
        support_right
    )

    if xB <= xA:
        raise ValueError(
            "support_right must be greater than support_left."
        )

    loads = tuple(
        applied_forces
    )

    total_force = sum(
        float(load.force)
        for load in loads
    )

    moment_about_A = sum(
        float(load.force)
        * (
            float(load.x)
            - xA
        )
        for load in loads
    )

    RB = (
        -moment_about_A
        / (
            xB - xA
        )
    )

    RA = (
        -total_force
        - RB
    )

    return SupportReactions(
        left=float(RA),
        right=float(RB),
    )


def apply_point_force_jump(
    state: np.ndarray,
    force: float,
) -> np.ndarray:
    """
    Apply

        Delta V = -P.
    """

    y = np.asarray(
        state,
        dtype=float,
    ).copy()

    if y.shape != (
        6,
    ):
        raise ValueError(
            "state must have shape (6,)."
        )

    y[4] -= float(
        force
    )

    return y


def evaluate_point_load_state(
    K3: np.ndarray,
    initial_state: np.ndarray,
    point_forces,
    x: float,
    *,
    length: float,
    side: str = "right",
) -> np.ndarray:
    """
    Propagate from x=0 to an arbitrary specimen coordinate.
    """

    length = float(
        length
    )

    x = float(
        x
    )

    if length <= 0.0:
        raise ValueError(
            "Beam length must be positive."
        )

    if x < 0.0 or x > length:
        raise ValueError(
            "Evaluation coordinate lies outside the specimen."
        )

    if side not in (
        "left",
        "right",
    ):
        raise ValueError(
            "side must be 'left' or 'right'."
        )

    loads = combine_point_forces(
        point_forces
    )

    y = np.asarray(
        initial_state,
        dtype=float,
    ).copy()

    if y.shape != (
        6,
    ):
        raise ValueError(
            "initial_state must have shape (6,)."
        )

    current_x = 0.0

    for load in loads:
        load_x = float(
            load.x
        )

        if load_x > x:
            break

        if load_x < current_x:
            raise RuntimeError(
                "Point forces are not ordered."
            )

        y = propagate_uniform_segment(
            K3,
            y,
            load_x - current_x,
            q=0.0,
        )

        current_x = load_x

        at_evaluation_point = np.isclose(
            load_x,
            x,
            rtol=0.0,
            atol=1.0e-12
            * max(
                1.0,
                abs(x),
            ),
        )

        if (
            at_evaluation_point
            and side == "left"
        ):
            return y

        y = apply_point_force_jump(
            y,
            load.force,
        )

        if at_evaluation_point:
            return y

    y = propagate_uniform_segment(
        K3,
        y,
        x - current_x,
        q=0.0,
    )

    return y


def _position_tolerance(
    x: float,
    length: float,
) -> float:
    return (
        1.0e-12
        * max(
            1.0,
            abs(float(x)),
            abs(float(length)),
        )
    )


def _force_at_position(
    point_forces,
    x: float,
    *,
    length: float,
) -> float:
    """
    Sum all concentrated forces at one geometric knot.
    """

    tolerance = _position_tolerance(
        x,
        length,
    )

    return float(
        sum(
            load.force
            for load in point_forces
            if abs(
                float(load.x)
                - float(x)
            ) <= tolerance
        )
    )


def _build_knot_positions(
    length: float,
    point_forces,
    *,
    K3: np.ndarray | None = None,
    max_growth_exponent: float = 6.0,
    max_segment_length: float = 30.0,
) -> np.ndarray:
    """
    Build the multiple-shooting knot set.

    Physical knots are always included at

        x = 0,
        all concentrated-force locations,
        x = L.

    Additional auxiliary shooting knots are inserted automatically
    when a physical interval would produce excessive exponential
    amplification in

        exp(A * dx).

    The relevant growth indicator is

        alpha = max |Re(lambda_i(A))|.

    Each interval is subdivided such that approximately

        alpha * dx <= max_growth_exponent

    and

        dx <= max_segment_length.

    These auxiliary knots do NOT introduce a finite-element or
    numerical longitudinal approximation: each subinterval still uses
    the exact constant-coefficient matrix exponential. They only
    improve conditioning of the global boundary-value solve.
    """

    length = float(
        length
    )

    if length <= 0.0:
        raise ValueError(
            "Beam length must be positive."
        )

    if max_growth_exponent <= 0.0:
        raise ValueError(
            "max_growth_exponent must be positive."
        )

    if max_segment_length <= 0.0:
        raise ValueError(
            "max_segment_length must be positive."
        )

    raw = sorted(
        [
            0.0,
            length,
            *(
                float(load.x)
                for load in point_forces
            ),
        ]
    )

    # --------------------------------------------------------------
    # Unique physical knots.
    # --------------------------------------------------------------

    physical = []

    for x in raw:

        if not physical:
            physical.append(
                x
            )
            continue

        tolerance = (
            1.0e-12
            * max(
                1.0,
                abs(x),
                abs(physical[-1]),
                abs(length),
            )
        )

        if abs(
            x - physical[-1]
        ) > tolerance:
            physical.append(
                x
            )

    # --------------------------------------------------------------
    # Spectral growth rate of the HOST state system.
    # --------------------------------------------------------------

    alpha = 0.0

    if K3 is not None:

        state_matrix, _ = beam_state_space(
            np.asarray(
                K3,
                dtype=float,
            )
        )

        eigenvalues = np.linalg.eigvals(
            state_matrix
        )

        alpha = float(
            np.max(
                np.abs(
                    np.real(
                        eigenvalues
                    )
                )
            )
        )

        if not np.isfinite(
            alpha
        ):
            raise RuntimeError(
                "Non-finite HOST state growth rate."
            )

    # --------------------------------------------------------------
    # Subdivide each physical interval.
    # --------------------------------------------------------------

    refined = [
        physical[0]
    ]

    for x0, x1 in zip(
        physical[:-1],
        physical[1:],
    ):

        dx = float(
            x1 - x0
        )

        if dx <= 0.0:
            continue

        n_length = int(
            np.ceil(
                dx
                / max_segment_length
            )
        )

        if alpha > 0.0:

            n_growth = int(
                np.ceil(
                    alpha
                    * dx
                    / max_growth_exponent
                )
            )

        else:

            n_growth = 1

        n_segments = max(
            1,
            n_length,
            n_growth,
        )

        for i in range(
            1,
            n_segments + 1,
        ):

            x = (
                x0
                + dx
                * float(i)
                / float(n_segments)
            )

            refined.append(
                x
            )

    return np.asarray(
        refined,
        dtype=float,
    )



def _find_knot_index(
    knot_positions: np.ndarray,
    x: float,
    *,
    length: float,
) -> int:
    """
    Return the knot index corresponding to x.
    """

    tolerance = _position_tolerance(
        x,
        length,
    )

    indices = np.where(
        np.abs(
            knot_positions
            - float(x)
        ) <= tolerance
    )[0]

    if len(
        indices
    ) != 1:
        raise RuntimeError(
            "Required beam coordinate is not represented "
            "by exactly one multiple-shooting knot."
        )

    return int(
        indices[0]
    )


def _equilibrated_linear_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve a linear system after simple row/column equilibration.

    HOST beam states mix displacement, rotation and several resultants
    whose numerical magnitudes differ strongly. Equilibration greatly
    improves the conditioning of the multiple-shooting algebraic
    system without changing the physical equations.
    """

    A = np.asarray(
        matrix,
        dtype=float,
    )

    b = np.asarray(
        rhs,
        dtype=float,
    )

    row_norms = np.linalg.norm(
        A,
        axis=1,
    )

    if np.any(
        row_norms == 0.0
    ):
        raise RuntimeError(
            "Multiple-shooting system contains a zero equation row."
        )

    row_scale = (
        1.0
        / row_norms
    )

    A_row = (
        row_scale[:, None]
        * A
    )

    b_row = (
        row_scale
        * b
    )

    column_norms = np.linalg.norm(
        A_row,
        axis=0,
    )

    if np.any(
        column_norms == 0.0
    ):
        raise RuntimeError(
            "Multiple-shooting system contains an unconstrained "
            "state variable."
        )

    column_scale = (
        1.0
        / column_norms
    )

    A_scaled = (
        A_row
        * column_scale[None, :]
    )

    try:
        scaled_solution = np.linalg.solve(
            A_scaled,
            b_row,
        )

    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Multiple-shooting HOST beam boundary-value "
            "system is singular."
        ) from exc

    return (
        column_scale
        * scaled_solution
    )


def _assemble_multiple_shooting_system(
    K3: np.ndarray,
    *,
    length: float,
    support_left: float,
    support_right: float,
    point_forces,
):
    """
    Assemble the exact piecewise HOST beam BVP.

    Unknowns
    --------
    The six-component RIGHT-side state at every knot,

        y_j^+ =
        [w, theta, beta, M, V, H]^T.

    Between knots,

        y_{j+1}^- = exp(A * dx) y_j^+.

    At a point force P,

        y_{j+1}^+ =
        y_{j+1}^-
        + [0,0,0,0,-P,0]^T.

    Boundary conditions
    -------------------
    Left physical end, immediately before any boundary force:

        M(0-) = 0
        V(0-) = 0
        H(0-) = 0

    Simple supports:

        w(x_A) = 0
        w(x_B) = 0

    Right physical end:

        H(L+) = 0

    Global statics already determine the two support reactions, so
    M(L+) = V(L+) = 0 follow from force/moment equilibrium and are
    retained as post-solution checks rather than independent equations.
    """

    K3 = np.asarray(
        K3,
        dtype=float,
    )

    length = float(
        length
    )

    combined = combine_point_forces(
        point_forces
    )

    knots = _build_knot_positions(
        length,
        combined,
        K3=K3,
    )

    n_knots = len(
        knots
    )

    n_unknowns = (
        6
        * n_knots
    )

    system = np.zeros(
        (
            n_unknowns,
            n_unknowns,
        ),
        dtype=float,
    )

    rhs = np.zeros(
        n_unknowns,
        dtype=float,
    )

    state_matrix, _ = beam_state_space(
        K3
    )

    row = 0

    # --------------------------------------------------------------
    # Exact transfer and point-force jump equations.
    # --------------------------------------------------------------

    for j in range(
        n_knots - 1
    ):
        x0 = float(
            knots[j]
        )

        x1 = float(
            knots[j + 1]
        )

        dx = (
            x1 - x0
        )

        transfer = expm(
            state_matrix
            * dx
        )

        left_slice = slice(
            6 * j,
            6 * j + 6,
        )

        right_slice = slice(
            6 * (j + 1),
            6 * (j + 1) + 6,
        )

        system[
            row:row + 6,
            left_slice,
        ] = (
            -transfer
        )

        system[
            row:row + 6,
            right_slice,
        ] = np.eye(
            6
        )

        force_at_next = _force_at_position(
            combined,
            x1,
            length=length,
        )

        jump = np.zeros(
            6,
            dtype=float,
        )

        jump[4] = (
            -force_at_next
        )

        rhs[
            row:row + 6
        ] = jump

        row += 6

    # --------------------------------------------------------------
    # Left physical-end natural conditions.
    #
    # Unknown y_0 is the state AFTER a possible force at x=0.
    # Since
    #
    #     V(0+) = V(0-) - P0
    #
    # and V(0-) = 0,
    #
    #     V(0+) = -P0.
    # --------------------------------------------------------------

    force_at_zero = _force_at_position(
        combined,
        0.0,
        length=length,
    )

    # M(0+) = 0
    system[
        row,
        3,
    ] = 1.0

    rhs[
        row
    ] = 0.0

    row += 1

    # V(0+) = -P0
    system[
        row,
        4,
    ] = 1.0

    rhs[
        row
    ] = (
        -force_at_zero
    )

    row += 1

    # H(0+) = 0
    system[
        row,
        5,
    ] = 1.0

    rhs[
        row
    ] = 0.0

    row += 1

    # --------------------------------------------------------------
    # Simple-support displacement constraints.
    # --------------------------------------------------------------

    index_A = _find_knot_index(
        knots,
        support_left,
        length=length,
    )

    index_B = _find_knot_index(
        knots,
        support_right,
        length=length,
    )

    system[
        row,
        6 * index_A,
    ] = 1.0

    rhs[
        row
    ] = 0.0

    row += 1

    system[
        row,
        6 * index_B,
    ] = 1.0

    rhs[
        row
    ] = 0.0

    row += 1

    # --------------------------------------------------------------
    # Right higher-order natural condition.
    # --------------------------------------------------------------

    system[
        row,
        6 * (n_knots - 1) + 5,
    ] = 1.0

    rhs[
        row
    ] = 0.0

    row += 1

    if row != n_unknowns:
        raise RuntimeError(
            "Internal multiple-shooting equation-count mismatch."
        )

    return (
        knots,
        combined,
        system,
        rhs,
    )


def _solve_multiple_shooting_states(
    K3: np.ndarray,
    *,
    length: float,
    support_left: float,
    support_right: float,
    point_forces,
):
    knots, combined, system, rhs = (
        _assemble_multiple_shooting_system(
            K3,
            length=length,
            support_left=support_left,
            support_right=support_right,
            point_forces=point_forces,
        )
    )

    solution_vector = _equilibrated_linear_solve(
        system,
        rhs,
    )

    states_right = np.asarray(
        solution_vector,
        dtype=float,
    ).reshape(
        (
            len(knots),
            6,
        )
    )

    return (
        knots,
        states_right,
        combined,
    )


def _evaluate_multiple_shooting_state(
    K3: np.ndarray,
    knot_positions: np.ndarray,
    knot_states_right: np.ndarray,
    point_forces,
    x: float,
    *,
    length: float,
    side: str = "right",
) -> np.ndarray:
    """
    Evaluate a solved multiple-shooting state using only local
    propagation from the nearest left knot.
    """

    x = float(
        x
    )

    length = float(
        length
    )

    if x < 0.0 or x > length:
        raise ValueError(
            "Evaluation coordinate lies outside the specimen."
        )

    if side not in (
        "left",
        "right",
    ):
        raise ValueError(
            "side must be 'left' or 'right'."
        )

    knots = np.asarray(
        knot_positions,
        dtype=float,
    )

    states = np.asarray(
        knot_states_right,
        dtype=float,
    )

    if states.shape != (
        len(knots),
        6,
    ):
        raise ValueError(
            "Invalid multiple-shooting knot-state array."
        )

    tolerance = _position_tolerance(
        x,
        length,
    )

    matches = np.where(
        np.abs(
            knots - x
        ) <= tolerance
    )[0]

    # --------------------------------------------------------------
    # Exactly on a knot.
    # --------------------------------------------------------------

    if len(
        matches
    ) == 1:
        index = int(
            matches[0]
        )

        state = states[
            index
        ].copy()

        if side == "left":
            # Undo the concentrated-force jump:
            #
            #     V+ = V- - P
            #
            # so
            #
            #     V- = V+ + P.
            force = _force_at_position(
                point_forces,
                knots[index],
                length=length,
            )

            state[4] += force

        return state

    # --------------------------------------------------------------
    # Inside one exact uniform segment.
    # --------------------------------------------------------------

    index = int(
        np.searchsorted(
            knots,
            x,
            side="right",
        )
        - 1
    )

    if index < 0 or index >= len(
        knots
    ) - 1:
        raise RuntimeError(
            "Could not identify local HOST beam segment."
        )

    dx = (
        x
        - float(
            knots[index]
        )
    )

    return propagate_uniform_segment(
        K3,
        states[index],
        dx,
        q=0.0,
    )


def solve_simply_supported_point_loads(
    K3: np.ndarray,
    *,
    length: float,
    support_left: float,
    support_right: float,
    applied_forces,
) -> HOSTPointLoadSolution:
    """
    Solve a uniform HOST beam with two simple supports and arbitrary
    concentrated transverse forces.

    The solution uses exact matrix-exponential propagation on each
    uniform interval combined with a multiple-shooting global boundary
    solve.

    This remains numerically stable for deep/shear-compliant sandwich
    beams whose higher-order state equations contain exponentially
    growing/decaying boundary-layer modes.
    """

    K3 = np.asarray(
        K3,
        dtype=float,
    )

    if K3.shape != (
        3,
        3,
    ):
        raise ValueError(
            "K3 must have shape (3, 3)."
        )

    length = float(
        length
    )

    xA = float(
        support_left
    )

    xB = float(
        support_right
    )

    if length <= 0.0:
        raise ValueError(
            "Beam length must be positive."
        )

    if not (
        0.0
        <= xA
        < xB
        <= length
    ):
        raise ValueError(
            "Supports must satisfy "
            "0 <= support_left < support_right <= length."
        )

    loads = tuple(
        _validate_point_force(
            load,
            length=length,
        )
        for load in applied_forces
    )

    if len(
        loads
    ) == 0:
        raise ValueError(
            "At least one applied force is required."
        )

    reactions = simply_supported_reactions(
        loads,
        support_left=xA,
        support_right=xB,
    )

    all_forces = tuple(
        loads
    ) + (
        PointForce(
            xA,
            reactions.left,
        ),
        PointForce(
            xB,
            reactions.right,
        ),
    )

    (
        knot_positions,
        knot_states_right,
        combined_forces,
    ) = _solve_multiple_shooting_states(
        K3,
        length=length,
        support_left=xA,
        support_right=xB,
        point_forces=all_forces,
    )

    solution = HOSTPointLoadSolution(
        K3=K3.copy(),
        length=length,
        support_left=xA,
        support_right=xB,
        applied_forces=loads,
        reactions=reactions,
        initial_state=(
            knot_states_right[
                0
            ].copy()
        ),
        knot_positions=(
            knot_positions.copy()
        ),
        knot_states_right=(
            knot_states_right.copy()
        ),
        combined_forces=combined_forces,
    )

    # --------------------------------------------------------------
    # Physical consistency checks.
    # --------------------------------------------------------------

    state_0 = solution.state(
        0.0,
        side="left",
    )

    state_A = solution.state(
        xA,
        side="right",
    )

    state_B = solution.state(
        xB,
        side="right",
    )

    state_L = solution.state(
        length,
        side="right",
    )

    force_scale = max(
        1.0,
        abs(reactions.left),
        abs(reactions.right),
        *(
            abs(load.force)
            for load in loads
        ),
    )

    moment_scale = (
        force_scale
        * max(
            1.0,
            length,
        )
    )

    displacement_tolerance = (
        1.0e-10
        * max(
            1.0,
            length,
        )
    )

    force_tolerance = (
        1.0e-8
        * force_scale
    )

    moment_tolerance = (
        1.0e-8
        * moment_scale
    )

    if abs(
        state_0[3]
    ) > moment_tolerance:
        raise RuntimeError(
            "Left physical-end bending moment is not zero."
        )

    if abs(
        state_0[4]
    ) > force_tolerance:
        raise RuntimeError(
            "Left physical-end transverse force is not zero."
        )

    if abs(
        state_0[5]
    ) > moment_tolerance:
        raise RuntimeError(
            "Left physical-end higher-order resultant is not zero."
        )

    if abs(
        state_A[0]
    ) > displacement_tolerance:
        raise RuntimeError(
            "Left support displacement constraint was not satisfied."
        )

    if abs(
        state_B[0]
    ) > displacement_tolerance:
        raise RuntimeError(
            "Right support displacement constraint was not satisfied."
        )

    if abs(
        state_L[3]
    ) > moment_tolerance:
        raise RuntimeError(
            "Right physical-end bending moment is not zero."
        )

    if abs(
        state_L[4]
    ) > force_tolerance:
        raise RuntimeError(
            "Right physical-end transverse force is not zero."
        )

    if abs(
        state_L[5]
    ) > moment_tolerance:
        raise RuntimeError(
            "Right physical-end higher-order resultant is not zero."
        )

    return solution



def three_point_bending_loads(
    *,
    load: float,
    load_position: float,
) -> tuple:
    """
    One concentrated loading roller.
    """

    return (
        PointForce(
            x=float(load_position),
            force=float(load),
        ),
    )


def symmetric_four_point_bending_loads(
    *,
    total_load: float,
    specimen_length: float,
    load_span: float,
) -> tuple:
    """
    Symmetric four-point bending loading.

    total_load is the sum of the two loading-roller forces.
    Each roller therefore applies total_load / 2.

    Roller locations:

        x1 = (L - S_L)/2
        x2 = (L + S_L)/2.
    """

    L = float(
        specimen_length
    )

    span = float(
        load_span
    )

    if L <= 0.0:
        raise ValueError(
            "specimen_length must be positive."
        )

    if span < 0.0 or span >= L:
        raise ValueError(
            "load_span must satisfy 0 <= load_span < specimen_length."
        )

    x1 = 0.5 * (
        L - span
    )

    x2 = 0.5 * (
        L + span
    )

    half_load = (
        0.5
        * float(total_load)
    )

    return (
        PointForce(
            x=x1,
            force=half_load,
        ),
        PointForce(
            x=x2,
            force=half_load,
        ),
    )


def symmetric_support_positions(
    *,
    specimen_length: float,
    support_span: float,
) -> tuple[float, float]:
    """
    Symmetric support positions for a finite specimen.
    """

    L = float(
        specimen_length
    )

    span = float(
        support_span
    )

    if L <= 0.0:
        raise ValueError(
            "specimen_length must be positive."
        )

    if span <= 0.0 or span > L:
        raise ValueError(
            "support_span must satisfy "
            "0 < support_span <= specimen_length."
        )

    return (
        0.5 * (
            L - span
        ),
        0.5 * (
            L + span
        ),
    )