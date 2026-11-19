"""
Global Ritz solver for the directional five-field HOST plate.

The solver is intended for static verification problems with mixed/free
boundaries that are not covered by the simply-supported Navier solution.

Kinematics
----------
The five generalized midsurface fields are

    q = [u0, v0, w0, beta_x, beta_y].

The directional five-field generalized-strain ordering is

    [eps0_xx,
     eps0_yy,
     gamma0_xy,
     kappa_xx,
     kappa_yy,
     kappa_xy,
     chi_xx,
     chi_yy,
     chi_xy_x,
     chi_xy_y,
     beta_x,
     beta_y].

with

    eps0_xx  = u0,x
    eps0_yy  = v0,y
    gamma0_xy = u0,y + v0,x

    kappa_xx = -w0,xx
    kappa_yy = -w0,yy
    kappa_xy = -2 w0,xy

    chi_xx   = beta_x,x
    chi_yy   = beta_y,y
    chi_xy_x = beta_x,y
    chi_xy_y = beta_y,x.

The section constitutive matrix supplied to this module must be the
directional 12x12 HOST plate stiffness per unit midsurface width, i.e.

    R = K12 E,

where the energy density per unit midsurface area is

    1/2 E^T K12 E.

Spatial approximation
---------------------
All five generalized fields are expanded in tensor-product orthonormal
Legendre polynomials over the full rectangular planform.  Essential and
gauge constraints are imposed through an exact null-space projection.
Free outer edges are therefore natural boundaries of the Ritz functional.

For the four-point-bending benchmark the line supports are implemented as
width-average constraints.  This is deliberate: it is the plate-level
counterpart of the section-distributing Abaqus fixture, which constrains a
generalized section translation rather than every point on the section.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss
from scipy.linalg import eigh, null_space


FIELD_NAMES = (
    "u0",
    "v0",
    "w0",
    "beta_x",
    "beta_y",
)

DIRECTIONAL_E12_LABELS = (
    "eps0_xx",
    "eps0_yy",
    "gamma0_xy",
    "kappa_xx",
    "kappa_yy",
    "kappa_xy",
    "chi_xx",
    "chi_yy",
    "chi_xy_x",
    "chi_xy_y",
    "beta_x",
    "beta_y",
)


# Each entry is
#
#     (field_index, derivative_order_x, derivative_order_y, coefficient)
#
# and maps one generalized strain component to one or more field derivatives.
_E12_CONTRIBUTIONS = {
    0: ((0, 1, 0, +1.0),),
    1: ((1, 0, 1, +1.0),),
    2: (
        (0, 0, 1, +1.0),
        (1, 1, 0, +1.0),
    ),
    3: ((2, 2, 0, -1.0),),
    4: ((2, 0, 2, -1.0),),
    5: ((2, 1, 1, -2.0),),
    6: ((3, 1, 0, +1.0),),
    7: ((4, 0, 1, +1.0),),
    8: ((3, 0, 1, +1.0),),
    9: ((4, 1, 0, +1.0),),
    10: ((3, 0, 0, +1.0),),
    11: ((4, 0, 0, +1.0),),
}


@dataclass(frozen=True)
class RitzPlateGeometry:
    """Rectangular midsurface geometry."""

    length: float
    width: float

    def __post_init__(self) -> None:
        if float(self.length) <= 0.0:
            raise ValueError("length must be positive.")
        if float(self.width) <= 0.0:
            raise ValueError("width must be positive.")


@dataclass(frozen=True)
class RitzPlateSolution:
    """Solved directional five-field Ritz plate state."""

    geometry: RitzPlateGeometry
    nx_degree: int
    ny_degree: int
    coefficients: np.ndarray

    support_left: float
    support_right: float
    load_left: float
    load_right: float
    total_load: float

    support_reaction_left: float
    support_reaction_right: float

    strain_energy: float
    linear_ramp_external_work: float

    projected_residual_relative: float
    constraint_residual_max: float
    scaled_reduced_condition_number: float

    def _field_coefficients(
        self,
        field: str,
    ) -> np.ndarray:
        if field not in FIELD_NAMES:
            raise ValueError(
                "Unknown field {!r}; expected one of {}.".format(
                    field,
                    FIELD_NAMES,
                )
            )

        field_index = FIELD_NAMES.index(field)

        ny_count = int(self.ny_degree) + 1
        nx_count = int(self.nx_degree) + 1
        n_basis = nx_count * ny_count

        start = field_index * n_basis
        stop = start + n_basis

        return np.asarray(
            self.coefficients[start:stop],
            dtype=float,
        ).reshape(
            nx_count,
            ny_count,
        )

    def field(
        self,
        field: str,
        x: float,
        y: float,
        *,
        dx: int = 0,
        dy: int = 0,
    ) -> float:
        """
        Evaluate a field or one of its derivatives at (x,y).
        """
        coeff = self._field_coefficients(
            field
        )

        X = orthonormal_legendre_values(
            0.0,
            float(self.geometry.length),
            int(self.nx_degree),
            np.asarray([float(x)]),
            derivative_order=int(dx),
        )[:, 0]

        Y = orthonormal_legendre_values(
            -0.5 * float(self.geometry.width),
            +0.5 * float(self.geometry.width),
            int(self.ny_degree),
            np.asarray([float(y)]),
            derivative_order=int(dy),
        )[:, 0]

        return float(
            X @ coeff @ Y
        )

    def width_average(
        self,
        field: str,
        x: float,
    ) -> float:
        """Width-average field value at a fixed x station."""
        coeff = self._field_coefficients(
            field
        )

        X = orthonormal_legendre_values(
            0.0,
            float(self.geometry.length),
            int(self.nx_degree),
            np.asarray([float(x)]),
            derivative_order=0,
        )[:, 0]

        Iy = orthonormal_legendre_integrals(
            -0.5 * float(self.geometry.width),
            +0.5 * float(self.geometry.width),
            int(self.ny_degree),
        )

        return float(
            X @ coeff @ Iy
            / float(self.geometry.width)
        )


def orthonormal_legendre_values(
    lower: float,
    upper: float,
    degree: int,
    coordinates: np.ndarray,
    *,
    derivative_order: int = 0,
) -> np.ndarray:
    """
    Evaluate orthonormal Legendre basis functions on [lower,upper].

    The basis is

        phi_n(x) = sqrt((2n+1)/L) P_n(xi),

        xi = 2(x-lower)/L - 1,

    so that

        integral phi_m phi_n dx = delta_mn.
    """
    lower = float(lower)
    upper = float(upper)
    degree = int(degree)
    derivative_order = int(derivative_order)

    if upper <= lower:
        raise ValueError("upper must exceed lower.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if derivative_order < 0:
        raise ValueError(
            "derivative_order must be non-negative."
        )

    x = np.asarray(
        coordinates,
        dtype=float,
    ).reshape(-1)

    length = upper - lower
    xi = (
        2.0 * (x - lower) / length
        - 1.0
    )

    values = np.empty(
        (degree + 1, x.size),
        dtype=float,
    )

    derivative_scale = (
        2.0 / length
    ) ** derivative_order

    for n in range(degree + 1):
        polynomial = (
            Legendre.basis(n)
            .deriv(derivative_order)
        )

        normalization = math.sqrt(
            (2.0 * n + 1.0)
            / length
        )

        values[n, :] = (
            normalization
            * derivative_scale
            * polynomial(xi)
        )

    return values


def orthonormal_legendre_integrals(
    lower: float,
    upper: float,
    degree: int,
) -> np.ndarray:
    """
    Exact integrals of the orthonormal Legendre basis.

    Only the constant mode has nonzero integral.
    """
    length = float(upper) - float(lower)

    if length <= 0.0:
        raise ValueError(
            "upper must exceed lower."
        )

    result = np.zeros(
        int(degree) + 1,
        dtype=float,
    )

    result[0] = math.sqrt(
        length
    )

    return result


def _gram_1d(
    lower: float,
    upper: float,
    degree: int,
    derivative_left: int,
    derivative_right: int,
    *,
    quadrature_order: int,
) -> np.ndarray:
    xi, weights = leggauss(
        int(quadrature_order)
    )

    lower = float(lower)
    upper = float(upper)

    jacobian = 0.5 * (
        upper - lower
    )

    x = (
        lower
        + (xi + 1.0) * jacobian
    )

    left_values = (
        orthonormal_legendre_values(
            lower,
            upper,
            int(degree),
            x,
            derivative_order=int(
                derivative_left
            ),
        )
    )

    right_values = (
        orthonormal_legendre_values(
            lower,
            upper,
            int(degree),
            x,
            derivative_order=int(
                derivative_right
            ),
        )
    )

    weighted = (
        weights
        * jacobian
    )

    return (
        left_values
        * weighted[np.newaxis, :]
    ) @ right_values.T


def _field_slice(
    field_index: int,
    n_basis: int,
) -> slice:
    start = int(field_index) * int(n_basis)

    return slice(
        start,
        start + int(n_basis),
    )


def assemble_directional_five_field_ritz_stiffness(
    K12_per_unit_width: np.ndarray,
    geometry: RitzPlateGeometry,
    nx_degree: int,
    ny_degree: int,
    *,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """
    Assemble the global Ritz stiffness matrix.

    K12_per_unit_width must have shape (12,12) and must NOT already contain
    integration over the physical plate width.
    """
    K12 = np.asarray(
        K12_per_unit_width,
        dtype=float,
    )

    if K12.shape != (12, 12):
        raise ValueError(
            "K12_per_unit_width must have shape (12,12)."
        )

    K12 = 0.5 * (
        K12 + K12.T
    )

    nx_degree = int(nx_degree)
    ny_degree = int(ny_degree)

    if nx_degree < 2:
        raise ValueError(
            "nx_degree must be at least 2 because w0 enters through "
            "second x derivatives."
        )

    if ny_degree < 2:
        raise ValueError(
            "ny_degree must be at least 2 because w0 enters through "
            "second y derivatives."
        )

    if quadrature_order is None:
        quadrature_order = max(
            2 * max(
                nx_degree,
                ny_degree,
            )
            + 8,
            24,
        )

    quadrature_order = int(
        quadrature_order
    )

    gx = {
        (p, r): _gram_1d(
            0.0,
            float(geometry.length),
            nx_degree,
            p,
            r,
            quadrature_order=quadrature_order,
        )
        for p in range(3)
        for r in range(3)
    }

    gy = {
        (p, r): _gram_1d(
            -0.5 * float(geometry.width),
            +0.5 * float(geometry.width),
            ny_degree,
            p,
            r,
            quadrature_order=quadrature_order,
        )
        for p in range(3)
        for r in range(3)
    }

    n_basis = (
        (nx_degree + 1)
        * (ny_degree + 1)
    )

    n_dof = (
        len(FIELD_NAMES)
        * n_basis
    )

    stiffness = np.zeros(
        (n_dof, n_dof),
        dtype=float,
    )

    for a in range(12):
        for b in range(12):
            constitutive = float(
                K12[a, b]
            )

            if constitutive == 0.0:
                continue

            for (
                field_a,
                px_a,
                py_a,
                coefficient_a,
            ) in _E12_CONTRIBUTIONS[a]:

                slice_a = _field_slice(
                    field_a,
                    n_basis,
                )

                for (
                    field_b,
                    px_b,
                    py_b,
                    coefficient_b,
                ) in _E12_CONTRIBUTIONS[b]:

                    slice_b = _field_slice(
                        field_b,
                        n_basis,
                    )

                    integral = np.kron(
                        gx[(px_a, px_b)],
                        gy[(py_a, py_b)],
                    )

                    stiffness[
                        slice_a,
                        slice_b,
                    ] += (
                        constitutive
                        * float(coefficient_a)
                        * float(coefficient_b)
                        * integral
                    )

    return 0.5 * (
        stiffness + stiffness.T
    )


def _line_integral_basis(
    geometry: RitzPlateGeometry,
    nx_degree: int,
    ny_degree: int,
    x: float,
) -> np.ndarray:
    X = orthonormal_legendre_values(
        0.0,
        float(geometry.length),
        int(nx_degree),
        np.asarray([float(x)]),
        derivative_order=0,
    )[:, 0]

    Iy = (
        orthonormal_legendre_integrals(
            -0.5 * float(geometry.width),
            +0.5 * float(geometry.width),
            int(ny_degree),
        )
    )

    return np.kron(
        X,
        Iy,
    )


def _field_functional(
    geometry: RitzPlateGeometry,
    nx_degree: int,
    ny_degree: int,
    field_index: int,
    local_functional: np.ndarray,
) -> np.ndarray:
    n_basis = (
        (int(nx_degree) + 1)
        * (int(ny_degree) + 1)
    )

    result = np.zeros(
        len(FIELD_NAMES) * n_basis,
        dtype=float,
    )

    result[
        _field_slice(
            int(field_index),
            n_basis,
        )
    ] = np.asarray(
        local_functional,
        dtype=float,
    )

    return result


def _build_4pb_load_vector(
    geometry: RitzPlateGeometry,
    nx_degree: int,
    ny_degree: int,
    load_left: float,
    load_right: float,
    total_load: float,
) -> np.ndarray:
    """
    Uniform transverse line resultants across the physical plate width.

    Positive total_load denotes positive w0 loading.
    """
    n_basis = (
        (int(nx_degree) + 1)
        * (int(ny_degree) + 1)
    )

    force = np.zeros(
        len(FIELD_NAMES) * n_basis,
        dtype=float,
    )

    load_per_line = (
        0.5 * float(total_load)
    )

    line_density = (
        load_per_line
        / float(geometry.width)
    )

    local = (
        _line_integral_basis(
            geometry,
            nx_degree,
            ny_degree,
            load_left,
        )
        + _line_integral_basis(
            geometry,
            nx_degree,
            ny_degree,
            load_right,
        )
    )

    force[
        _field_slice(
            2,
            n_basis,
        )
    ] = (
        line_density
        * local
    )

    return force


def _build_section_distributing_constraints(
    geometry: RitzPlateGeometry,
    nx_degree: int,
    ny_degree: int,
    support_left: float,
    support_right: float,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Essential/gauge constraints for the theory-compatible 4PB fixture.

    Physical section-distributing constraints:
        width-average w0 = 0 at both support sections.

    In-plane stabilization matching the full Abaqus fxsec fixture:
        width-average u0 = 0 at left support,
        width-average v0 = 0 at left and right supports.

    One additional pure gauge removes rigid-body rotation about x:
        coefficient of the global w0 ~ y rigid-tilt mode = 0.

    The last condition changes no generalized strain because w0=c*y is an
    exact rigid rotation in the five-field kinematics.
    """
    width = float(
        geometry.width
    )

    avg_left = (
        _line_integral_basis(
            geometry,
            nx_degree,
            ny_degree,
            support_left,
        )
        / width
    )

    avg_right = (
        _line_integral_basis(
            geometry,
            nx_degree,
            ny_degree,
            support_right,
        )
        / width
    )

    rows = []
    names = []

    rows.append(
        _field_functional(
            geometry,
            nx_degree,
            ny_degree,
            2,
            avg_left,
        )
    )
    names.append(
        "support_w_left_average"
    )

    rows.append(
        _field_functional(
            geometry,
            nx_degree,
            ny_degree,
            2,
            avg_right,
        )
    )
    names.append(
        "support_w_right_average"
    )

    rows.append(
        _field_functional(
            geometry,
            nx_degree,
            ny_degree,
            0,
            avg_left,
        )
    )
    names.append(
        "stabilize_u_left_average"
    )

    rows.append(
        _field_functional(
            geometry,
            nx_degree,
            ny_degree,
            1,
            avg_left,
        )
    )
    names.append(
        "stabilize_v_left_average"
    )

    rows.append(
        _field_functional(
            geometry,
            nx_degree,
            ny_degree,
            1,
            avg_right,
        )
    )
    names.append(
        "stabilize_v_right_average"
    )

    if int(ny_degree) >= 1:
        n_basis = (
            (int(nx_degree) + 1)
            * (int(ny_degree) + 1)
        )

        rigid_tilt = np.zeros(
            len(FIELD_NAMES) * n_basis,
            dtype=float,
        )

        # Coefficient ordering is ix-major, iy-minor.  The normalized
        # P0(x)*P1(y) mode is exactly a rigid w0~y tilt.
        local_index = 1

        rigid_tilt[
            2 * n_basis
            + local_index
        ] = 1.0

        rows.append(
            rigid_tilt
        )
        names.append(
            "gauge_rigid_w_y_tilt"
        )

    constraints = np.vstack(
        rows
    )

    if (
        np.linalg.matrix_rank(
            constraints
        )
        != constraints.shape[0]
    ):
        raise RuntimeError(
            "Ritz constraint matrix is rank deficient."
        )

    return (
        constraints,
        tuple(names),
    )


def solve_directional_five_field_ritz_4pb(
    K12_per_unit_width: np.ndarray,
    *,
    length: float,
    width: float,
    support_left: float,
    support_right: float,
    load_left: float,
    load_right: float,
    total_load: float,
    nx_degree: int,
    ny_degree: int,
    quadrature_order: int | None = None,
) -> RitzPlateSolution:
    """
    Solve the finite-width directional HOST plate for symmetric 4PB.

    The outer planform edges are free natural boundaries.  The two internal
    support sections constrain only width-average w0, matching the generalized
    section-distributing FE reference.
    """
    geometry = RitzPlateGeometry(
        length=float(length),
        width=float(width),
    )

    for name, x in (
        ("support_left", support_left),
        ("support_right", support_right),
        ("load_left", load_left),
        ("load_right", load_right),
    ):
        if not (
            0.0
            <= float(x)
            <= float(length)
        ):
            raise ValueError(
                "{} lies outside the plate.".format(
                    name
                )
            )

    stiffness = (
        assemble_directional_five_field_ritz_stiffness(
            K12_per_unit_width,
            geometry,
            int(nx_degree),
            int(ny_degree),
            quadrature_order=quadrature_order,
        )
    )

    force = _build_4pb_load_vector(
        geometry,
        int(nx_degree),
        int(ny_degree),
        float(load_left),
        float(load_right),
        float(total_load),
    )

    (
        constraints_raw,
        constraint_names,
    ) = (
        _build_section_distributing_constraints(
            geometry,
            int(nx_degree),
            int(ny_degree),
            float(support_left),
            float(support_right),
        )
    )

    row_norms = np.linalg.norm(
        constraints_raw,
        axis=1,
    )

    constraints = (
        constraints_raw
        / row_norms[:, np.newaxis]
    )

    Z = null_space(
        constraints,
        rcond=1.0e-12,
    )

    reduced_stiffness = (
        Z.T @ stiffness @ Z
    )

    reduced_force = (
        Z.T @ force
    )

    reduced_stiffness = 0.5 * (
        reduced_stiffness
        + reduced_stiffness.T
    )

    diagonal = np.diag(
        reduced_stiffness
    )

    diagonal_max = float(
        np.max(diagonal)
    )

    if diagonal_max <= 0.0:
        raise RuntimeError(
            "Reduced Ritz stiffness has no positive diagonal scale."
        )

    diagonal_floor = (
        diagonal_max
        * 1.0e-18
    )

    scaling = (
        1.0
        / np.sqrt(
            np.maximum(
                diagonal,
                diagonal_floor,
            )
        )
    )

    scaled_stiffness = (
        scaling[:, np.newaxis]
        * reduced_stiffness
        * scaling[np.newaxis, :]
    )

    scaled_force = (
        scaling
        * reduced_force
    )

    eigenvalues, eigenvectors = eigh(
        scaled_stiffness
    )

    eigenvalue_max = float(
        eigenvalues[-1]
    )

    negative_tolerance = (
        max(
            1.0,
            abs(eigenvalue_max),
        )
        * 1.0e-12
    )

    if float(eigenvalues[0]) <= 0.0:
        if float(eigenvalues[0]) < -negative_tolerance:
            raise RuntimeError(
                "Reduced Ritz stiffness is indefinite: min eigenvalue={:.6e}."
                .format(
                    float(eigenvalues[0])
                )
            )

        raise RuntimeError(
            "Reduced Ritz stiffness retains a zero-energy mode: "
            "min eigenvalue={:.6e}.".format(
                float(eigenvalues[0])
            )
        )

    transformed_force = (
        eigenvectors.T
        @ scaled_force
    )

    scaled_solution = (
        eigenvectors
        @ (
            transformed_force
            / eigenvalues
        )
    )

    reduced_solution = (
        scaling
        * scaled_solution
    )

    coefficients = (
        Z @ reduced_solution
    )

    projected_residual = (
        Z.T
        @ (
            stiffness @ coefficients
            - force
        )
    )

    projected_force_norm = max(
        float(
            np.linalg.norm(
                reduced_force
            )
        ),
        1.0e-30,
    )

    projected_residual_relative = float(
        np.linalg.norm(
            projected_residual
        )
        / projected_force_norm
    )

    constraint_residual_max = float(
        np.max(
            np.abs(
                constraints_raw
                @ coefficients
            )
        )
    )

    # Recover physical Lagrange multipliers from
    #
    #     K q + C^T lambda = f
    #
    # using the unnormalized physical constraint rows.
    residual_force = (
        force
        - stiffness @ coefficients
    )

    multipliers = np.linalg.lstsq(
        constraints_raw.T,
        residual_force,
        rcond=None,
    )[0]

    multiplier_by_name = {
        name: float(value)
        for name, value
        in zip(
            constraint_names,
            multipliers,
        )
    }

    strain_energy = (
        0.5
        * float(
            coefficients
            @ stiffness
            @ coefficients
        )
    )

    linear_ramp_external_work = (
        0.5
        * float(
            force
            @ coefficients
        )
    )

    condition_number = float(
        eigenvalues[-1]
        / eigenvalues[0]
    )

    return RitzPlateSolution(
        geometry=geometry,
        nx_degree=int(nx_degree),
        ny_degree=int(ny_degree),
        coefficients=np.asarray(
            coefficients,
            dtype=float,
        ),
        support_left=float(
            support_left
        ),
        support_right=float(
            support_right
        ),
        load_left=float(
            load_left
        ),
        load_right=float(
            load_right
        ),
        total_load=float(
            total_load
        ),
        support_reaction_left=float(
            multiplier_by_name[
                "support_w_left_average"
            ]
        ),
        support_reaction_right=float(
            multiplier_by_name[
                "support_w_right_average"
            ]
        ),
        strain_energy=float(
            strain_energy
        ),
        linear_ramp_external_work=float(
            linear_ramp_external_work
        ),
        projected_residual_relative=float(
            projected_residual_relative
        ),
        constraint_residual_max=float(
            constraint_residual_max
        ),
        scaled_reduced_condition_number=float(
            condition_number
        ),
    )
