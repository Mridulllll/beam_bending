"""
Two-mode one-dimensional HOST beam formulation.

Kinematics
----------
    u(s,z) = u0(s) - z w'(s) + f1(z) beta1(s) + f2(z) beta2(s)
    w(s,z) = w(s)

Generalized strains
-------------------
    eps_ss   = eps0 + z*kappa + f1*chi1 + f2*chi2
    gamma_sz = f1'*beta1 + f2'*beta2

with
    kappa = -w''
    chi1  = beta1'
    chi2  = beta2'.

The full section generalized order is

    [eps0, kappa, chi1, chi2, beta1, beta2].

After exact N=0 condensation, the reduced constitutive order is

    [kappa, chi1, chi2, beta1, beta2]

with resultants

    [M, H1, H2, Q1, Q2].

The exact uniform-section first-order state is

    y = [w, theta, beta1, beta2, M, V, H1, H2].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import expm

from host.variational_shear_modes import (
    BeamLayer2D,
)


ScalarFunction = Callable[[float], float]


@dataclass(frozen=True)
class WarpingMode:
    f: ScalarFunction
    df: ScalarFunction


@dataclass(frozen=True)
class TwoModeBeamSection:
    width: float
    layers: tuple[BeamLayer2D, ...]
    mode1: WarpingMode
    mode2: WarpingMode
    K6: np.ndarray
    K5: np.ndarray

    def __post_init__(self) -> None:
        width = float(self.width)
        K6 = np.asarray(self.K6, dtype=float)
        K5 = np.asarray(self.K5, dtype=float)

        if width <= 0.0:
            raise ValueError("Beam width must be positive.")

        if K6.shape != (6, 6):
            raise ValueError("K6 must have shape (6, 6).")

        if K5.shape != (5, 5):
            raise ValueError("K5 must have shape (5, 5).")

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "K6", K6)
        object.__setattr__(self, "K5", K5)


def two_mode_thickness_operator(
    z: float,
    mode1: WarpingMode,
    mode2: WarpingMode,
) -> np.ndarray:
    """
    Return the 2x6 beam thickness operator.
    """

    z = float(z)

    return np.asarray(
        [
            [
                1.0,
                z,
                float(mode1.f(z)),
                float(mode2.f(z)),
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                float(mode1.df(z)),
                float(mode2.df(z)),
            ],
        ],
        dtype=float,
    )


def assemble_two_mode_section_stiffness(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
    mode1: WarpingMode,
    mode2: WarpingMode,
    quadrature_order: int = 12,
    integration_breakpoints: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Assemble the full 6x6 two-mode section stiffness.

    ``integration_breakpoints`` may be supplied when either warping mode
    is only piecewise smooth.  Every physical layer is then subdivided at
    those coordinates before Gauss integration.

    This is essential for the section-FE variational mode, whose f2 is
    piecewise linear and whose f2' is piecewise constant.  Integrating one
    physical core layer with a small fixed Gauss rule would otherwise
    under-resolve the many section-mode elements and change the condensed
    shear mechanics.
    """

    layers = tuple(layers)
    width = float(width)

    if width <= 0.0:
        raise ValueError("width must be positive.")

    if not layers:
        raise ValueError("At least one beam layer is required.")

    if int(quadrature_order) < 2:
        raise ValueError("quadrature_order must be >= 2.")

    points, weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )

    if integration_breakpoints is None:
        breakpoints = np.asarray([], dtype=float)
    else:
        breakpoints = np.asarray(
            tuple(float(value) for value in integration_breakpoints),
            dtype=float,
        )

        if breakpoints.ndim != 1:
            raise ValueError("integration_breakpoints must be one-dimensional.")

        if not np.all(np.isfinite(breakpoints)):
            raise ValueError("integration_breakpoints must be finite.")

        breakpoints = np.unique(breakpoints)

    K = np.zeros((6, 6), dtype=float)

    for layer in layers:
        z_layer_0 = float(layer.z0)
        z_layer_1 = float(layer.z1)

        internal = breakpoints[
            (breakpoints > z_layer_0)
            & (breakpoints < z_layer_1)
        ]

        segment_nodes = np.concatenate(
            (
                np.asarray([z_layer_0], dtype=float),
                internal,
                np.asarray([z_layer_1], dtype=float),
            )
        )

        for z0, z1 in zip(
            segment_nodes[:-1],
            segment_nodes[1:],
        ):
            z0 = float(z0)
            z1 = float(z1)

            jac = 0.5 * (z1 - z0)
            midpoint = 0.5 * (z0 + z1)

            for point, weight in zip(points, weights):
                z = midpoint + jac * float(point)

                B = two_mode_thickness_operator(
                    z,
                    mode1,
                    mode2,
                )

                K += (
                    width
                    * jac
                    * float(weight)
                    * (B.T @ layer.C @ B)
                )

    return 0.5 * (K + K.T)


def condense_zero_axial_force(
    K6: np.ndarray,
) -> np.ndarray:
    """
    Condense eps0 from the full section law under N=0.
    """

    K6 = np.asarray(K6, dtype=float)

    if K6.shape != (6, 6):
        raise ValueError("K6 must have shape (6, 6).")

    K00 = float(K6[0, 0])

    if K00 <= 0.0:
        raise ValueError("K6[0,0] must be positive.")

    K0t = K6[0, 1:]
    Ktt = K6[1:, 1:]

    K5 = Ktt - np.outer(K0t, K0t) / K00

    return 0.5 * (K5 + K5.T)


def build_two_mode_beam_section(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
    mode1: WarpingMode,
    mode2: WarpingMode,
    quadrature_order: int = 12,
    integration_breakpoints: Sequence[float] | None = None,
) -> TwoModeBeamSection:
    K6 = assemble_two_mode_section_stiffness(
        layers,
        width=width,
        mode1=mode1,
        mode2=mode2,
        quadrature_order=quadrature_order,
        integration_breakpoints=integration_breakpoints,
    )

    K5 = condense_zero_axial_force(K6)

    if np.min(np.linalg.eigvalsh(K5)) <= 0.0:
        raise ValueError("Condensed two-mode section stiffness is not SPD.")

    return TwoModeBeamSection(
        width=float(width),
        layers=tuple(layers),
        mode1=mode1,
        mode2=mode2,
        K6=K6,
        K5=K5,
    )


def recover_generalized_quantities(
    K5: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover [kappa,chi1,chi2,beta1,beta2] and [Q1,Q2].

    State order:
        [w,theta,beta1,beta2,M,V,H1,H2].
    """

    K5 = np.asarray(K5, dtype=float)
    y = np.asarray(state, dtype=float).reshape(-1)

    if K5.shape != (5, 5):
        raise ValueError("K5 must have shape (5, 5).")

    if y.shape != (8,):
        raise ValueError("state must have 8 components.")

    S = np.linalg.inv(K5)

    beta = y[2:4]
    resultants_a = np.asarray(
        [y[4], y[6], y[7]],
        dtype=float,
    )

    S_beta_a = S[3:5, 0:3]
    S_beta_Q = S[3:5, 3:5]

    Q = np.linalg.solve(
        S_beta_Q,
        beta - S_beta_a @ resultants_a,
    )

    resultants = np.concatenate(
        (resultants_a, Q)
    )

    generalized = S @ resultants

    return generalized, Q


def two_mode_state_derivative(
    K5: np.ndarray,
    state: np.ndarray,
    *,
    q: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the exact constant-section first-order beam equations.
    """

    y = np.asarray(state, dtype=float).reshape(-1)

    generalized, Q = recover_generalized_quantities(
        K5,
        y,
    )

    kappa = float(generalized[0])
    chi1 = float(generalized[1])
    chi2 = float(generalized[2])

    derivative = np.zeros(8, dtype=float)

    derivative[0] = y[1]
    derivative[1] = -kappa
    derivative[2] = chi1
    derivative[3] = chi2
    derivative[4] = y[5]
    derivative[5] = -float(q)
    derivative[6] = Q[0]
    derivative[7] = Q[1]

    return derivative


def two_mode_state_matrix(
    K5: np.ndarray,
) -> np.ndarray:
    """
    Return A such that y' = A y for q=0.
    """

    A = np.zeros((8, 8), dtype=float)

    for j in range(8):
        basis = np.zeros(8, dtype=float)
        basis[j] = 1.0

        A[:, j] = two_mode_state_derivative(
            K5,
            basis,
            q=0.0,
        )

    return A


def propagate_two_mode_segment(
    K5: np.ndarray,
    state0: np.ndarray,
    length: float,
    *,
    q: float = 0.0,
) -> np.ndarray:
    """
    Exact matrix-exponential propagation over a uniform segment.
    """

    length = float(length)
    y0 = np.asarray(state0, dtype=float).reshape(-1)

    if y0.shape != (8,):
        raise ValueError("state0 must have 8 components.")

    A = two_mode_state_matrix(K5)

    if float(q) == 0.0:
        return expm(A * length) @ y0

    # Affine system y' = A y + b.  Use one augmented exponential.
    augmented = np.zeros((9, 9), dtype=float)
    augmented[:8, :8] = A
    augmented[5, 8] = -float(q)

    initial = np.zeros(9, dtype=float)
    initial[:8] = y0
    initial[8] = 1.0

    return (expm(augmented * length) @ initial)[:8]


def local_shear_condensation_ratio(
    K5: np.ndarray,
) -> float:
    """
    Return K_eff/K11 for local Q2=0 condensation of the beta block.
    """

    K5 = np.asarray(K5, dtype=float)

    block = K5[3:5, 3:5]

    K11 = float(block[0, 0])
    K12 = float(block[0, 1])
    K22 = float(block[1, 1])

    return (
        K11 - K12 * K12 / K22
    ) / K11


def boundary_layer_spectrum(
    K5: np.ndarray,
    *,
    zero_tol: float = 1.0e-8,
) -> dict:
    """
    Return longitudinal characteristic exponents and decay lengths.
    """

    A = two_mode_state_matrix(K5)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )
    threshold = float(zero_tol) * scale

    modes = []

    for i, value_raw in enumerate(eigenvalues):
        value = complex(value_raw)

        if abs(value) <= threshold:
            continue

        vector = np.asarray(
            eigenvectors[:, i],
            dtype=complex,
        )

        real = float(value.real)
        imag = float(value.imag)

        decay_length = (
            1.0 / abs(real)
            if abs(real) > threshold
            else float("inf")
        )

        wavelength = (
            2.0 * np.pi / abs(imag)
            if abs(imag) > threshold
            else float("inf")
        )

        beta_norm = float(
            np.sqrt(
                abs(vector[2]) ** 2
                + abs(vector[3]) ** 2
            )
        )

        beta2_fraction = (
            float(abs(vector[3]) / beta_norm)
            if beta_norm > 0.0
            else float("nan")
        )

        modes.append(
            {
                "eigenvalue": value,
                "stable": bool(real < -threshold),
                "unstable": bool(real > threshold),
                "decay_length": float(decay_length),
                "wavelength": float(wavelength),
                "beta2_fraction": beta2_fraction,
            }
        )

    stable = sorted(
        (mode for mode in modes if mode["stable"]),
        key=lambda mode: mode["decay_length"],
        reverse=True,
    )

    return {
        "A": A,
        "zero_eigenvalue_count": int(
            np.sum(np.abs(eigenvalues) <= threshold)
        ),
        "modes": modes,
        "stable_modes": stable,
    }
