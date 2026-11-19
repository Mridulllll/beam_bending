"""
Multimode HOST buckling under combined membrane loading.

Reference membrane resultants:

    Nx_ref
    Ny_ref
    Nxy_ref

Positive Nx_ref and Ny_ref denote compression.

The buckling problem is

    (K_E - lambda K_G) W = 0.

Each Navier mode first undergoes exact elastic static condensation:

    [U, V, W, X, Y] -> W.

The resulting elastic multimode matrix is diagonal because distinct
Navier modes are elastically orthogonal.

Normal prestresses Nx, Ny also produce diagonal geometric terms.

In-plane shear Nxy couples different Navier modes and therefore
requires the present multimode generalized eigenvalue formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig

from host.condensation import (
    condense_to_transverse_stiffness,
)

from host.navier import (
    directional_navier_modal_stiffness,
    navier_modal_stiffness_basis_aware,
    navier_mode,
)


@dataclass(frozen=True)
class NavierBucklingMode:
    m: int
    n: int


@dataclass(frozen=True)
class MultimodeBucklingResult:
    load_factor: float
    eigenvector: np.ndarray
    modes: tuple
    dominant_mode: NavierBucklingMode
    elastic_matrix: np.ndarray
    geometric_matrix: np.ndarray


def build_mode_list(
    max_m: int,
    max_n: int,
):
    if max_m < 1 or max_n < 1:
        raise ValueError(
            "Mode limits must be positive."
        )

    return tuple(
        NavierBucklingMode(
            m=m,
            n=n,
        )
        for m in range(
            1,
            max_m + 1,
        )
        for n in range(
            1,
            max_n + 1,
        )
    )


def _integer_sine_integral(k: int) -> float:
    """
    Integral on t in [0,1]:

        integral sin(k*pi*t) dt.

    Valid for positive or negative integer k.
    """

    k = int(k)

    if k == 0:
        return 0.0

    return (
        1.0
        - (-1.0)**k
    ) / (
        k * np.pi
    )


def _integral_cos_sin(
    first: int,
    second: int,
    length: float,
) -> float:
    """
    integral_0^L

        cos(first*pi*x/L)
        sin(second*pi*x/L)

    dx
    """

    return (
        0.5
        * float(length)
        * (
            _integer_sine_integral(
                second + first
            )
            + _integer_sine_integral(
                second - first
            )
        )
    )


def _integral_sin_cos(
    first: int,
    second: int,
    length: float,
) -> float:
    """
    integral_0^L

        sin(first*pi*x/L)
        cos(second*pi*x/L)

    dx
    """

    return (
        0.5
        * float(length)
        * (
            _integer_sine_integral(
                first + second
            )
            + _integer_sine_integral(
                first - second
            )
        )
    )


def geometric_coupling_entry(
    mode_i: NavierBucklingMode,
    mode_j: NavierBucklingMode,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    Nxy_ref: float = 0.0,
) -> float:
    """
    Evaluate one multimode geometric-stiffness entry.

    With

        phi_mn =
        sin(m*pi*x/a) sin(n*pi*y/b),

    the bilinear geometric form is

        integral [
            Nx  phi_i,x phi_j,x
          + Ny  phi_i,y phi_j,y
          + Nxy (
                phi_i,x phi_j,y
              + phi_i,y phi_j,x
            )
        ] dA.

    This produces a symmetric geometric matrix.
    """

    a = float(a)
    b = float(b)

    if a <= 0.0 or b <= 0.0:
        raise ValueError(
            "Plate dimensions must be positive."
        )

    mi = int(mode_i.m)
    ni = int(mode_i.n)

    mj = int(mode_j.m)
    nj = int(mode_j.n)

    alpha_i = mi * np.pi / a
    beta_i = ni * np.pi / b

    alpha_j = mj * np.pi / a
    beta_j = nj * np.pi / b

    value = 0.0

    # --------------------------------------------------------------
    # Normal prestress.
    #
    # Orthogonality means these contributions are diagonal in
    # (m,n) mode space.
    # --------------------------------------------------------------

    if mi == mj and ni == nj:
        area_factor = (
            a * b / 4.0
        )

        value += (
            float(Nx_ref)
            * alpha_i
            * alpha_j
            * area_factor
        )

        value += (
            float(Ny_ref)
            * beta_i
            * beta_j
            * area_factor
        )

    # --------------------------------------------------------------
    # Shear prestress.
    # --------------------------------------------------------------

    if Nxy_ref != 0.0:
        # phi_i,x * phi_j,y
        Ix_1 = _integral_cos_sin(
            mi,
            mj,
            a,
        )

        Iy_1 = _integral_sin_cos(
            ni,
            nj,
            b,
        )

        term_1 = (
            alpha_i
            * beta_j
            * Ix_1
            * Iy_1
        )

        # phi_i,y * phi_j,x
        Ix_2 = _integral_sin_cos(
            mi,
            mj,
            a,
        )

        Iy_2 = _integral_cos_sin(
            ni,
            nj,
            b,
        )

        term_2 = (
            beta_i
            * alpha_j
            * Ix_2
            * Iy_2
        )

        value += (
            float(Nxy_ref)
            * (
                term_1
                + term_2
            )
        )

    return float(value)


def assemble_multimode_buckling_matrices(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    Nxy_ref: float = 0.0,
    max_m: int = 6,
    max_n: int = 6,
):
    """
    Assemble condensed elastic and geometric multimode matrices.
    """

    if (
        Nx_ref == 0.0
        and Ny_ref == 0.0
        and Nxy_ref == 0.0
    ):
        raise ValueError(
            "At least one reference membrane resultant is required."
        )

    modes = build_mode_list(
        max_m,
        max_n,
    )

    n_modes = len(
        modes
    )

    K_E = np.zeros(
        (n_modes, n_modes),
        dtype=float,
    )

    K_G = np.zeros(
        (n_modes, n_modes),
        dtype=float,
    )

    # --------------------------------------------------------------
    # Elastic stiffness.
    # --------------------------------------------------------------

    for i, mode in enumerate(
        modes
    ):
        K_modal = (
            navier_modal_stiffness_basis_aware(
                K_hsdt,
                a=a,
                b=b,
                m=mode.m,
                n=mode.n,
            )
        )

        K_E[i, i] = (
            condense_to_transverse_stiffness(
                K_modal
            )
        )

    # --------------------------------------------------------------
    # Geometric stiffness.
    # --------------------------------------------------------------

    for i, mode_i in enumerate(
        modes
    ):
        for j in range(
            i,
            n_modes,
        ):
            mode_j = modes[j]

            value = (
                geometric_coupling_entry(
                    mode_i,
                    mode_j,
                    a,
                    b,
                    Nx_ref=Nx_ref,
                    Ny_ref=Ny_ref,
                    Nxy_ref=Nxy_ref,
                )
            )

            K_G[i, j] = value
            K_G[j, i] = value

    return (
        K_E,
        K_G,
        modes,
    )


def solve_multimode_buckling(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    Nxy_ref: float = 0.0,
    max_m: int = 6,
    max_n: int = 6,
    imaginary_tolerance: float = 1.0e-8,
) -> MultimodeBucklingResult:
    """
    Solve

        K_E v = lambda K_G v

    and return the smallest positive finite real eigenvalue.
    """

    K_E, K_G, modes = (
        assemble_multimode_buckling_matrices(
            K_hsdt,
            a,
            b,
            Nx_ref=Nx_ref,
            Ny_ref=Ny_ref,
            Nxy_ref=Nxy_ref,
            max_m=max_m,
            max_n=max_n,
        )
    )

    eigenvalues, eigenvectors = eig(
        K_E,
        K_G,
    )

    candidates = []

    for i, value in enumerate(
        eigenvalues
    ):
        if not np.isfinite(
            value.real
        ):
            continue

        if abs(
            value.imag
        ) > (
            imaginary_tolerance
            * max(
                1.0,
                abs(value.real),
            )
        ):
            continue

        lam = float(
            value.real
        )

        if lam <= 0.0:
            continue

        candidates.append(
            (
                lam,
                i,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No positive finite real buckling eigenvalue was found."
        )

    lam, index = min(
        candidates,
        key=lambda item: item[0],
    )

    vector = np.real(
        eigenvectors[:, index]
    )

    norm = np.linalg.norm(
        vector
    )

    if norm > 0.0:
        vector = vector / norm

    dominant_index = int(
        np.argmax(
            np.abs(vector)
        )
    )

    return MultimodeBucklingResult(
        load_factor=lam,
        eigenvector=vector,
        modes=modes,
        dominant_mode=modes[
            dominant_index
        ],
        elastic_matrix=K_E,
        geometric_matrix=K_G,
    )

# ==================================================================
# Directional-warping multimode buckling
# ==================================================================

def assemble_directional_multimode_buckling_matrices(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    Nxy_ref: float = 0.0,
    max_m: int = 6,
    max_n: int = 6,
):
    """
    Assemble directional-HSDT condensed elastic and geometric
    multimode buckling matrices.

    The elastic modal stiffness uses the 12-component directional
    five-field theory.

    The geometric stiffness is unchanged because it is generated by
    the membrane prestress acting on the transverse displacement field.
    """

    if (
        Nx_ref == 0.0
        and Ny_ref == 0.0
        and Nxy_ref == 0.0
    ):
        raise ValueError(
            "At least one reference membrane resultant is required."
        )

    modes = build_mode_list(
        max_m,
        max_n,
    )

    n_modes = len(
        modes
    )

    K_E = np.zeros(
        (n_modes, n_modes),
        dtype=float,
    )

    K_G = np.zeros(
        (n_modes, n_modes),
        dtype=float,
    )

    # --------------------------------------------------------------
    # Directional elastic stiffness.
    # --------------------------------------------------------------

    for i, mode in enumerate(
        modes
    ):
        K_modal = (
            directional_navier_modal_stiffness(
                K_hsdt,
                a=a,
                b=b,
                m=mode.m,
                n=mode.n,
            )
        )

        K_E[i, i] = (
            condense_to_transverse_stiffness(
                K_modal
            )
        )

    # --------------------------------------------------------------
    # Geometric stiffness.
    #
    # Identical to the scalar formulation.
    # --------------------------------------------------------------

    for i, mode_i in enumerate(
        modes
    ):
        for j in range(
            i,
            n_modes,
        ):
            mode_j = modes[j]

            value = geometric_coupling_entry(
                mode_i,
                mode_j,
                a,
                b,
                Nx_ref=Nx_ref,
                Ny_ref=Ny_ref,
                Nxy_ref=Nxy_ref,
            )

            K_G[i, j] = value
            K_G[j, i] = value

    return (
        K_E,
        K_G,
        modes,
    )


def solve_directional_multimode_buckling(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    *,
    Nx_ref: float = 0.0,
    Ny_ref: float = 0.0,
    Nxy_ref: float = 0.0,
    max_m: int = 6,
    max_n: int = 6,
    imaginary_tolerance: float = 1.0e-8,
) -> MultimodeBucklingResult:
    """
    Solve the directional multimode buckling problem

        K_E v = lambda K_G v

    and return the smallest positive finite real eigenvalue.
    """

    K_E, K_G, modes = (
        assemble_directional_multimode_buckling_matrices(
            K_hsdt,
            a,
            b,
            Nx_ref=Nx_ref,
            Ny_ref=Ny_ref,
            Nxy_ref=Nxy_ref,
            max_m=max_m,
            max_n=max_n,
        )
    )

    eigenvalues, eigenvectors = eig(
        K_E,
        K_G,
    )

    candidates = []

    for i, value in enumerate(
        eigenvalues
    ):
        if not np.isfinite(
            value.real
        ):
            continue

        if abs(
            value.imag
        ) > (
            imaginary_tolerance
            * max(
                1.0,
                abs(value.real),
            )
        ):
            continue

        lam = float(
            value.real
        )

        if lam <= 0.0:
            continue

        candidates.append(
            (
                lam,
                i,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No positive finite real directional buckling "
            "eigenvalue was found."
        )

    lam, index = min(
        candidates,
        key=lambda item: item[0],
    )

    vector = np.real(
        eigenvectors[:, index]
    )

    norm = np.linalg.norm(
        vector
    )

    if norm > 0.0:
        vector = vector / norm

    dominant_index = int(
        np.argmax(
            np.abs(vector)
        )
    )

    return MultimodeBucklingResult(
        load_factor=lam,
        eigenvector=vector,
        modes=modes,
        dominant_mode=modes[
            dominant_index
        ],
        elastic_matrix=K_E,
        geometric_matrix=K_G,
    )