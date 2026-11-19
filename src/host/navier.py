"""
Navier solution for the five-field HOST HSDT.

Simply supported rectangular plate:

    0 <= x <= a
    0 <= y <= b

Modal displacement fields:

    u0     = U cos(alpha*x) sin(beta*y)
    v0     = V sin(alpha*x) cos(beta*y)
    w0     = W sin(alpha*x) sin(beta*y)
    beta_x = X cos(alpha*x) sin(beta*y)
    beta_y = Y sin(alpha*x) cos(beta*y)

with

    alpha = m*pi/a
    beta  = n*pi/b.

For each (m,n) mode the plate problem reduces to

    K_mn q_mn = f_mn

with

    q_mn = [U, V, W, X, Y]^T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NavierMode:
    m: int
    n: int
    alpha: float
    beta: float


def navier_mode(
    a: float,
    b: float,
    m: int,
    n: int,
) -> NavierMode:
    a = float(a)
    b = float(b)

    if a <= 0.0 or b <= 0.0:
        raise ValueError(
            "Plate dimensions a and b must be positive."
        )

    if m < 1 or n < 1:
        raise ValueError(
            "Navier mode numbers m and n must be positive integers."
        )

    return NavierMode(
        m=int(m),
        n=int(n),
        alpha=float(m) * np.pi / a,
        beta=float(n) * np.pi / b,
    )


def generalized_strain_amplitudes(
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Return the 11x5 modal generalized-strain operator G_mn.

    The generalized strain amplitudes satisfy

        E_H = G_mn q_mn

    after accounting for the compatible sine/cosine modal factors.

    Ordering of E_H:

        [eps0_xx,
         eps0_yy,
         gamma0_xy,

         kappa_xx,
         kappa_yy,
         kappa_xy,

         chi_xx,
         chi_yy,
         chi_xy,

         beta_x,
         beta_y]
    """

    a = float(alpha)
    b = float(beta)

    G = np.zeros((11, 5), dtype=float)

    # --------------------------------------------------------------
    # Membrane strains
    # --------------------------------------------------------------

    # eps0_xx = u0,x
    G[0, 0] = -a

    # eps0_yy = v0,y
    G[1, 1] = -b

    # gamma0_xy = u0,y + v0,x
    G[2, 0] = +b
    G[2, 1] = +a

    # --------------------------------------------------------------
    # Bending curvatures
    #
    # kappa_xx = -w0,xx = alpha^2 W
    # kappa_yy = -w0,yy = beta^2 W
    # kappa_xy = -2*w0,xy
    #
    # w0,xy carries cos(alpha*x) cos(beta*y), while the normal
    # curvatures carry sin*sin. Its modal amplitude is therefore
    # represented separately through the same scalar W.
    # --------------------------------------------------------------

    G[3, 2] = a**2
    G[4, 2] = b**2
    G[5, 2] = -2.0 * a * b

    # --------------------------------------------------------------
    # Higher-order warping strains
    # --------------------------------------------------------------

    G[6, 3] = -a
    G[7, 4] = -b

    G[8, 3] = +b
    G[8, 4] = +a

    # --------------------------------------------------------------
    # Shear generalized variables
    # --------------------------------------------------------------

    G[9, 3] = 1.0
    G[10, 4] = 1.0

    return G


def navier_modal_stiffness(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Construct the 5x5 modal stiffness matrix.

    The orthogonality integrals of squared sine/cosine modes over the
    rectangular plate give the common area factor a*b/4 for m,n >= 1.
    """

    K_hsdt = np.asarray(
        K_hsdt,
        dtype=float,
    )

    if K_hsdt.shape != (11, 11):
        raise ValueError(
            "HSDT stiffness must have shape (11, 11)."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    G = generalized_strain_amplitudes(
        mode.alpha,
        mode.beta,
    )

    area_factor = (
        float(a)
        * float(b)
        / 4.0
    )

    Kmn = (
        area_factor
        * (G.T @ K_hsdt @ G)
    )

    return 0.5 * (
        Kmn + Kmn.T
    )


def uniform_pressure_coefficient(
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> float:
    """
    Fourier coefficient for uniform pressure q0.

    For the sine-sine Navier expansion:

        q(x,y) =
        sum q_mn sin(m*pi*x/a) sin(n*pi*y/b)

    the coefficient is

        q_mn = 16*q0/(m*n*pi^2)

    for odd m,n, and zero otherwise.
    """

    if m % 2 == 0 or n % 2 == 0:
        return 0.0

    return (
        16.0
        * float(q0)
        / (
            float(m)
            * float(n)
            * np.pi**2
        )
    )


def navier_modal_load(
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Return the 5-component modal force vector for uniform pressure.

    Only the W amplitude is directly loaded.
    """

    qmn = uniform_pressure_coefficient(
        q0,
        a,
        b,
        m,
        n,
    )

    area_factor = (
        float(a)
        * float(b)
        / 4.0
    )

    f = np.zeros(5, dtype=float)

    f[2] = (
        area_factor
        * qmn
    )

    return f


def solve_navier_mode(
    K_hsdt: np.ndarray,
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Solve one Navier mode.

    Returns

        [U, V, W, X, Y]^T.
    """

    Kmn = navier_modal_stiffness(
        K_hsdt,
        a,
        b,
        m,
        n,
    )

    fmn = navier_modal_load(
        q0,
        a,
        b,
        m,
        n,
    )

    return np.linalg.solve(
        Kmn,
        fmn,
    )

def generalized_strain_basis_operators(
    alpha: float,
    beta: float,
):
    """
    Return the four orthogonal modal generalized-strain operators.

    Spatial basis families:

        SS = sin(alpha*x) sin(beta*y)
        CC = cos(alpha*x) cos(beta*y)
        CS = cos(alpha*x) sin(beta*y)
        SC = sin(alpha*x) cos(beta*y)

    Each operator has shape (11, 5), acting on

        q_mn = [U, V, W, X, Y]^T.

    This representation is valid for a fully coupled anisotropic
    HSDT stiffness because cross-products between different modal
    basis families vanish by orthogonality.
    """

    a = float(alpha)
    b = float(beta)

    G_ss = np.zeros(
        (11, 5),
        dtype=float,
    )

    G_cc = np.zeros(
        (11, 5),
        dtype=float,
    )

    G_cs = np.zeros(
        (11, 5),
        dtype=float,
    )

    G_sc = np.zeros(
        (11, 5),
        dtype=float,
    )

    # --------------------------------------------------------------
    # SS family
    # --------------------------------------------------------------

    # eps0_xx
    G_ss[0, 0] = -a

    # eps0_yy
    G_ss[1, 1] = -b

    # kappa_xx
    G_ss[3, 2] = a**2

    # kappa_yy
    G_ss[4, 2] = b**2

    # chi_xx
    G_ss[6, 3] = -a

    # chi_yy
    G_ss[7, 4] = -b

    # --------------------------------------------------------------
    # CC family
    # --------------------------------------------------------------

    # gamma0_xy
    G_cc[2, 0] = b
    G_cc[2, 1] = a

    # kappa_xy
    G_cc[5, 2] = -2.0 * a * b

    # chi_xy
    G_cc[8, 3] = b
    G_cc[8, 4] = a

    # --------------------------------------------------------------
    # CS family
    # --------------------------------------------------------------

    G_cs[9, 3] = 1.0

    # --------------------------------------------------------------
    # SC family
    # --------------------------------------------------------------

    G_sc[10, 4] = 1.0

    return {
        "SS": G_ss,
        "CC": G_cc,
        "CS": G_cs,
        "SC": G_sc,
    }


def navier_modal_stiffness_basis_aware(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Rigorous basis-aware 5x5 Navier modal stiffness.

    For integer Navier modes,

        integral SS^2 dA =
        integral CC^2 dA =
        integral CS^2 dA =
        integral SC^2 dA = a*b/4,

    while all cross-family products vanish.

    Hence

        K_mn = (a*b/4) sum_r G_r^T K_H G_r.
    """

    K_hsdt = np.asarray(
        K_hsdt,
        dtype=float,
    )

    if K_hsdt.shape != (11, 11):
        raise ValueError(
            "HSDT stiffness must have shape (11, 11)."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    operators = (
        generalized_strain_basis_operators(
            mode.alpha,
            mode.beta,
        )
    )

    Kmn = np.zeros(
        (5, 5),
        dtype=float,
    )

    for G in operators.values():
        Kmn += G.T @ K_hsdt @ G

    Kmn *= (
        float(a)
        * float(b)
        / 4.0
    )

    return 0.5 * (
        Kmn + Kmn.T
    )


def solve_navier_mode_basis_aware(
    K_hsdt: np.ndarray,
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Solve one rigorous basis-aware Navier mode.
    """

    Kmn = navier_modal_stiffness_basis_aware(
        K_hsdt,
        a,
        b,
        m,
        n,
    )

    fmn = navier_modal_load(
        q0,
        a,
        b,
        m,
        n,
    )

    return np.linalg.solve(
        Kmn,
        fmn,
    )


def center_deflection_uniform_pressure_hsdt(
    K_hsdt: np.ndarray,
    q0: float,
    a: float,
    b: float,
    max_mode: int = 31,
) -> float:
    """
    Center deflection of the simply supported HSDT plate under
    uniform pressure using odd Navier modes.
    """

    w = 0.0

    for m in range(
        1,
        max_mode + 1,
        2,
    ):
        for n in range(
            1,
            max_mode + 1,
            2,
        ):
            modal = (
                solve_navier_mode_basis_aware(
                    K_hsdt,
                    q0,
                    a,
                    b,
                    m,
                    n,
                )
            )

            Wmn = modal[2]

            w += (
                Wmn
                * np.sin(
                    m * np.pi / 2.0
                )
                * np.sin(
                    n * np.pi / 2.0
                )
            )

    return float(w)


# ==================================================================
# Directional-warping Navier formulation
# ==================================================================

def directional_generalized_strain_basis_operators(
    alpha: float,
    beta: float,
):
    """
    Return the four orthogonal directional-HSDT modal
    generalized-strain operators.

    Spatial basis families:

        SS = sin(alpha*x) sin(beta*y)
        CC = cos(alpha*x) cos(beta*y)
        CS = cos(alpha*x) sin(beta*y)
        SC = sin(alpha*x) cos(beta*y)

    Each operator has shape (12, 5) and acts on

        q_mn = [U, V, W, X, Y]^T.

    Directional five-field generalized strain ordering:

        0   eps0_xx
        1   eps0_yy
        2   gamma0_xy

        3   kappa_xx
        4   kappa_yy
        5   kappa_xy

        6   chi_xx
        7   chi_yy
        8   chi_xy_x
        9   chi_xy_y

        10  beta_x
        11  beta_y
    """

    a = float(alpha)
    b = float(beta)

    G_ss = np.zeros((12, 5), dtype=float)
    G_cc = np.zeros((12, 5), dtype=float)
    G_cs = np.zeros((12, 5), dtype=float)
    G_sc = np.zeros((12, 5), dtype=float)

    # --------------------------------------------------------------
    # SS family
    # --------------------------------------------------------------

    # eps0_xx = u0,x
    G_ss[0, 0] = -a

    # eps0_yy = v0,y
    G_ss[1, 1] = -b

    # kappa_xx = -w0,xx
    G_ss[3, 2] = a**2

    # kappa_yy = -w0,yy
    G_ss[4, 2] = b**2

    # chi_xx = beta_x,x
    G_ss[6, 3] = -a

    # chi_yy = beta_y,y
    G_ss[7, 4] = -b

    # --------------------------------------------------------------
    # CC family
    # --------------------------------------------------------------

    # gamma0_xy = u0,y + v0,x
    G_cc[2, 0] = b
    G_cc[2, 1] = a

    # kappa_xy = -2*w0,xy
    G_cc[5, 2] = -2.0 * a * b

    # chi_xy_x = beta_x,y
    G_cc[8, 3] = b

    # chi_xy_y = beta_y,x
    G_cc[9, 4] = a

    # --------------------------------------------------------------
    # CS family
    # --------------------------------------------------------------

    # beta_x
    G_cs[10, 3] = 1.0

    # --------------------------------------------------------------
    # SC family
    # --------------------------------------------------------------

    # beta_y
    G_sc[11, 4] = 1.0

    return {
        "SS": G_ss,
        "CC": G_cc,
        "CS": G_cs,
        "SC": G_sc,
    }


def directional_navier_modal_stiffness(
    K_hsdt: np.ndarray,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Construct the rigorous basis-aware 5x5 directional-HSDT
    Navier modal stiffness matrix.

    For integer Navier modes,

        integral SS^2 dA =
        integral CC^2 dA =
        integral CS^2 dA =
        integral SC^2 dA = a*b/4,

    while cross-products between distinct basis families vanish.
    """

    K_hsdt = np.asarray(
        K_hsdt,
        dtype=float,
    )

    if K_hsdt.shape != (12, 12):
        raise ValueError(
            "Directional HSDT stiffness must have shape (12, 12)."
        )

    mode = navier_mode(
        a,
        b,
        m,
        n,
    )

    operators = directional_generalized_strain_basis_operators(
        mode.alpha,
        mode.beta,
    )

    Kmn = np.zeros(
        (5, 5),
        dtype=float,
    )

    for G in operators.values():
        Kmn += (
            G.T
            @ K_hsdt
            @ G
        )

    Kmn *= (
        float(a)
        * float(b)
        / 4.0
    )

    return 0.5 * (
        Kmn + Kmn.T
    )


def solve_directional_navier_mode(
    K_hsdt: np.ndarray,
    q0: float,
    a: float,
    b: float,
    m: int,
    n: int,
) -> np.ndarray:
    """
    Solve one directional-HSDT Navier mode.

    Returns

        [U, V, W, X, Y]^T.
    """

    Kmn = directional_navier_modal_stiffness(
        K_hsdt,
        a,
        b,
        m,
        n,
    )

    fmn = navier_modal_load(
        q0,
        a,
        b,
        m,
        n,
    )

    return np.linalg.solve(
        Kmn,
        fmn,
    )


def center_deflection_uniform_pressure_directional_hsdt(
    K_hsdt: np.ndarray,
    q0: float,
    a: float,
    b: float,
    max_mode: int = 31,
) -> float:
    """
    Center deflection of a simply supported directional-HSDT plate
    under uniform transverse pressure using odd Navier modes.
    """

    if max_mode < 1:
        raise ValueError(
            "max_mode must be at least 1."
        )

    w = 0.0

    for m in range(
        1,
        max_mode + 1,
        2,
    ):
        for n in range(
            1,
            max_mode + 1,
            2,
        ):
            modal = solve_directional_navier_mode(
                K_hsdt,
                q0,
                a,
                b,
                m,
                n,
            )

            Wmn = modal[2]

            w += (
                Wmn
                * np.sin(
                    m * np.pi / 2.0
                )
                * np.sin(
                    n * np.pi / 2.0
                )
            )

    return float(w)