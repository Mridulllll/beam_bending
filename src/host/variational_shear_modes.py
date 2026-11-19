"""
Variational through-thickness shear/warping modes for HOST beams.

The current one-mode HOST beam uses an odd warping function f1(z).  This
module derives an additional odd mode f2(z) from section constitutive
properties only.

On the symmetric half-section z in [0, h/2], the constrained weak problem is

    int G(z) f2'(z) v'(z) dz
        = lambda int E(z) f2(z) v(z) dz,

subject to

    f2(0) = 0,
    int E(z) f1(z) f2(z) dz = 0.

The outer-surface condition G f2' = 0 and interface continuity of f2 and
G f2' follow naturally from the weak form.

The section eigenproblem is discretized with linear 1-D finite elements.
This is a section-level basis generator; it is independent of the later
longitudinal beam discretization/solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import eigh, null_space


ScalarFunction = Callable[[float], float]


@dataclass(frozen=True)
class BeamLayer2D:
    """
    One beam layer after free-stress reduction.

    ``C`` uses physical strain order [eps_ss, gamma_sz] and matching
    stress order [sigma_ss, tau_sz].
    """

    name: str
    z0: float
    z1: float
    C: np.ndarray

    def __post_init__(self) -> None:
        z0 = float(self.z0)
        z1 = float(self.z1)
        C = np.asarray(self.C, dtype=float)

        if not z1 > z0:
            raise ValueError("BeamLayer2D requires z1 > z0.")

        if C.shape != (2, 2):
            raise ValueError("BeamLayer2D.C must have shape (2, 2).")

        if not np.all(np.isfinite(C)):
            raise ValueError("BeamLayer2D.C must contain finite values.")

        C = 0.5 * (C + C.T)

        if np.min(np.linalg.eigvalsh(C)) <= 0.0:
            raise ValueError("BeamLayer2D.C must be positive definite.")

        object.__setattr__(self, "z0", z0)
        object.__setattr__(self, "z1", z1)
        object.__setattr__(self, "C", C)


@dataclass(frozen=True)
class VariationalShearMode:
    """
    Piecewise-linear odd warping mode on a symmetric section.

    ``nodes_half`` and ``values_half`` describe z >= 0.  ``values_half[0]``
    is zero.  The full mode is obtained by odd continuation; its derivative
    is even.
    """

    eigenvalue: float
    nodes_half: np.ndarray
    values_half: np.ndarray
    E_orthogonality_residual: float
    modal_E_norm: float
    modal_G_gradient_norm: float

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes_half, dtype=float)
        values = np.asarray(self.values_half, dtype=float)

        if nodes.ndim != 1 or values.ndim != 1:
            raise ValueError("nodes_half and values_half must be 1-D.")

        if nodes.shape != values.shape or len(nodes) < 2:
            raise ValueError("nodes_half/values_half shape mismatch.")

        if abs(nodes[0]) > 1.0e-14:
            raise ValueError("VariationalShearMode must start at z=0.")

        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("nodes_half must be strictly increasing.")

        if abs(values[0]) > 1.0e-10:
            raise ValueError("Odd mode requires f(0)=0.")

        if float(self.eigenvalue) <= 0.0:
            raise ValueError("eigenvalue must be positive.")

        object.__setattr__(self, "nodes_half", nodes)
        object.__setattr__(self, "values_half", values)

    @property
    def characteristic_length(self) -> float:
        return 1.0 / np.sqrt(float(self.eigenvalue))

    @property
    def slopes_half(self) -> np.ndarray:
        return np.diff(self.values_half) / np.diff(self.nodes_half)

    @property
    def half_thickness(self) -> float:
        return float(self.nodes_half[-1])

    def _element_index(self, z_abs: float) -> int:
        z_abs = float(z_abs)

        if z_abs < -1.0e-12 or z_abs > self.half_thickness + 1.0e-12:
            raise ValueError("z lies outside the variational mode section.")

        if z_abs >= self.half_thickness:
            return len(self.nodes_half) - 2

        return int(
            np.searchsorted(
                self.nodes_half,
                z_abs,
                side="right",
            )
            - 1
        )

    def f(self, z: float) -> float:
        z = float(z)

        if z < 0.0:
            return -self.f(-z)

        e = self._element_index(z)

        z0 = float(self.nodes_half[e])
        value0 = float(self.values_half[e])
        slope = float(self.slopes_half[e])

        return value0 + slope * (z - z0)

    def df(self, z: float) -> float:
        e = self._element_index(abs(float(z)))
        return float(self.slopes_half[e])


def free_stress_beam_stiffness(
    C3: np.ndarray,
    *,
    axis: str = "x",
) -> np.ndarray:
    """
    Condense a 3-D engineering-shear stiffness to [eps_ss, gamma_sz].

    Permanent 3-D Voigt order:

        [xx, yy, zz, xy, xz, yz].

    All inactive beam stresses are set to zero before condensation.
    """

    C3 = np.asarray(C3, dtype=float)

    if C3.shape != (6, 6):
        raise ValueError("C3 must have shape (6, 6).")

    axis_text = str(axis).strip().lower()

    if axis_text == "x":
        active = (0, 4)
    elif axis_text == "y":
        active = (1, 5)
    else:
        raise ValueError("axis must be 'x' or 'y'.")

    inactive = tuple(
        i
        for i in range(6)
        if i not in active
    )

    Caa = C3[np.ix_(active, active)]
    Cai = C3[np.ix_(active, inactive)]
    Cia = C3[np.ix_(inactive, active)]
    Cii = C3[np.ix_(inactive, inactive)]

    reduced = Caa - Cai @ np.linalg.solve(Cii, Cia)

    return 0.5 * (reduced + reduced.T)


def symmetric_half_layers(
    layers: Sequence[BeamLayer2D],
    *,
    atol: float = 1.0e-10,
) -> tuple[BeamLayer2D, ...]:
    """
    Extract z >= 0 layers from a symmetric full-section layer stack.

    This helper is intentionally strict.  A layer crossing z=0 is split at
    the midplane.  The returned layers must cover [0, h/2] contiguously.
    """

    result = []

    for layer in layers:
        z0 = float(layer.z0)
        z1 = float(layer.z1)

        if z1 <= 0.0:
            continue

        result.append(
            BeamLayer2D(
                layer.name,
                max(0.0, z0),
                z1,
                layer.C,
            )
        )

    if not result:
        raise ValueError("No positive-half layers found.")

    result.sort(key=lambda layer: layer.z0)

    if abs(result[0].z0) > atol:
        raise ValueError("Positive-half section must begin at z=0.")

    for left, right in zip(result[:-1], result[1:]):
        if abs(left.z1 - right.z0) > atol:
            raise ValueError("Positive-half layers must be contiguous.")

    return tuple(result)


def _elements_per_layer(
    layers: Sequence[BeamLayer2D],
    elements_per_layer,
) -> tuple[int, ...]:
    if np.isscalar(elements_per_layer):
        value = int(elements_per_layer)

        if value < 1:
            raise ValueError("elements_per_layer must be >= 1.")

        return tuple(value for _ in layers)

    values = tuple(int(value) for value in elements_per_layer)

    if len(values) != len(layers):
        raise ValueError(
            "elements_per_layer sequence must match number of layers."
        )

    if any(value < 1 for value in values):
        raise ValueError("Every layer requires at least one element.")

    return values


def build_half_section_mesh(
    layers: Sequence[BeamLayer2D],
    *,
    elements_per_layer,
) -> np.ndarray:
    """
    Build a conforming 1-D mesh over contiguous positive-half layers.
    """

    layers = tuple(layers)
    counts = _elements_per_layer(layers, elements_per_layer)

    nodes = [float(layers[0].z0)]

    for layer, count in zip(layers, counts):
        segment = np.linspace(
            float(layer.z0),
            float(layer.z1),
            int(count) + 1,
        )[1:]

        nodes.extend(float(value) for value in segment)

    return np.asarray(nodes, dtype=float)


def _layer_at(
    layers: Sequence[BeamLayer2D],
    z: float,
) -> BeamLayer2D:
    z = float(z)

    for i, layer in enumerate(layers):
        is_last = i == len(layers) - 1

        if (
            layer.z0 <= z < layer.z1
            or (is_last and layer.z0 <= z <= layer.z1)
        ):
            return layer

    raise RuntimeError("Could not identify section layer at z={}".format(z))


def assemble_variational_matrices(
    layers: Sequence[BeamLayer2D],
    nodes_half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble

        K = int G(z) f'(z) v'(z) dz
        M = int E(z) f(z)  v(z)  dz

    on the positive half-section.
    """

    layers = tuple(layers)
    nodes = np.asarray(nodes_half, dtype=float)

    n = len(nodes)

    K = np.zeros((n, n), dtype=float)
    M = np.zeros((n, n), dtype=float)

    for e in range(n - 1):
        z0 = float(nodes[e])
        z1 = float(nodes[e + 1])
        length = z1 - z0
        midpoint = 0.5 * (z0 + z1)

        layer = _layer_at(layers, midpoint)

        Eeff = float(layer.C[0, 0])
        Geff = float(layer.C[1, 1])

        Ke = (
            Geff
            / length
            * np.asarray(
                [
                    [1.0, -1.0],
                    [-1.0, 1.0],
                ],
                dtype=float,
            )
        )

        Me = (
            Eeff
            * length
            / 6.0
            * np.asarray(
                [
                    [2.0, 1.0],
                    [1.0, 2.0],
                ],
                dtype=float,
            )
        )

        indices = np.asarray([e, e + 1], dtype=int)

        K[np.ix_(indices, indices)] += Ke
        M[np.ix_(indices, indices)] += Me

    return K, M


def derive_variational_second_shear_mode(
    layers_half: Sequence[BeamLayer2D],
    f1: ScalarFunction,
    *,
    elements_per_layer,
    eigenvalue_tol: float = 1.0e-12,
) -> VariationalShearMode:
    """
    Derive the lowest-energy odd mode orthogonal to the supplied f1.
    """

    layers = tuple(layers_half)

    if not layers:
        raise ValueError("At least one positive-half layer is required.")

    nodes = build_half_section_mesh(
        layers,
        elements_per_layer=elements_per_layer,
    )

    K, M = assemble_variational_matrices(
        layers,
        nodes,
    )

    f1_values = np.asarray(
        [float(f1(z)) for z in nodes],
        dtype=float,
    )

    # Odd symmetry fixes f2(0)=0.
    free = np.arange(1, len(nodes), dtype=int)

    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    f1f = f1_values[free]

    constraint = Mf @ f1f

    Z = null_space(
        constraint.reshape(1, -1),
        rcond=1.0e-12,
    )

    if Z.shape[1] == 0:
        raise RuntimeError(
            "No admissible subspace remains after f1 orthogonality."
        )

    Kr = Z.T @ Kf @ Z
    Mr = Z.T @ Mf @ Z

    eigenvalues, eigenvectors = eigh(Kr, Mr)

    positive = np.where(
        eigenvalues > float(eigenvalue_tol)
    )[0]

    if len(positive) == 0:
        raise RuntimeError("No positive variational eigenvalue found.")

    index = int(positive[0])
    eigenvalue = float(eigenvalues[index])

    values = np.zeros(len(nodes), dtype=float)
    values[free] = Z @ eigenvectors[:, index]

    slopes = np.diff(values) / np.diff(nodes)
    scale = float(np.max(np.abs(slopes)))

    if scale <= 0.0:
        raise RuntimeError("Derived variational mode is degenerate.")

    values /= scale

    orthogonality = float(f1_values @ M @ values)
    modal_E_norm = float(values @ M @ values)
    modal_G_gradient_norm = float(values @ K @ values)

    # Deterministic sign: prefer positive first non-negligible half-section
    # slope.  Mode sign has no mechanical significance.
    slopes = np.diff(values) / np.diff(nodes)
    significant = np.where(np.abs(slopes) > 1.0e-12)[0]

    if len(significant) and slopes[int(significant[0])] < 0.0:
        values *= -1.0
        orthogonality *= -1.0

    return VariationalShearMode(
        eigenvalue=eigenvalue,
        nodes_half=nodes,
        values_half=values,
        E_orthogonality_residual=orthogonality,
        modal_E_norm=modal_E_norm,
        modal_G_gradient_norm=modal_G_gradient_norm,
    )
