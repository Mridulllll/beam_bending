"""
Equilibrium/energy-derived primary shear warping for HOST beams.

This module implements the layered-beam shear construction associated with
the equilibrium/energy approach used by Gordaninejad and Bert for thick
sandwich beams, generalized here to the free-stress-condensed HOST beam
constitutive matrices.

For a layered beam section with physical constitutive order

    [sigma_ss, tau_sz]^T = Cb(z) [eps_ss, gamma_sz]^T,

the primary mode is derived from the diagonal section properties

    E(z) = Cb[0,0],
    G(z) = Cb[1,1].

With

    A   = int_A E dA,
    B   = int_A z E dA,
    D   = int_A z^2 E dA,
    z_n = B/A,
    D_R = D - B^2/A,

the N=0 bending stress is

    sigma_ss = E(z) (z-z_n) kappa,

and beam equilibrium M' = V gives the unit-physical-shear stress field

    tau_hat(z)
      = -1/D_R * int_{z_b}^{z} E(s) (s-z_n) ds.

The associated shear strain is

    gamma_hat(z) = tau_hat(z) / G(z).

The normalized primary HOST warping is defined by

    f1'(z) = gamma_hat(z) / max_z |gamma_hat(z)|,
    f1(0)  = 0.

For piecewise-constant layer properties, tau_hat and gamma_hat are piecewise
quadratic and f1 is piecewise cubic.  All section integrals in this module are
evaluated analytically; no numerical quadrature is used.

The energy-equivalent physical shear stiffness is

    1/S_eq = int_A tau_hat(z)^2 / G(z) dA,

and the corresponding Timoshenko-style correction factor is

    K_eq = S_eq / int_A G(z) dA.

For a homogeneous rectangular section this recovers K_eq = 5/6 exactly.

The off-diagonal Cb[0,1] normal/shear coupling is intentionally not used to
derive the primary equilibrium mode.  The complete 2x2 Cb is retained later
when HOST section stiffness matrices are assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from host.variational_shear_modes import BeamLayer2D


def _sorted_contiguous_layers(
    layers: Sequence[BeamLayer2D],
    *,
    atol: float = 1.0e-10,
) -> tuple[BeamLayer2D, ...]:
    result = tuple(
        sorted(
            layers,
            key=lambda layer: float(layer.z0),
        )
    )

    if not result:
        raise ValueError("At least one beam layer is required.")

    for left, right in zip(result[:-1], result[1:]):
        if abs(float(left.z1) - float(right.z0)) > float(atol):
            raise ValueError("Beam layers must be contiguous.")

    return result


def _q_polynomial_coefficients(
    layer: BeamLayer2D,
    *,
    q_at_layer_start: float,
    neutral_axis: float,
) -> tuple[float, float, float]:
    """
    Return a,b,c such that

        q(z) = a z^2 + b z + c

    within one layer, where

        q(z) = int_{z_bottom}^{z} E(s) (s-z_n) ds.
    """
    E = float(layer.C[0, 0])
    z0 = float(layer.z0)
    zn = float(neutral_axis)

    a = 0.5 * E
    b = -E * zn
    c = (
        float(q_at_layer_start)
        - 0.5 * E * z0 * z0
        + E * zn * z0
    )

    return a, b, c


def _integral_q(
    a: float,
    b: float,
    c: float,
    z0: float,
    z1: float,
) -> float:
    return float(
        (a / 3.0) * (z1 ** 3 - z0 ** 3)
        + (b / 2.0) * (z1 ** 2 - z0 ** 2)
        + c * (z1 - z0)
    )


def _integral_q_squared(
    a: float,
    b: float,
    c: float,
    z0: float,
    z1: float,
) -> float:
    return float(
        (a * a / 5.0) * (z1 ** 5 - z0 ** 5)
        + (a * b / 2.0) * (z1 ** 4 - z0 ** 4)
        + ((b * b + 2.0 * a * c) / 3.0)
        * (z1 ** 3 - z0 ** 3)
        + (b * c) * (z1 ** 2 - z0 ** 2)
        + (c * c) * (z1 - z0)
    )


@dataclass(frozen=True)
class EquilibriumShearMode:
    """
    Exact piecewise-polynomial primary shear mode for a layered beam.
    """

    width: float
    layers: tuple[BeamLayer2D, ...]
    A: float
    B: float
    D: float
    neutral_axis: float
    reduced_bending_stiffness: float
    equivalent_shear_stiffness: float
    raw_shear_stiffness: float
    shear_correction_factor: float
    normalization: float
    unit_shear_resultant_check: float
    _q_starts: np.ndarray
    _gamma_primitive_starts: np.ndarray

    def __post_init__(self) -> None:
        if float(self.width) <= 0.0:
            raise ValueError("width must be positive.")

        if float(self.reduced_bending_stiffness) <= 0.0:
            raise ValueError("reduced_bending_stiffness must be positive.")

        if float(self.equivalent_shear_stiffness) <= 0.0:
            raise ValueError("equivalent_shear_stiffness must be positive.")

        if float(self.normalization) <= 0.0:
            raise ValueError("normalization must be positive.")

        object.__setattr__(
            self,
            "_q_starts",
            np.asarray(
                self._q_starts,
                dtype=float,
            ),
        )

        object.__setattr__(
            self,
            "_gamma_primitive_starts",
            np.asarray(
                self._gamma_primitive_starts,
                dtype=float,
            ),
        )

    @property
    def z_bottom(self) -> float:
        return float(self.layers[0].z0)

    @property
    def z_top(self) -> float:
        return float(self.layers[-1].z1)

    def _layer_index(self, z: float) -> int:
        z = float(z)

        if z < self.z_bottom - 1.0e-12 or z > self.z_top + 1.0e-12:
            raise ValueError("z lies outside the beam section.")

        if z >= self.z_top:
            return len(self.layers) - 1

        for index, layer in enumerate(self.layers):
            if float(layer.z0) <= z < float(layer.z1):
                return index

        raise RuntimeError(
            "Could not identify layer at z={:.12g}.".format(z)
        )

    def _q_polynomial(
        self,
        index: int,
    ) -> tuple[float, float, float]:
        return _q_polynomial_coefficients(
            self.layers[index],
            q_at_layer_start=float(
                self._q_starts[index]
            ),
            neutral_axis=float(
                self.neutral_axis
            ),
        )

    def first_moment_below(self, z: float) -> float:
        index = self._layer_index(z)
        layer = self.layers[index]
        a, b, c = self._q_polynomial(index)

        z = float(z)

        return float(
            a * z * z
            + b * z
            + c
        )

    def tau_per_unit_shear(self, z: float) -> float:
        """
        Physical tau_sz/V for unit beam shear resultant V=1.

        Units are 1/mm^2 when force is in N and geometry in mm.
        """
        return float(
            -self.first_moment_below(z)
            / self.reduced_bending_stiffness
        )

    def gamma_per_unit_shear(self, z: float) -> float:
        """
        Physical gamma_sz/V for unit beam shear resultant V=1.
        """
        index = self._layer_index(z)
        G = float(
            self.layers[index].C[1, 1]
        )

        return float(
            self.tau_per_unit_shear(z)
            / G
        )

    def df(self, z: float) -> float:
        """
        Normalized HOST primary shear-mode derivative.

        At a material interface the physical shear strain generally has two
        one-sided traces because G(z) jumps while tau_sz remains continuous.
        The exact interface value therefore follows the layer-ownership
        convention of ``_layer_index`` and must not be used as a strain
        continuity condition.
        """
        return float(
            self.gamma_per_unit_shear(z)
            / self.normalization
        )

    def _gamma_primitive_from_bottom(
        self,
        z: float,
    ) -> float:
        index = self._layer_index(z)
        layer = self.layers[index]

        a, b, c = self._q_polynomial(index)

        integral_q = _integral_q(
            a,
            b,
            c,
            float(layer.z0),
            float(z),
        )

        G = float(
            layer.C[1, 1]
        )

        return float(
            self._gamma_primitive_starts[index]
            - integral_q
            / (
                self.reduced_bending_stiffness
                * G
            )
        )

    def f(self, z: float) -> float:
        """
        Normalized warping primitive anchored by f(0)=0.
        """
        raw = (
            self._gamma_primitive_from_bottom(
                float(z)
            )
            - self._gamma_primitive_from_bottom(
                0.0
            )
        )

        return float(
            raw / self.normalization
        )


def derive_equilibrium_shear_mode(
    layers: Sequence[BeamLayer2D],
    *,
    width: float,
) -> EquilibriumShearMode:
    """
    Derive the exact equilibrium/energy primary shear mode.
    """

    layers = _sorted_contiguous_layers(
        layers
    )

    width = float(width)

    if width <= 0.0:
        raise ValueError("width must be positive.")

    A_per_width = 0.0
    B_per_width = 0.0
    D_per_width = 0.0
    raw_G_per_width = 0.0

    for layer in layers:
        z0 = float(layer.z0)
        z1 = float(layer.z1)

        E = float(layer.C[0, 0])
        G = float(layer.C[1, 1])

        A_per_width += E * (z1 - z0)
        B_per_width += 0.5 * E * (
            z1 * z1 - z0 * z0
        )
        D_per_width += (E / 3.0) * (
            z1 ** 3 - z0 ** 3
        )
        raw_G_per_width += G * (z1 - z0)

    A = width * A_per_width
    B = width * B_per_width
    D = width * D_per_width

    neutral_axis = B / A
    reduced_bending_stiffness = (
        D - B * B / A
    )

    if reduced_bending_stiffness <= 0.0:
        raise RuntimeError(
            "Reduced bending stiffness is not positive."
        )

    q_starts = []
    q_value = 0.0

    gamma_primitive_starts = []
    gamma_primitive = 0.0

    energy_compliance_per_width = 0.0
    integral_tau_per_width = 0.0

    normalization_candidates = []

    for layer in layers:
        z0 = float(layer.z0)
        z1 = float(layer.z1)

        E = float(layer.C[0, 0])
        G = float(layer.C[1, 1])

        q_starts.append(
            q_value
        )

        gamma_primitive_starts.append(
            gamma_primitive
        )

        a, b, c = _q_polynomial_coefficients(
            layer,
            q_at_layer_start=q_value,
            neutral_axis=neutral_axis,
        )

        integral_q = _integral_q(
            a,
            b,
            c,
            z0,
            z1,
        )

        integral_q_squared = _integral_q_squared(
            a,
            b,
            c,
            z0,
            z1,
        )

        energy_compliance_per_width += (
            integral_q_squared
            / (
                reduced_bending_stiffness ** 2
                * G
            )
        )

        integral_tau_per_width += (
            -integral_q
            / reduced_bending_stiffness
        )

        gamma_primitive += (
            -integral_q
            / (
                reduced_bending_stiffness
                * G
            )
        )

        q_value += E * (
            0.5 * (
                z1 * z1
                - z0 * z0
            )
            - neutral_axis * (
                z1 - z0
            )
        )

        # gamma_hat(z) is quadratic within a layer.  Its stationary point
        # occurs at z=neutral_axis because d(tau)/dz is proportional to
        # -E(z-neutral_axis).  Check both layer endpoints and the stationary
        # point when it lies inside this layer.  Endpoint values are evaluated
        # with the current layer's G so interface strain jumps are included.
        candidate_z = [
            z0,
            z1,
        ]

        if z0 <= neutral_axis <= z1:
            candidate_z.append(
                neutral_axis
            )

        for z in candidate_z:
            q = (
                a * z * z
                + b * z
                + c
            )

            gamma = (
                -q
                / (
                    reduced_bending_stiffness
                    * G
                )
            )

            normalization_candidates.append(
                abs(float(gamma))
            )

    compliance = (
        width
        * energy_compliance_per_width
    )

    if compliance <= 0.0:
        raise RuntimeError(
            "Equilibrium shear compliance is not positive."
        )

    equivalent_shear_stiffness = (
        1.0 / compliance
    )

    raw_shear_stiffness = (
        width * raw_G_per_width
    )

    shear_correction_factor = (
        equivalent_shear_stiffness
        / raw_shear_stiffness
    )

    normalization = max(
        normalization_candidates
    )

    if normalization <= 0.0:
        raise RuntimeError(
            "Equilibrium shear mode normalization is zero."
        )

    unit_shear_resultant_check = (
        width
        * integral_tau_per_width
    )

    return EquilibriumShearMode(
        width=width,
        layers=layers,
        A=A,
        B=B,
        D=D,
        neutral_axis=neutral_axis,
        reduced_bending_stiffness=reduced_bending_stiffness,
        equivalent_shear_stiffness=equivalent_shear_stiffness,
        raw_shear_stiffness=raw_shear_stiffness,
        shear_correction_factor=shear_correction_factor,
        normalization=normalization,
        unit_shear_resultant_check=unit_shear_resultant_check,
        _q_starts=np.asarray(
            q_starts,
            dtype=float,
        ),
        _gamma_primitive_starts=np.asarray(
            gamma_primitive_starts,
            dtype=float,
        ),
    )
