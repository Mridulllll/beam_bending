"""
Cell-count parameterization for periodic lattice sandwich beams.

The primary specimen variables are integer unit-cell counts rather
than independent dimensional beam quantities.

For a four-point-bending specimen:

    lc  = unit-cell edge length

    Nh  = cells through core height
    Nb  = cells across specimen width
    Na  = cells in each shear span
    Nm  = cells in the constant-moment region
    No  = overhang cells outside each support

Derived dimensions:

    hc = Nh * lc
    b  = Nb * lc

    a  = Na * lc
    SL = Nm * lc

    S  = (2*Na + Nm) * lc
    L  = (2*No + 2*Na + Nm) * lc

The number of face-affected/sacrificial cells can be tracked
explicitly in the height direction without hard-coding a universal
homogenization threshold into the beam theory itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from host.beam import (
    SandwichBeamGeometry,
)


def _positive_integer(
    value,
    *,
    name: str,
    allow_zero: bool = False,
) -> int:
    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        Integral,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    value = int(
        value
    )

    lower = (
        0
        if allow_zero
        else 1
    )

    if value < lower:
        comparison = (
            "non-negative"
            if allow_zero
            else "positive"
        )

        raise ValueError(
            f"{name} must be {comparison}."
        )

    return value


@dataclass(frozen=True)
class CellCountBeamDesign:
    """
    Four-point-bending sandwich specimen described by periodic
    unit-cell counts.
    """

    cell_size: float

    n_height: int
    n_width: int

    n_shear: int
    n_moment: int
    n_overhang: int

    face_bottom: float
    face_top: float

    face_boundary_cells: int = 1

    def __post_init__(
        self,
    ):
        cell_size = float(
            self.cell_size
        )

        face_bottom = float(
            self.face_bottom
        )

        face_top = float(
            self.face_top
        )

        if cell_size <= 0.0:
            raise ValueError(
                "cell_size must be positive."
            )

        if face_bottom <= 0.0:
            raise ValueError(
                "face_bottom must be positive."
            )

        if face_top <= 0.0:
            raise ValueError(
                "face_top must be positive."
            )

        object.__setattr__(
            self,
            "cell_size",
            cell_size,
        )

        object.__setattr__(
            self,
            "face_bottom",
            face_bottom,
        )

        object.__setattr__(
            self,
            "face_top",
            face_top,
        )

        for name in (
            "n_height",
            "n_width",
            "n_shear",
            "n_moment",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(
                    getattr(
                        self,
                        name,
                    ),
                    name=name,
                ),
            )

        object.__setattr__(
            self,
            "n_overhang",
            _positive_integer(
                self.n_overhang,
                name="n_overhang",
                allow_zero=True,
            ),
        )

        object.__setattr__(
            self,
            "face_boundary_cells",
            _positive_integer(
                self.face_boundary_cells,
                name="face_boundary_cells",
                allow_zero=True,
            ),
        )

    @property
    def core_height(
        self,
    ) -> float:
        return (
            self.n_height
            * self.cell_size
        )

    @property
    def width(
        self,
    ) -> float:
        return (
            self.n_width
            * self.cell_size
        )

    @property
    def shear_span(
        self,
    ) -> float:
        return (
            self.n_shear
            * self.cell_size
        )

    @property
    def load_span(
        self,
    ) -> float:
        return (
            self.n_moment
            * self.cell_size
        )

    @property
    def support_span(
        self,
    ) -> float:
        return (
            (
                2 * self.n_shear
                + self.n_moment
            )
            * self.cell_size
        )

    @property
    def overhang(
        self,
    ) -> float:
        return (
            self.n_overhang
            * self.cell_size
        )

    @property
    def length(
        self,
    ) -> float:
        return (
            (
                2 * self.n_overhang
                + 2 * self.n_shear
                + self.n_moment
            )
            * self.cell_size
        )

    @property
    def total_height(
        self,
    ) -> float:
        return (
            self.core_height
            + self.face_bottom
            + self.face_top
        )

    @property
    def support_left(
        self,
    ) -> float:
        return self.overhang

    @property
    def support_right(
        self,
    ) -> float:
        return (
            self.length
            - self.overhang
        )

    @property
    def load_left(
        self,
    ) -> float:
        return (
            self.overhang
            + self.shear_span
        )

    @property
    def load_right(
        self,
    ) -> float:
        return (
            self.load_left
            + self.load_span
        )

    @property
    def height_interior_cells(
        self,
    ) -> int:
        """
        Nominal interior cells after removing the chosen number of
        face-affected cells from the top and bottom of the core.
        """

        return max(
            0,
            self.n_height
            - 2 * self.face_boundary_cells,
        )

    def height_homogenization_ok(
        self,
        *,
        minimum_interior_cells: int = 4,
    ) -> bool:
        minimum = _positive_integer(
            minimum_interior_cells,
            name="minimum_interior_cells",
        )

        return (
            self.height_interior_cells
            >= minimum
        )

    @property
    def S_over_h(
        self,
    ) -> float:
        return (
            self.support_span
            / self.total_height
        )

    @property
    def a_over_h(
        self,
    ) -> float:
        return (
            self.shear_span
            / self.total_height
        )

    @property
    def load_span_over_h(
        self,
    ) -> float:
        return (
            self.load_span
            / self.total_height
        )

    def beam_geometry(
        self,
    ) -> SandwichBeamGeometry:
        return SandwichBeamGeometry(
            length=self.length,
            width=self.width,
            core_height=self.core_height,
            face_bottom=self.face_bottom,
            face_top=self.face_top,
        )