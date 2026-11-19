import numpy as np
import pytest

from host.beam_cell_design import (
    CellCountBeamDesign,
)


def baseline():
    return CellCountBeamDesign(
        cell_size=10.0,
        n_height=6,
        n_width=6,
        n_shear=10,
        n_moment=8,
        n_overhang=2,
        face_bottom=1.0,
        face_top=1.0,
        face_boundary_cells=1,
    )


def test_integer_cell_design_maps_exactly_to_dimensions():
    design = baseline()

    assert design.core_height == 60.0
    assert design.width == 60.0

    assert design.shear_span == 100.0
    assert design.load_span == 80.0

    assert design.support_span == 280.0
    assert design.overhang == 20.0
    assert design.length == 320.0

    assert design.total_height == 62.0

    assert design.support_left == 20.0
    assert design.load_left == 120.0
    assert design.load_right == 200.0
    assert design.support_right == 300.0


def test_six_height_cells_give_four_nominal_interior_cells():
    design = baseline()

    assert design.n_height == 6
    assert design.face_boundary_cells == 1
    assert design.height_interior_cells == 4

    assert design.height_homogenization_ok(
        minimum_interior_cells=4
    )


def test_five_height_cells_fail_one_plus_four_plus_one_screen():
    design = CellCountBeamDesign(
        cell_size=10.0,
        n_height=5,
        n_width=6,
        n_shear=8,
        n_moment=6,
        n_overhang=2,
        face_bottom=1.0,
        face_top=1.0,
        face_boundary_cells=1,
    )

    assert design.height_interior_cells == 3

    assert not design.height_homogenization_ok(
        minimum_interior_cells=4
    )


def test_cell_size_scales_all_cell_derived_dimensions():
    d10 = baseline()

    d75 = CellCountBeamDesign(
        cell_size=7.5,
        n_height=d10.n_height,
        n_width=d10.n_width,
        n_shear=d10.n_shear,
        n_moment=d10.n_moment,
        n_overhang=d10.n_overhang,
        face_bottom=1.0,
        face_top=1.0,
        face_boundary_cells=1,
    )

    scale = 0.75

    np.testing.assert_allclose(
        d75.core_height,
        scale * d10.core_height,
    )

    np.testing.assert_allclose(
        d75.width,
        scale * d10.width,
    )

    np.testing.assert_allclose(
        d75.shear_span,
        scale * d10.shear_span,
    )

    np.testing.assert_allclose(
        d75.load_span,
        scale * d10.load_span,
    )

    np.testing.assert_allclose(
        d75.support_span,
        scale * d10.support_span,
    )

    np.testing.assert_allclose(
        d75.length,
        scale * d10.length,
    )

    # Faces are intentionally fixed in millimetres here, so total
    # sandwich height does not scale exactly by 0.75.
    np.testing.assert_allclose(
        d75.total_height,
        47.0,
    )


def test_beam_geometry_conversion_preserves_dimensions():
    design = baseline()

    geometry = design.beam_geometry()

    assert geometry.length == design.length
    assert geometry.width == design.width
    assert geometry.core_height == design.core_height
    assert geometry.face_bottom == design.face_bottom
    assert geometry.face_top == design.face_top


def test_noninteger_cell_counts_are_rejected():
    with pytest.raises(
        TypeError
    ):
        CellCountBeamDesign(
            cell_size=10.0,
            n_height=6.5,
            n_width=6,
            n_shear=8,
            n_moment=6,
            n_overhang=2,
            face_bottom=1.0,
            face_top=1.0,
        )