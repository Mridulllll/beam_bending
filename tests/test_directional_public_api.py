"""
Regression test for the public direction-dependent HOST API.
"""

import host


def test_directional_public_api_is_exposed():
    expected = (
        "material_informed_directional_generalized_warping",
        "find_directional_critical_buckling_mode",
        "directional_critical_uniaxial_load_x",
        "directional_critical_uniaxial_load_y",
        "directional_critical_equal_biaxial_load",
        "solve_directional_multimode_buckling",
        "directional_proportional_buckling_point",
        "directional_nx_nxy_interaction",
        "directional_nx_ny_interaction",
    )

    for name in expected:
        assert hasattr(
            host,
            name,
        ), name

        assert callable(
            getattr(
                host,
                name,
            )
        ), name