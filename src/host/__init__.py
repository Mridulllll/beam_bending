"""
HOST â€” Higher-Order Sandwich Theory.

Research-oriented analytical framework for sandwich structures with
homogenized lattice and TPMS cores.
"""

from host.buckling import (
    critical_equal_biaxial_load,
    critical_uniaxial_load_x,
    find_critical_buckling_mode,
)

from host.buckling_interaction import (
    nx_nxy_interaction,
    nx_ny_interaction,
    proportional_buckling_point,
)

from host.constitutive import (
    isotropic_stiffness_3d,
    sigma_zz_zero_stiffness,
)

from host.sandwich import (
    build_three_layer_sandwich,
)

from host.sandwich_warping import (
    material_informed_generalized_warping,
)

from host.tpms_core import (
    build_homogenized_tpms_core,
    representative_transverse_shear_modulus,
)

from host.tpms_io import (
    load_tpms_homogenization_json,
)


__all__ = [
    "build_homogenized_tpms_core",
    "build_three_layer_sandwich",
    "critical_equal_biaxial_load",
    "critical_uniaxial_load_x",
    "find_critical_buckling_mode",
    "isotropic_stiffness_3d",
    "load_tpms_homogenization_json",
    "material_informed_generalized_warping",
    "nx_nxy_interaction",
    "nx_ny_interaction",
    "proportional_buckling_point",
    "representative_transverse_shear_modulus",
    "sigma_zz_zero_stiffness",
]


__version__ = "0.1.0"

# ------------------------------------------------------------------
# Direction-dependent HOST public API
# ------------------------------------------------------------------

from host.sandwich_warping import (
    material_informed_directional_generalized_warping,
)

from host.buckling import (
    directional_critical_equal_biaxial_load,
    directional_critical_uniaxial_load_x,
    directional_critical_uniaxial_load_y,
    find_directional_critical_buckling_mode,
)

from host.multimode_buckling import (
    solve_directional_multimode_buckling,
)

from host.buckling_interaction import (
    directional_nx_nxy_interaction,
    directional_nx_ny_interaction,
    directional_proportional_buckling_point,
)

_DIRECTIONAL_PUBLIC_API = (
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

if "__all__" in globals():
    __all__ = tuple(__all__) + tuple(
        name
        for name in _DIRECTIONAL_PUBLIC_API
        if name not in __all__
    )