# HOST ↔ 3D FE four-point-bending comparison contract

This contract defines the quantities and sign conventions used when comparing
HOST and other sandwich theories against the homogenized 3D Abaqus reference.

## Coordinate and sign convention

- Global/model `X`: beam longitudinal direction.
- Global/model `Y`: beam width direction.
- Global/model `Z`: sandwich thickness direction.
- Downward transverse displacement is `U3 < 0`.
- Positive beam bending moment is the postprocessor quantity
  `My_beam_convention_Nmm = -∫_A z S11 dA`.
- `Qz_raw_from_S13_N = ∫_A S13 dA`.
- Abaqus `E13` is engineering shear strain `gamma_xz`.

## FE reference package

A compact reference is exported to:

`reference_results/4pb/<reference-name>/`

The large ODB remains in Abaqus scratch storage.

Required files:

- `reference_manifest.json`
- `source_case_manifest.json`
- `summary.json`
- `equilibrium.csv`
- `global_response.csv`
- `face_response.csv`
- `section_resultants.csv`
- `through_thickness_profiles.csv`
- `energy_summary.csv`

## Comparison levels

### Level 1 — global response

Mandatory for every theory:

- transverse displacement `w(x)` ↔ FE width-averaged `U3(x)`;
- midspan displacement `w_mid`;
- total strain energy when the theory provides it.

Primary error measures:

- signed and absolute midspan error;
- relative midspan error;
- L2 and L∞ error of the displacement curve over the support span.

### Level 2 — generalized section resultants

Where available from the theory:

- `Nx(x)` ↔ `Nx_raw_N`;
- `Qz(x)` ↔ `Qz_raw_from_S13_N`;
- `My(x)` ↔ `My_beam_convention_Nmm`.

The FE resultants are volume-integrated over one structured x-slab and divided
by slab width, i.e. a mean cross-sectional resultant over that slab.

### Level 3 — face response

At least in the constant-moment region:

- top/bottom `epsilon_xx`;
- top/bottom `sigma_xx`.

For 3D solid FE references, outer-surface face quantities are evaluated at the
same physical outer-surface z coordinate by fitting the integration-point
profile through the face sheet and evaluating at `z_top` / `z_bottom`.

### Level 4 — through-thickness discrimination

At the standardized stations:

- support left/right;
- shear-span midpoints;
- load lines;
- midspan.

Compare, as supported by each theory:

- `u_x(z)`;
- `epsilon_xx(z)`;
- `gamma_xz(z)`;
- `epsilon_zz(z)`;
- `sigma_xx(z)`;
- `tau_xz(z)`;
- `sigma_zz(z)`.

These quantities are the principal discriminators between CST/FSDT/TSDT,
layerwise/zig-zag theories, and HOST.

## Recommended FE discretizations

For the current Gyroid RD12 development benchmark:

- high-fidelity reference:
  `C3D20R`, core/in-plane `ne=4`, face through-thickness `ne_z=4`;
- economical campaign mesh:
  `C3D20R`, core/in-plane `ne=2`, face through-thickness `ne_z=2`.

The C3D8I family remains useful as an independent formulation cross-check, not
as the primary high-fidelity reference.

The discretization choice must remain parameterized and should be revisited
for substantially different geometry/slenderness regimes.
