# HOST â€” Higher-Order Sandwich Theory

HOST is a research-oriented analytical framework for sandwich structures with homogenized surface-based lattice and TPMS cores.

The current implementation focuses on higher-order plate kinematics, homogenization-informed sandwich warping, Navier solutions for bending, and linear buckling under proportional in-plane loading.

## Scope

The framework is intended for sandwich structures with:

- isotropic or anisotropic facesheets,
- homogenized lattice or TPMS cores,
- full 6x6 constitutive stiffness tensors,
- strong face/core stiffness mismatch,
- transverse shear deformation,
- static bending,
- linear buckling,
- multimode shear buckling,
- combined-load buckling interaction.

The TPMS core is treated as a homogenized continuum at the sandwich scale.

## Theory hierarchy

```text
TPMS / lattice unit-cell homogenization
              |
              v
        C_hom (6 x 6)
              |
              v
sandwich constitutive integration
              |
              v
higher-order sandwich theory
              |
       +------+------+
       |             |
       v             v
 static bending    buckling
                     |
             +-------+--------+
             |                |
             v                v
      normal compression   multimode shear
                              |
                              v
                    combined-load interaction
```

## Higher-order kinematics

The generalized displacement field is

```text
u(x,y,z) = u0(x,y) - z*w0,x(x,y) + f(z)*beta_x(x,y)

v(x,y,z) = v0(x,y) - z*w0,y(x,y) + f(z)*beta_y(x,y)

w(x,y,z) = w0(x,y) + g(z)*eta(x,y)
```

The generalized midsurface variables are

```text
q = [u0, v0, w0, beta_x, beta_y, eta]
```

The formulation can represent:

- membrane deformation,
- bending,
- higher-order transverse shear deformation,
- transverse-normal deformation in the general formulation.

The current five-field HSDT baseline suppresses the transverse-normal degree of freedom.

## Constitutive convention

HOST uses the permanent Voigt ordering

```text
[xx, yy, zz, xy, xz, yz]
```

with engineering shear strains.

A complete homogenized 6x6 stiffness tensor can therefore be supplied directly by a periodic TPMS unit-cell homogenization analysis.

The full homogenized tensor is retained in the sandwich constitutive integration.

## Homogenized TPMS cores

At the structural scale, the explicit TPMS geometry is replaced by its homogenized constitutive response:

```text
TPMS unit cell
      |
      v
periodic homogenization
      |
      v
C_hom
      |
      v
equivalent continuum core
      |
      v
HOST sandwich model
```

This separates the unit-cell length scale from the plate/sandwich length scale.

The current implementation also extracts the homogenized transverse shear moduli

```text
G_xz
G_yz
```

for use in the material-informed warping model.

Strong transverse shear anisotropy is detected rather than silently averaged.

## Reddy HSDT reference

A global Reddy-type higher-order shear deformation theory is retained as a reference formulation.

It provides:

- an established higher-order plate baseline,
- zero transverse shear stress at the outer surfaces,
- a useful reference for verification and limiting cases.

For strongly heterogeneous sandwiches, however, a global through-thickness polynomial does not account explicitly for the large stiffness contrast between facesheets and core.

## Material-informed sandwich warping

HOST therefore also includes a sandwich-specific material-informed warping formulation.

The physical motivation is

```text
gamma_xz(z) ~ tau_xz(z) / G_xz(z)

gamma_yz(z) ~ tau_yz(z) / G_yz(z)
```

so that transverse shear deformation can localize in a compliant core while shear traction remains continuous across material interfaces.

The implemented warping satisfies:

- odd symmetry,
- zero shear at the external surfaces,
- continuity of the displacement warping field,
- continuity of transverse shear traction across face/core interfaces.

For sufficiently similar `G_xz` and `G_yz`, a common scalar warping function is currently used.

Direction-dependent warping functions are a planned extension for strongly anisotropic cores.

## Generalized stiffness

Through-thickness integration produces the generalized sandwich stiffness matrix from the layerwise constitutive tensors and kinematic operators.

The implementation supports:

- homogeneous layers,
- heterogeneous face/core systems,
- asymmetric sandwiches,
- through-thickness stiffness functions,
- graded cores,
- full anisotropic 6x6 constitutive tensors.

The generalized stiffness is decomposed and tested for membrane, bending, warping, shear, and coupling contributions.

## Static plate response

Simply supported rectangular plates are solved using Navier expansions.

The implementation includes:

- uniform transverse pressure,
- modal stiffness assembly,
- modal static condensation,
- center-deflection calculation,
- energy decomposition.

Reference formulations include:

- Kirchhoff thin-plate theory,
- energy-equivalent FSDT sandwich theory,
- global Reddy HSDT,
- material-informed HOST.

## Kirchhoff verification

The HSDT implementation has been verified against the Kirchhoff thin-plate limit.

As plate slenderness increases,

```text
w_HSDT / w_Kirchhoff -> 1
```

and the numerical error decreases monotonically.

The corresponding convergence data and figures are stored under:

```text
verification/hsdt_kirchhoff/
```

## FSDT sandwich reference

An independent energy-equivalent FSDT reference is included.

For a heterogeneous sandwich, the transverse shear stiffness is evaluated through shear compliance rather than the arithmetic integral of layer shear moduli.

This captures the concentration of transverse shear deformation in a compliant core.

The reference was used to verify the material-informed HOST formulation over large face/core stiffness contrasts.

## Linear buckling

HOST includes global linear buckling of simply supported sandwich plates under:

- uniaxial x-compression,
- uniaxial y-compression,
- equal biaxial compression.

For normal membrane resultants, individual Navier modes remain uncoupled.

The critical load is determined from the condensed transverse modal stiffness and the geometric stiffness contribution.

The thin-plate implementation has been verified against the classical Kirchhoff buckling limit.

## Multimode shear buckling

In-plane shear prestress introduces coupling between different Navier modes.

HOST therefore solves a multimode generalized eigenvalue problem:

```text
(K_E - lambda*K_G) W = 0
```

where:

```text
K_E = elastic modal stiffness matrix
K_G = geometric stiffness matrix
W   = vector of transverse Navier amplitudes
```

The non-transverse variables are statically condensed before assembling the multimode problem.

## Modal truncation

The transverse displacement is represented by

```text
w(x,y) =
sum_m sum_n
W_mn
sin(m*pi*x/a)
sin(n*pi*y/b)
```

A `16 x 16` modal basis means

```text
m = 1, ..., 16
n = 1, ..., 16
```

and therefore contains 256 transverse Navier basis functions.

This is an analytical modal truncation and is independent of the number of TPMS unit cells.

A convergence study is included under:

```text
verification/multimode_buckling/
```

For the investigated Gyroid RD12 sandwich, the change from a 12 x 12 to a 16 x 16 basis was below 0.1 percent.

## Combined-load buckling

The multimode formulation supports proportional membrane loading

```text
(Nx, Ny, Nxy) =
lambda * (Nx_ref, Ny_ref, Nxy_ref)
```

This allows buckling interaction curves such as

```text
Nx - Ny
Nx - Nxy
```

to be generated.

Current TPMS interaction examples are stored under:

```text
verification/tpms_buckling_interaction/
```

## Repository structure

```text
HOST/
|
+-- src/
|   +-- host/
|       Analytical HOST library
|
+-- tests/
|       Unit, regression and verification tests
|
+-- symbolic/
|       Symbolic derivations
|
+-- examples/
|       Executable studies and benchmark scripts
|
+-- verification/
|       Numerical verification data and generated figures
|
+-- pyproject.toml
+-- README.md
+-- .gitignore
```

## Main library modules

```text
src/host/kinematics.py
    generalized displacement field and strain derivation

src/host/generalized_strains.py
    generalized strain operators

src/host/constitutive.py
    constitutive matrices and reductions

src/host/generalized_stiffness.py
    through-thickness generalized stiffness integration

src/host/stiffness_blocks.py
    stiffness decomposition and coupling classification

src/host/warping.py
    Reddy reference warping

src/host/sandwich_warping.py
    material-informed sandwich warping

src/host/sandwich.py
    three-layer sandwich construction

src/host/navier.py
    Navier plate solution

src/host/kirchhoff.py
    classical thin-plate reference

src/host/fsdt_reference.py
    energy-equivalent FSDT sandwich reference

src/host/buckling.py
    normal and biaxial linear buckling

src/host/multimode_buckling.py
    multimode shear and combined-load buckling

src/host/buckling_interaction.py
    proportional buckling interaction curves

src/host/tpms_core.py
    homogenized TPMS core handling

src/host/tpms_io.py
    homogenization JSON input
```

## Installation

Python 3.10 is currently used for development.

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install HOST in editable development mode:

```powershell
python -m pip install -e ".[dev]"
```

## Tests

Run the complete verification and regression suite with:

```powershell
python -m pytest -v -p no:cacheprovider
```

The test suite covers:

- symbolic kinematics,
- generalized strain reconstruction,
- constitutive symmetry and positive definiteness,
- plane-stress and sigma_zz reductions,
- generalized stiffness integration,
- symmetric and asymmetric sandwiches,
- stiffness coupling blocks,
- higher-order warping,
- sandwich-specific warping,
- field equations,
- natural boundary operators,
- energy consistency,
- Navier modal operators,
- Kirchhoff convergence,
- FSDT reference behavior,
- TPMS constitutive handling,
- normal buckling,
- multimode shear buckling,
- modal convergence,
- combined-load interaction.

## Example calculations

### HSDT / Kirchhoff convergence

```powershell
python .\examples\verify_hsdt_kirchhoff_convergence.py
```

### Sandwich core stiffness sweep

```powershell
python .\examples\three_layer_sandwich_core_stiffness_sweep.py
```

### Reddy versus material-informed warping

```powershell
python .\examples\compare_reddy_material_informed_warping.py
```

### FSDT / Reddy / HOST comparison

```powershell
python .\examples\compare_fsdt_reddy_material_informed.py
```

### Run a homogenized TPMS sandwich

```powershell
python .\examples\run_tpms_sandwich_from_json.py `
    "path\to\homogenized_tpms.json"
```

### Run TPMS buckling

```powershell
python .\examples\run_tpms_buckling_from_json.py `
    "path\to\homogenized_tpms.json"
```

### Multimode shear convergence

```powershell
python .\examples\verify_multimode_shear_convergence.py
```

### TPMS buckling interaction

```powershell
python .\examples\plot_tpms_buckling_interaction.py
```

## Current limitations

The current implementation is a research framework rather than a general-purpose production plate solver.

Current limitations include:

- simply supported Navier-type rectangular plates,
- linear elastic material behavior,
- linearized global buckling,
- no nonlinear postbuckling,
- no plasticity or damage,
- no local facesheet wrinkling model,
- no explicit local TPMS-cell instability model,
- no general finite-element implementation of the current HOST formulation,
- scalar material-informed warping currently assumes sufficiently similar transverse shear moduli,
- scale separation between the TPMS cell and structural dimensions is assumed.

## Planned development

Planned extensions include:

- direction-dependent material-informed warping,
- improved treatment of anisotropic homogenized cores,
- additional boundary conditions,
- systematic finite-element validation,
- local facesheet and core instability criteria,
- nonlinear and postbuckling extensions,
- graded TPMS core constitutive fields,
- topology and relative-density parametric studies.

## Research status

HOST is under active research development.

Its purpose is to provide an analytically transparent framework for studying additively manufactured sandwich structures with homogenized surface-based lattice cores, particularly sheet-based TPMS architectures.

The code and verification results should currently be treated as research software under active validation.

## Direction-dependent HOST formulation

HOST includes a direction-dependent higher-order formulation for
homogenized sandwich cores with unequal transverse-shear stiffnesses
\(G_{xz}\) and \(G_{yz}\).

The formulation uses independent warping functions \(f_x(z)\) and
\(f_y(z)\), a 15-component generalized strain representation, and a
12-component five-field directional HSDT stiffness.

Supported analyses include:

- static Navier plate response;
- uniaxial and biaxial linear buckling;
- multimode pure-shear buckling;
- combined \(N_x-N_{xy}\) and \(N_x-N_y\) buckling interaction.

The historical scalar HOST formulation remains available and is
recovered in the limit \(f_x=f_y\).

The material-informed directional construction assumes approximately
principal transverse-shear material axes. Full constitutive coupling
is retained in stiffness integration; only the warping-function
construction neglects small off-diagonal transverse-shear coupling.

See
[`docs/theory/directional_warping.md`](docs/theory/directional_warping.md)
for the formulation, assumptions, and verification hierarchy.