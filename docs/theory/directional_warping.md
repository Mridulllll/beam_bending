# Direction-dependent HOST warping

## Scope

The direction-dependent higher-order sandwich theory extends the
historical scalar HOST warping description by introducing independent
through-thickness warping functions in the two transverse-shear
directions,

\[
f_x(z), \qquad f_y(z).
\]

The extension is intended for sandwich plates with homogenized
anisotropic cores for which the effective transverse-shear moduli

\[
G_{xz}
\quad\text{and}\quad
G_{yz}
\]

are not equal.

The historical scalar theory remains available and unchanged.

---

## Kinematics

The directional HOST displacement field is

\[
u(x,y,z)
=
u_0
-
z\,w_{0,x}
+
f_x(z)\,\beta_x,
\]

\[
v(x,y,z)
=
v_0
-
z\,w_{0,y}
+
f_y(z)\,\beta_y,
\]

\[
w(x,y,z)
=
w_0
+
g(z)\,\eta.
\]

The associated engineering strains are

\[
\varepsilon_{xx}
=
u_{0,x}
-
z\,w_{0,xx}
+
f_x\,\beta_{x,x},
\]

\[
\varepsilon_{yy}
=
v_{0,y}
-
z\,w_{0,yy}
+
f_y\,\beta_{y,y},
\]

\[
\varepsilon_{zz}
=
g'\eta,
\]

\[
\gamma_{xy}
=
u_{0,y}
+
v_{0,x}
-
2z\,w_{0,xy}
+
f_x\,\beta_{x,y}
+
f_y\,\beta_{y,x},
\]

\[
\gamma_{xz}
=
f_x'\beta_x
+
g\,\eta_{,x},
\]

\[
\gamma_{yz}
=
f_y'\beta_y
+
g\,\eta_{,y}.
\]

---

## Directional generalized strain representation

When

\[
f_x \neq f_y,
\]

the two twisting-warping terms

\[
\beta_{x,y}
\quad\text{and}\quad
\beta_{y,x}
\]

cannot be represented by the single scalar generalized strain used by
the historical 14-component theory.

The directional generalized strain vector therefore contains
15 components,

\[
\mathbf E_{\mathrm{dir}}
=
[
\varepsilon^0_{xx},
\varepsilon^0_{yy},
\gamma^0_{xy},
\kappa_{xx},
\kappa_{yy},
\kappa_{xy},
\chi_{xx},
\chi_{yy},
\chi^{x}_{xy},
\chi^{y}_{xy},
\beta_x,
\beta_y,
\eta_{,x},
\eta_{,y},
\eta
]^T.
\]

The directional thickness operator has dimension

\[
6\times15,
\]

and through-thickness constitutive integration produces

\[
\mathbf K_{15}.
\]

For the five-field HSDT formulation with the quasi-3D variable
suppressed, the first 12 generalized components are retained,

\[
\mathbf K_{12}.
\]

The historical scalar formulation remains

\[
\mathbf K_{14}
\rightarrow
\mathbf K_{11}.
\]

There is deliberately no artificial universal mapping between
arbitrary 14-component and 15-component generalized strain vectors.
Scalar reduction is instead verified at the physical-strain, energy,
modal-stiffness, solution, and buckling levels.

---

## Material-informed directional warping

For a symmetric face/core/face sandwich, the material-informed
directional construction uses separate transverse-shear moduli,

\[
G_{xz}(z),
\qquad
G_{yz}(z).
\]

Using the target shear-stress shape

\[
p(z)
=
1-\frac{4z^2}{h^2},
\]

the directional derivatives are constructed independently according to

\[
f_x'(z)
\propto
\frac{G_{xz,\mathrm{core}}}
     {G_{xz}(z)}
p(z),
\]

\[
f_y'(z)
\propto
\frac{G_{yz,\mathrm{core}}}
     {G_{yz}(z)}
p(z).
\]

The piecewise-integrated functions are continuous across the
face/core interfaces and satisfy zero transverse shear at the external
surfaces.

---

## Constitutive assumption

The present material-informed directional construction assumes that
the selected material axes approximately diagonalize the transverse
shear constitutive block,

\[
\mathbf G_s
=
\begin{bmatrix}
C_{55} & C_{56}\\
C_{65} & C_{66}
\end{bmatrix}
\approx
\begin{bmatrix}
G_{xz} & 0\\
0 & G_{yz}
\end{bmatrix}.
\]

The full reduced constitutive tensor is nevertheless retained during
generalized-stiffness integration. Therefore small residual
constitutive couplings are not deleted from the plate model; they are
only neglected when deriving the independent material-informed
warping functions.

If the off-diagonal transverse-shear coupling becomes appreciable, a
fully general formulation would require vector- or matrix-valued
warping rather than independent scalar functions \(f_x\) and \(f_y\).

For the investigated Gyroid RD12 benchmark states, the normalized
transverse-shear coupling

\[
\frac{|C_{56}|}
{\sqrt{C_{55}C_{66}}}
\]

was approximately 0.22% for the near-isotropic initial state and
0.47% for the anisotropic optimized state. These tensors are therefore
well suited to the present directional formulation.

---

## Navier plate response

The directional five-field system uses modal unknowns

\[
\mathbf q
=
[U,V,W,X,Y]^T.
\]

Because the directional generalized strains contain separate
\(\beta_{x,y}\) and \(\beta_{y,x}\) terms, basis-aware Navier operators
are used.

For a given \((m,n)\) mode the directional modal stiffness is

\[
\mathbf K_{mn}^{\mathrm{dir}}
=
\frac{ab}{4}
\sum_r
\mathbf B_r^T
\mathbf K_{12}
\mathbf B_r,
\]

where the sum runs over the compatible sine/cosine basis families.

In the scalar limit

\[
f_x=f_y,
\]

the directional modal stiffness and modal solution reproduce the
historical scalar formulation.

---

## Linear buckling

For uniform normal membrane prestress, the elastic modal system is
condensed to the transverse amplitude \(W\).

For a single Navier mode,

\[
\lambda_{\mathrm{cr}}
=
\frac{k_{W,\mathrm{eff}}}
{
(ab/4)
\left(
N_x^{\mathrm{ref}}\alpha^2
+
N_y^{\mathrm{ref}}\beta^2
\right)
}.
\]

Directional HOST therefore modifies the elastic modal stiffness while
the classical geometric-prestress contribution remains unchanged.

Pure shear prestress \(N_{xy}\) couples different Navier modes and is
solved with the multimode generalized eigenvalue problem

\[
\mathbf K_E \mathbf v
=
\lambda
\mathbf K_G \mathbf v.
\]

The directional multimode formulation uses directional elastic modal
stiffnesses while retaining the same geometric coupling matrix.

---

## Combined-load interaction

Directional HOST supports proportional interaction paths including

\[
N_x-N_{xy}
\]

and

\[
N_x-N_y.
\]

The calculated interaction curves demonstrate that the correction from
Reddy HSDT to directional HOST is not a constant scale factor.

For the investigated Gyroid RD12 states, the directional/Reddy
correction was stronger for shear-dominated buckling than for pure
compression and varied continuously with load mixture.

---

## Verification hierarchy

The implementation is verified through the following sequence:

1. symbolic directional kinematics;
2. exact physical-strain reconstruction;
3. scalar-limit physical strain equivalence;
4. directional generalized stiffness and strain energy;
5. material-informed interface and surface conditions;
6. scalar-limit Navier modal stiffness;
7. scalar-limit static plate response;
8. scalar-limit single-mode normal buckling;
9. scalar-limit multimode pure-shear buckling;
10. scalar-limit combined-load buckling interaction;
11. anisotropic homogenized-TPMS benchmarks.

The scalar limit

\[
f_x=f_y
\]

recovers the historical HOST formulation throughout this hierarchy.

---

## Representative Gyroid benchmark

For a 100 mm x 60 mm sandwich plate with the anisotropic optimized
Gyroid RD12 core,

\[
G_{xz}=727.43\ \mathrm{MPa},
\qquad
G_{yz}=967.71\ \mathrm{MPa}.
\]

The converged normal-buckling results were

\[
N_{x,\mathrm{cr}}^{\mathrm{Reddy}}
=
9268.55\ \mathrm{N/mm},
\]

\[
N_{x,\mathrm{cr}}^{\mathrm{dir}}
=
8954.85\ \mathrm{N/mm},
\]

with governing mode

\[
(m,n)=(4,1),
\]

while

\[
N_{y,\mathrm{cr}}^{\mathrm{Reddy}}
=
6741.53\ \mathrm{N/mm},
\]

\[
N_{y,\mathrm{cr}}^{\mathrm{dir}}
=
6630.92\ \mathrm{N/mm},
\]

with governing mode

\[
(m,n)=(1,1).
\]

The corresponding directional/Reddy ratios are approximately

\[
0.9662
\quad\text{and}\quad
0.9836,
\]

demonstrating a mode- and direction-dependent higher-order shear
correction.

The governing modes and critical loads were unchanged when the
single-mode search range was increased from 8 x 8 to 24 x 24.