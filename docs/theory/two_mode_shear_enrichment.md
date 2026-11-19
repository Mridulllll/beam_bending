# Equilibrium-Derived Primary Shear Mode and Variational Higher-Order Enrichment

## 1. Purpose and status

This note documents the current candidate one-dimensional higher-order sandwich-beam formulation used in HOST.

The final hierarchy is

\[
f_1(z)
=
\text{equilibrium/energy-derived primary shear-warping mode},
\]

\[
f_2(z)
=
\text{lowest independent variational higher-order mode}.
\]

The earlier material-informed primary mode,

\[
f'_{1,\mathrm{legacy}}(z)
=
\frac{G_c}{G(z)}
\left(1-\frac{4z^2}{h^2}\right),
\]

is retained only as a legacy/reference basis. It is no longer the preferred primary mode.

The present formulation is derived entirely from section geometry and homogenized constitutive properties. No finite-element displacement or stress field is used to construct either production mode.

---

## 2. Beam kinematics

For a beam coordinate \(x\) and thickness coordinate \(z\), use

\[
u(x,z)
=
u_0(x)
-
z\,w_{0,x}(x)
+
f_1(z)\beta_1(x)
+
f_2(z)\beta_2(x),
\]

\[
w(x,z)=w_0(x).
\]

The generalized longitudinal and transverse-shear strains are

\[
\varepsilon_{xx}
=
\varepsilon_0
+
z\kappa
+
f_1\chi_1
+
f_2\chi_2,
\]

\[
\gamma_{xz}
=
f_1'\beta_1
+
f_2'\beta_2,
\]

with

\[
\varepsilon_0=u_{0,x},
\qquad
\kappa=-w_{0,xx},
\qquad
\chi_1=\beta_{1,x},
\qquad
\chi_2=\beta_{2,x}.
\]

The six generalized strains are therefore

\[
\mathbf e_6
=
\begin{bmatrix}
\varepsilon_0 &
\kappa &
\chi_1 &
\chi_2 &
\beta_1 &
\beta_2
\end{bmatrix}^{T}.
\]

The formulation retains the beam-level assumption

\[
\varepsilon_{zz}=0.
\]

This is intended for global sandwich bending away from local indentation, face wrinkling, and other genuinely three-dimensional thickness-normal effects.

---

## 3. Layer constitutive reduction

The permanent three-dimensional Voigt ordering is

\[
[xx,yy,zz,xy,xz,yz].
\]

For beam bending in the \(x\)-direction, the active physical strain pair is

\[
\boldsymbol\epsilon_b
=
\begin{bmatrix}
\varepsilon_{xx}\\
\gamma_{xz}
\end{bmatrix}.
\]

All inactive physical stresses are set to zero and the full three-dimensional stiffness is condensed by a Schur complement to

\[
\begin{bmatrix}
\sigma_{xx}\\
\tau_{xz}
\end{bmatrix}
=
\mathbf C_b(z)
\begin{bmatrix}
\varepsilon_{xx}\\
\gamma_{xz}
\end{bmatrix},
\]

where

\[
\mathbf C_b
=
\begin{bmatrix}
C_{11}^{b} & C_{15}^{b}\\
C_{15}^{b} & C_{55}^{b}
\end{bmatrix}.
\]

For the equilibrium construction of the primary mode, define

\[
E_x(z)=C_{11}^{b}(z),
\qquad
G_{xz}(z)=C_{55}^{b}(z).
\]

The complete \(2\times2\) matrix \(\mathbf C_b\), including \(C_{15}^{b}\), is retained when the final HOST section stiffness matrices are assembled.

For the canonical Gyroid RD12 section,

\[
\frac{|C_{15}^{b}|}
{\sqrt{C_{11}^{b}C_{55}^{b}}}
\approx
2.23\times10^{-3},
\]

so the diagonal equilibrium construction is a controlled approximation for the mode generation.

---

## 4. Equilibrium-derived primary shear mode

### 4.1 Reduced bending stiffness

For beam width \(b\), define

\[
A
=
b\int_{-h/2}^{h/2}E_x(z)\,dz,
\]

\[
B
=
b\int_{-h/2}^{h/2}zE_x(z)\,dz,
\]

\[
D
=
b\int_{-h/2}^{h/2}z^2E_x(z)\,dz.
\]

The neutral-axis coordinate is

\[
z_n=\frac{B}{A},
\]

and the reduced bending stiffness under zero axial force is

\[
\boxed{
D_R
=
D-\frac{B^2}{A}.
}
\]

This is the scalar Schur complement associated with \(N=0\).

For a symmetric sandwich,

\[
B=0,
\qquad
z_n=0.
\]

### 4.2 Equilibrium shear stress

Under \(N=0\),

\[
\sigma_{xx}(x,z)
=
E_x(z)
\left(z-z_n\right)
\kappa(x).
\]

Since

\[
M=D_R\kappa
\]

and beam equilibrium requires

\[
M_{,x}=V,
\]

it follows that

\[
\kappa_{,x}
=
\frac{V}{D_R}.
\]

The two-dimensional equilibrium equation

\[
\sigma_{xx,x}
+
\tau_{xz,z}
=
0
\]

then gives the through-thickness shear stress associated with a physical beam shear resultant \(V\):

\[
\boxed{
\tau_{xz}(z)
=
-\frac{V}{D_R}
\int_{z_b}^{z}
E_x(\zeta)
\left(\zeta-z_n\right)
\,d\zeta .
}
\]

Equivalently, for unit physical shear,

\[
\widehat{\tau}(z)
=
\frac{\tau_{xz}(z)}{V}
=
-\frac{1}{D_R}
\int_{z_b}^{z}
E_x(\zeta)
\left(\zeta-z_n\right)
\,d\zeta.
\]

This construction satisfies

\[
\tau_{xz}(z_b)=0,
\qquad
\tau_{xz}(z_t)=0,
\]

and preserves interlaminar shear-traction continuity at perfectly bonded interfaces.

### 4.3 Energy-equivalent physical shear stiffness

The shear strain associated with the equilibrium stress field is

\[
\widehat{\gamma}(z)
=
\frac{\widehat{\tau}(z)}
{G_{xz}(z)}.
\]

The physical shear strain energy for resultant \(V\) is

\[
U_s
=
\frac12
b
\int_{z_b}^{z_t}
\frac{\tau_{xz}^2}
{G_{xz}(z)}
\,dz.
\]

Define an equivalent physical shear stiffness \(S_{\mathrm{eq}}\) by

\[
U_s
=
\frac{V^2}
{2S_{\mathrm{eq}}}.
\]

Therefore,

\[
\boxed{
\frac{1}{S_{\mathrm{eq}}}
=
b
\int_{z_b}^{z_t}
\frac{\widehat{\tau}(z)^2}
{G_{xz}(z)}
\,dz.
}
\]

A Timoshenko-style energy-equivalent correction coefficient may be defined as

\[
\boxed{
K_{\mathrm{eq}}
=
\frac{S_{\mathrm{eq}}}
{b\int G_{xz}(z)\,dz}.
}
\]

For a homogeneous rectangular section this construction recovers exactly

\[
\boxed{
K_{\mathrm{eq}}=\frac56.
}
\]

### 4.4 Primary HOST warping

The primary HOST shear-mode derivative is obtained directly from the equilibrium shear strain,

\[
f_1'(z)
\propto
\widehat{\gamma}(z).
\]

The production normalization is arbitrary from the standpoint of the physical subspace. HOST uses a deterministic normalization,

\[
\boxed{
f_1'(z)
=
\frac{\widehat{\gamma}(z)}
{\max_{\zeta}|\widehat{\gamma}(\zeta)|},
}
\]

with

\[
f_1(0)=0.
\]

For piecewise-constant layer properties, \(\widehat{\tau}\) and \(f_1'\) are piecewise quadratic and \(f_1\) is piecewise cubic.

At a material interface, shear traction is continuous,

\[
[\tau_{xz}]=0,
\]

but shear strain generally is not:

\[
[\gamma_{xz}]
=
\left[
\frac{\tau_{xz}}{G_{xz}}
\right]
\neq 0
\]

when \(G_{xz}\) jumps. Hence \(f_1'\) possesses physically admissible one-sided interface traces.

---

## 5. Closed form for a symmetric three-layer sandwich

Let

\[
H=\frac h2,
\qquad
c=\frac{h_c}{2},
\]

with face properties \(E_f,G_f\) and core properties \(E_c,G_c\).

For a symmetric section,

\[
z_n=0.
\]

For \(z\ge0\), the equilibrium shear stress is

### Face: \(c\le z\le H\)

\[
\boxed{
\frac{\tau_{xz}(z)}{V}
=
\frac{E_f}
{2D_R}
\left(H^2-z^2\right).
}
\]

### Core: \(0\le z\le c\)

\[
\boxed{
\frac{\tau_{xz}(z)}{V}
=
\frac{
E_f(H^2-c^2)
+
E_c(c^2-z^2)
}
{2D_R}.
}
\]

The corresponding shear strain is

\[
\gamma_{xz}(z)
=
\frac{\tau_{xz}(z)}
{G(z)},
\]

so \(f_1'\) is piecewise quadratic and \(f_1\) is piecewise cubic.

---

## 6. Variational secondary mode

The primary equilibrium mode represents the energetically and statically admissible interior shear distribution.

A second independent mode is introduced to represent higher-order longitudinal adaptation and boundary-layer behavior.

On the symmetric half-section \(z\in[0,h/2]\), seek an odd mode

\[
f_2(0)=0
\]

that is orthogonal to the primary mode under the longitudinal-energy inner product,

\[
\boxed{
\int_0^{h/2}
E_x(z)
f_1(z)f_2(z)
\,dz
=
0.
}
\]

The weak variational problem is

\[
\boxed{
\int_0^{h/2}
G_{xz}(z)
f_2'(z)v'(z)
\,dz
=
\lambda_2
\int_0^{h/2}
E_x(z)
f_2(z)v(z)
\,dz,
}
\]

for all admissible \(v\) in the constrained subspace.

Equivalently, introducing a Lagrange multiplier \(\mu\) for the orthogonality constraint gives the strong form

\[
\boxed{
-\left(G_{xz}f_2'\right)'
=
\lambda_2 E_x f_2
+
\mu E_x f_1.
}
\]

The interface and outer-surface conditions are

\[
[f_2]=0,
\]

\[
[G_{xz}f_2']=0,
\]

\[
G_{xz}f_2'\big|_{z=h/2}=0.
\]

The lowest positive constrained eigenmode is selected as the secondary HOST mode.

The section characteristic scale associated with this problem is

\[
\boxed{
\ell_{\mathrm{section},2}
=
\frac{1}
{\sqrt{\lambda_2}}.
}
\]

This is a through-thickness modal scale. It must not automatically be interpreted as a longitudinal boundary-layer decay length.

---

## 7. Generalized section stiffness

Define the physical strain vector

\[
\boldsymbol\epsilon_b
=
\begin{bmatrix}
\varepsilon_{xx}\\
\gamma_{xz}
\end{bmatrix}
=
\mathbf B(z)\mathbf e_6,
\]

with

\[
\mathbf B(z)
=
\begin{bmatrix}
1 & z & f_1 & f_2 & 0 & 0\\
0 & 0 & 0 & 0 & f_1' & f_2'
\end{bmatrix}.
\]

The six-dimensional generalized section stiffness is

\[
\boxed{
\mathbf K_6
=
b
\int_{z_b}^{z_t}
\mathbf B^T
\mathbf C_b
\mathbf B
\,dz.
}
\]

The generalized constitutive relation is

\[
\begin{bmatrix}
N\\
M\\
H_1\\
H_2\\
Q_1\\
Q_2
\end{bmatrix}
=
\mathbf K_6
\begin{bmatrix}
\varepsilon_0\\
\kappa\\
\chi_1\\
\chi_2\\
\beta_1\\
\beta_2
\end{bmatrix}.
\]

---

## 8. Exact zero-axial-force condensation

Partition

\[
\mathbf K_6
=
\begin{bmatrix}
K_{00} & \mathbf K_{0t}\\
\mathbf K_{t0} & \mathbf K_{tt}
\end{bmatrix}.
\]

For

\[
N=0,
\]

the axial generalized strain is

\[
\varepsilon_0
=
-K_{00}^{-1}
\mathbf K_{0t}
\mathbf e_5,
\]

where

\[
\mathbf e_5
=
\begin{bmatrix}
\kappa&
\chi_1&
\chi_2&
\beta_1&
\beta_2
\end{bmatrix}^{T}.
\]

The condensed section stiffness is the Schur complement

\[
\boxed{
\mathbf K_5
=
\mathbf K_{tt}
-
\mathbf K_{t0}
K_{00}^{-1}
\mathbf K_{0t}.
}
\]

Hence

\[
\begin{bmatrix}
M\\
H_1\\
H_2\\
Q_1\\
Q_2
\end{bmatrix}
=
\mathbf K_5
\begin{bmatrix}
\kappa\\
\chi_1\\
\chi_2\\
\beta_1\\
\beta_2
\end{bmatrix}.
\]

The classical sandwich reduction

\[
D_R=D-\frac{B^2}{A}
\]

is the scalar special case of this generalized condensation operation.

---

## 9. Variational beam functional and field equations

The total potential energy can be written as

\[
\boxed{
\Pi
=
\frac12
\int_0^L
\mathbf e_5^T
\mathbf K_5
\mathbf e_5
\,dx
-
W_{\mathrm{ext}}.
}
\]

Stationarity gives the beam equilibrium structure

\[
M_{,x}=V,
\]

\[
V_{,x}+q=0,
\]

\[
H_{1,x}=Q_1,
\]

\[
H_{2,x}=Q_2.
\]

Together with

\[
w_{,x}=\theta,
\qquad
\theta_{,x}=-\kappa,
\]

\[
\beta_{1,x}=\chi_1,
\qquad
\beta_{2,x}=\chi_2,
\]

this yields the eight-state first-order system

\[
\boxed{
\mathbf y
=
\begin{bmatrix}
w&
\theta&
\beta_1&
\beta_2&
M&
V&
H_1&
H_2
\end{bmatrix}^{T},
}
\]

\[
\boxed{
\mathbf y_{,x}
=
\mathbf A\mathbf y
+
\mathbf b\,q.
}
\]

For uniform sections, exact segment propagation is obtained from the matrix exponential.

---

## 10. Longitudinal boundary-layer spectrum

The eigenvalues of the homogeneous state matrix,

\[
\mathbf A\mathbf v_i=r_i\mathbf v_i,
\]

define the longitudinal characteristic exponents.

For a stable mode

\[
r_i=-k_i+i\omega_i,
\]

the physical decay length is

\[
\boxed{
\ell_i^{\mathrm{BL}}
=
\frac{1}{k_i}
=
\frac{1}
{|\operatorname{Re}(r_i)|}.
}
\]

If \(\omega_i\neq0\), the oscillation wavelength is

\[
\Lambda_i
=
\frac{2\pi}
{|\omega_i|}.
\]

These longitudinal scales are distinct from

\[
1/\sqrt{\lambda_2}.
\]

For the canonical Gyroid RD12 sandwich, the equilibrium-primary formulation gives approximately

\[
\ell_{\mathrm{section},2}
=
13.8392\ \mathrm{mm},
\]

\[
\ell_{\mathrm{slow}}^{\mathrm{BL}}
=
13.9745\ \mathrm{mm},
\]

\[
\ell_{\mathrm{fast}}^{\mathrm{BL}}
=
0.6695\ \mathrm{mm}.
\]

The slow \(\sim14\) mm scale is nearly unchanged relative to the earlier material-informed basis, while the fast scale is basis-sensitive. The slow scale is therefore the more robust higher-order characteristic length in the present benchmark.

---

## 11. Interpretation of the two modes

The preferred physical interpretation is

\[
\boxed{
f_1:
\text{equilibrium-consistent interior shear distribution}
}
\]

and

\[
\boxed{
f_2:
\text{higher-order longitudinal adaptation / boundary-layer enrichment}.
}
\]

The secondary mode should not be interpreted primarily as a scalar shear-correction factor.

With the earlier material-informed primary basis, local condensation of the secondary mode gave

\[
\alpha_{\mathrm{legacy+var}}
\approx
0.952699.
\]

This was close to a separate global fitted shear factor and initially suggested an intrinsic \(4.7\%\) softening.

After replacing the primary basis by the equilibrium-derived mode,

\[
\alpha_{\mathrm{eq+var}}
\approx
0.999914.
\]

Therefore the earlier \(\sim4.7\%\) condensation was largely a basis-compensation effect: the secondary mode was repairing the assumed primary shear shape.

---

## 12. Canonical Gyroid RD12 verification

For the canonical homogenized Gyroid RD12 sandwich section,

\[
S_{\mathrm{eq}}
=
3.864302\times10^6\ \mathrm{N},
\]

\[
K_{\mathrm{eq}}
=
0.552937625.
\]

The equilibrium construction satisfies

\[
V_{\mathrm{reconstructed}}=1
\]

to machine precision for a unit input shear resultant.

The variational secondary eigenvalue is

\[
\lambda_2
=
5.22132\times10^{-3}\ \mathrm{mm}^{-2},
\]

giving

\[
\ell_{\mathrm{section},2}
=
13.8392\ \mathrm{mm}.
\]

The six-case 4PB slenderness verification gives:

| \(a/h\) | FE \(w\) [mm] | equilibrium \(f_1\), 1 mode | error | equilibrium \(f_1\)+variational \(f_2\) | error |
|---:|---:|---:|---:|---:|---:|
| 0.968 | 0.02018573 | 0.01991343 | -1.349% | 0.01998260 | -1.006% |
| 1.452 | 0.04142270 | 0.04123650 | -0.450% | 0.04130546 | -0.283% |
| 1.935 | 0.07354360 | 0.07344298 | -0.137% | 0.07351191 | -0.043% |
| 2.903 | 0.18048109 | 0.18055242 | +0.040% | 0.18062132 | +0.078% |
| 3.871 | 0.36108920 | 0.36133420 | +0.068% | 0.36140308 | +0.087% |
| 4.839 | 0.63545999 | 0.63588080 | +0.066% | 0.63594963 | +0.077% |

Across these six cases,

\[
\mathrm{RMS}_{1\mathrm{mode}}
=
0.5847\%,
\]

\[
\mathrm{RMS}_{2\mathrm{mode}}
=
0.4309\%,
\]

and

\[
\max|\mathrm{error}|_{2\mathrm{mode}}
=
1.0063\%.
\]

The legacy material-informed one-mode model gives

\[
\mathrm{RMS}
=
2.4689\%,
\]

while legacy material-informed plus the variational secondary mode gives

\[
\mathrm{RMS}
=
0.6755\%.
\]

Thus the equilibrium-derived primary mode provides the dominant improvement, while the variational secondary mode produces a smaller additional gain.

---

## 13. Through-thickness FE validation

At the bulk shear station of the canonical 4PB benchmark, the earlier material-informed primary mode differed substantially from the three-dimensional FE shear field.

Mean profile errors over the six slenderness cases were approximately

\[
\mathrm{RMSE}_{\gamma,\mathrm{legacy}}
=
25.79\%,
\]

\[
\mathrm{RMSE}_{\tau,\mathrm{legacy}}
=
26.57\%.
\]

The equilibrium-derived primary mode reduced these to approximately

\[
\boxed{
\mathrm{RMSE}_{\gamma,\mathrm{eq}}
=
0.33\%
}
\]

and

\[
\boxed{
\mathrm{RMSE}_{\tau,\mathrm{eq}}
=
0.19\%.
}
\]

This indicates that the dominant discrepancy of the earlier one-mode model came from the assumed primary shear shape rather than from an unavoidable need for a second distributed shear mode.

---

## 14. Relation to Gordaninejad and Bert

Gordaninejad and Bert, *A New Theory for Bending of Thick Sandwich Beams*, International Journal of Mechanical Sciences, 31(11/12), 925-934, 1989, derive a thick-sandwich beam theory in which shear deformation is retained in all layers and the shear correction is determined through equilibrium and equivalent shear strain energy.

The present HOST development uses the same physical principle for the primary shear field but extends it in three directions:

1. the core may be a fully homogenized anisotropic TPMS material;
2. the complete free-stress-condensed \(2\times2\) longitudinal/shear constitutive law is retained in the final section stiffness;
3. an additional variationally generated higher-order warping mode is included to represent longitudinal boundary-layer adaptation.

The relation

\[
D_R=D-\frac{B^2}{A}
\]

also provides a classical scalar counterpart to the generalized HOST Schur condensation

\[
\mathbf K_5
=
\mathbf K_{tt}
-
\mathbf K_{t0}
K_{00}^{-1}
\mathbf K_{0t}.
\]

---

## 15. Current limits and required transfer validation

The numerical evidence above is currently established for the canonical homogenized Gyroid RD12 4PB benchmark.

The formulation should not yet be claimed to be universally validated for all surface-based lattice cores.

Required transfer checks include at least:

- additional TPMS topology, e.g. Primitive and Diamond;
- additional relative density;
- changed face thickness;
- at least one different core thickness;
- deep and slender geometries;
- comparison of global deflection, shear profiles, generalized resultants, and boundary-layer scales.

These transfer checks should use the same theory without fitted parameters.

---

## 16. Production implementation

The current HOST production implementation is separated into:

- `src/host/equilibrium_shear_modes.py`
  Exact layerwise equilibrium/energy primary shear mode.

- `src/host/variational_shear_modes.py`
  Constrained variational secondary mode.

- `src/host/beam_two_mode.py`
  \(K_6\), \(N=0\rightarrow K_5\) condensation, generalized recovery, state matrix, exact propagation, and boundary-layer spectrum.

- `src/host/beam_two_mode_point_loads.py`
  Exact uniform-segment multiple-shooting solution with point-force jumps.

- `verification/slenderness/run_variational_two_mode_host.py`
  Canonical 4PB production verification.

- `verification/slenderness/analyze_bert_equilibrium_shear.py`
  Independent equilibrium/energy audit and FE profile comparison.

The legacy material-informed primary mode remains accessible for regression and historical comparison but is not the preferred production basis.
