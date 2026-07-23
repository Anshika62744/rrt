# Theory notes

Short derivations behind the three products. Symbols: `s` is the line-of-sight
(LOS) parameter in R_sun; `ê` the unit look direction; `r₀` the sky-plane
position of a pixel (⊥ `ê`, so `|r(s)|² = b² + s²` with impact parameter `b`).

## 1. Siddon ray–cell integration

Rather than resampling the MHD grid onto the LOS, each ray is intersected
**analytically** with the three families of cell boundaries of a spherical grid,
then the integrand is evaluated once per traversed cell (exact cell traversal, no
interpolation of the field).

- **Radial shells** `|r(s)| = rᵢ` → `s² + b·s + c = 0` with
  `b = 2 r₀·ê`, `c = |r₀|² − rᵢ²`. Two roots (near/far).
- **Polar cones** `z(s)² = cos²θ · |r(s)|²` → a quadratic in `s`
  (degenerate → linear when `êz² = cos²θ`). Cones at `θ = 0, π` are the polar
  axis, not surfaces, and are dropped.
- **Azimuthal half-planes** `y(s)cosφ = x(s)sinφ` → **linear** in `s`. On a
  periodic (full-2π) grid the first and last half-plane coincide and the duplicate
  is dropped.

All crossings in `[s_lo, s_hi]` are collected and sorted; each segment
`[sₖ, sₖ₊₁]` is evaluated at its midpoint. The **cell** a midpoint falls in is
found by digitising into cell **edges**:

```
i = searchsorted(edges, x, side="right") − 1
```

This edge rule is shared by both integrators and the LOS diagnostic, so the
plotted voxel bands are exactly the cells the integrator reads.

**Domain detection** (`prepare_domain`) inspects the coordinate arrays and reports
`full_theta` / `full_phi`; the angular rejection tests are skipped for whichever
coordinate is complete, so a wedge, polar cap, full shell and full disk all use
the same kernel.

**Photospheric clip.** The photosphere is opaque. For an on-disk pixel (`b ≤ 1`)
the LOS is stopped at the near-side crossing `s = −√(1−b²)`, so the far side is
never integrated; off-limb pixels take the full chord `|s| ≤ √(r_max²−b²)`.

## 2. EUV / AIA emission

For an optically-thin plasma the intensity in an AIA channel is

```
I_λ = ∫ nₑ²(s) · R_λ(T(s)) · dℓ          [DN s⁻¹ pixel⁻¹]
```

with `dℓ = ds · R_sun` in cm. A **flat unit response** (`R ≡ 1`) reduces this to
the column emission measure `EM = ∫ nₑ² dℓ` [cm⁻⁵]; for an isothermal corona
`I_λ = R_λ(T_iso) · EM`.

`R_λ(T)` spans many decades and has steep hot peaks (94, 131 Å), so it is
interpolated **log-log**: `log₁₀R` vs `log₁₀T`, clamped at the table ends. The
`rrt.euv` kernel and the `rrt.los` diagnostic use the identical rule.

> **Why a raw-EM map looks saturated.** `EM = ∫ nₑ²` is dominated by the densest,
> coolest plasma at the base, so the on-disk area pins to the colorbar top. A
> *real* AIA image weights by `R_λ(T)`, which only lights up plasma near the
> channel's peak temperature — that is what shows loops instead of a blob. Pass a
> real `(T_resp, R_resp)` (not the flat response) and/or clip the dense base with
> `r_range` to see structure.

## 3. White-light polarised brightness (pB)

Thomson scattering of photospheric light by coronal electrons. Using the van de
Hulst / Billings geometry factors `A(r)`, `B(r)` (functions of `Ω`, with
`sinΩ = R_sun/r`):

```
pB = ∫ nₑ(s) · K(r(s)) · dℓ ,   K(r) = σ_e · [(1−u)·A(r) + u·B(r)]
```

- `σ_e = r_e² = 7.95e-26 cm²` — the classical electron radius squared (package
  default). This is the standard van de Hulst / Billings coronal-pB constant: the
  `A`, `B` factors already carry the angular dependence of the scattering, so the
  prefactor is `r_e²`, **not** the angle-integrated total cross-section
  `σ_T = (8π/3) r_e² = 6.65e-25`. Pass `SIGMA_T=6.65e-25` only if your `A, B`
  definitions omit that angular integration.
- `u` — limb-darkening coefficient (default 0.63).

Units: `[cm⁻³]·[cm²]·[cm]` cancels, so pB is **dimensionless** — a brightness
ratio (relative to the mean solar disk brightness for this normalisation), not a
per-length quantity. The same opaque-body clip applies. An optional **NRGF** (Normalised
Radial Graded Filter) removes the steep radial falloff per annulus so streamers
and CME structure stand out.

## 4. LOS diagnostics — why they prove correctness

`plot_los_voxel_proof` back-substitutes each kept sample into the defining
geometry and reports the residuals:

- every kept coordinate lies inside its **assigned** `(r, θ, φ)` cell box;
- the fitted closest approach equals the impact parameter `b`;
- `r(s)` matches `√(b² + (s−s₀)²)` to ~machine zero;
- the spherical→Cartesian round-trip of each midpoint returns the ray point.

Because the walker uses the same intersection helpers, edge rule and clip as the
integrators, agreement of its per-segment `Σ I` with `run_siddon` for a pixel is a
direct end-to-end check (asserted in the test suite to `rtol=1e-9`).
