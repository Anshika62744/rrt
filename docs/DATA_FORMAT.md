# Data format

`rrt` reads spherical `(r, θ, φ)` MHD cubes from HDF5. One loader,
[`rrt.io.load_mhd`](../src/rrt/io.py), handles everything below; every
auto-detected quantity has an explicit override.

## The three input cases

| case | how you call it | temperature |
|------|-----------------|-------------|
| **1. separate** | `load_mhd("rho.h5", temperature="t.h5")` | from the second file |
| **2. single**   | `load_mhd("both.h5", temperature="vars/T")` or auto-probe | from a dataset in the density file |
| **3. density-only** | `load_mhd("rho.h5", t_iso=1.0e6)` | isothermal fallback `t_iso` [K] |

`input_case="auto"` decides by: a temperature **path** → separate; a temperature
**dataset key** (or a probe hit in the density file among
`vars/T, vars/temperature, vars/Temp, vars/te, vars/temp`) → single; otherwise
density-only.

## Supported HDF5 layouts

### ARMS (`fmt="arms"`)

```
/                         attrs: axis_order="(phi, theta, r, var)",
                                 ghost_cells_included=<bool>, nghost=<int>
├── coords/
│   ├── r      (…)        attrs: units="R_sun"
│   ├── theta  (…)        attrs: units="rad" | "deg"
│   └── phi    (…)        attrs: units="rad" | "deg"
└── vars/
    ├── rho    (nphi, ntheta, nr)   attrs: units="cm^-3" | "g cm^-3", axis_order
    └── T      (nphi, ntheta, nr)   attrs: units="K" | "MK" | "log10"   (optional)
```

- **Ghost cells**: if `ghost_cells_included` is true (or an `nghost` attr is
  present), `nghost` cells are stripped from **every** axis of coords and fields.
  Override with `nghost=<int>` (0 keeps them).
- **Axis order**: taken from the `axis_order` attribute (authoritative). If it is
  absent, the loader falls back to matching dataset dimensions to the coordinate
  lengths, and finally to `(phi, theta, r)` with a recorded warning.

### MAS / PSI (`fmt="mas"`)

```
/
├── Data   (…, …, …)      one field cube, code units, no attributes
├── dim1   (…)            one coordinate axis
├── dim2   (…)
└── dim3   (…)
```

Axes are identified from their ranges: `r` reaches furthest from the Sun; of the
remaining two, `φ` has the wider span than `θ`. Code units are converted with
`ne_scale` / `t_scale`. (This heuristic assumes `r.max()` exceeds 2π, as
coronal-scale MAS grids do.)

## Units, angles, axis order — auto-detected

| quantity | detection | override |
|----------|-----------|----------|
| density  | `g`+`cm` in units, or median positive value `< 1e-5` → mass CGS → `/(µ·m_p)` | `density_units="mass_cgs" \| "number_cgs"` |
| temperature | units attr, else median: `3.5–9` → log10, `0.05–50` → MK, else K | `temperature_units="K" \| "MK" \| "log10"` |
| angles   | `θ>π+ε` or `\|φ\|>2π+ε` → degrees → radians | `angle_units="rad" \| "deg"` |
| axis order | `axis_order` attr → size match → `(2,1,0)` + warning | `datasets={…}` |

The loader also forces every coordinate **ascending** (flipping the matching data
axis) and `np.unwrap`s φ; a still-decreasing φ raises a clear error.

## Output — `MHDCube`

```python
cube.r, cube.theta, cube.phi   # 1-D ascending centres (R_sun, rad, rad)
cube.ne                        # (nr, nθ, nφ) cm⁻³, contiguous
cube.T                         # (nr, nθ, nφ) K (or isothermal fill)
cube.meta                      # provenance: input_case, fmt, nghost, axis_perm, …
cube.summary()                 # human-readable one-block report
```

## AIA response table

A NumPy `.npz` (canonical `aia_temp_response_chiantifix.npz`) with:

- `logte` — `log10(T/K)` grid,
- `resp{λ}` — one response curve per channel (`resp171`, `resp193`, …),
  in `DN cm⁵ s⁻¹ pixel⁻¹`.

`rrt.response.load_response(path, wavelength)` returns `(T_resp, R_resp)` with
`T_resp = 10**logte`. R(T) is interpolated **log-log** everywhere.

## Coordinate & unit conventions

| quantity | symbol | unit |
|----------|--------|------|
| radius | `r` | R_sun |
| angles | `θ, φ` | radians (internally) |
| electron density | `nₑ` | cm⁻³ |
| temperature | `T` | K |
| EUV intensity | `I_λ` | DN s⁻¹ pixel⁻¹ |
| white-light pB | `pB` | dimensionless brightness ratio — `[cm⁻³]·[cm²]·[cm]` cancels (with `σ_e = r_e² = 7.95e-26`) |
