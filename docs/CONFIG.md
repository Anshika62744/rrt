# Config reference

Every `rrt-euv` / `rrt-wl` / `rrt-los` run is driven by one YAML file. Only
`data.density` is required; everything else has a default. The **command** picks
the product, not the filename — the same config can be fed to more than one
command if it carries the keys that command needs.

- `rrt-euv` needs `response:` + `wavelength:` (or `wavelengths:`)
- `rrt-wl` needs nothing extra (a `white_light:` block tunes it)
- `rrt-los` needs `los: {probes: […]}`

Ready-made examples: [`examples/configs/`](../examples/configs/).

---

## `data:` — the cube

| Key | Default | Meaning |
|---|---|---|
| `density` | *required* | Path to the density HDF5 file, or `[path, dataset_key]`. |
| `temperature` | *(none)* | A second **path** (separate-file case), a **dataset key** like `vars/T` inside the density file, or omitted → isothermal `t_iso`. |
| `input_case` | `auto` | `auto` \| `separate` \| `single` \| `density_only`. |
| `fmt` | `auto` | `auto` \| `arms` \| `mas`. Auto-detected from the file layout. |
| `t_iso` | `1.0e+6` | Isothermal temperature [K] when no temperature field exists. |
| `ne_scale` | `1.0` | Multiplies density (MAS code units → cm⁻³; `1.0e+8` for these runs). |
| `t_scale` | `1.0` | Multiplies temperature (MAS code units → K; `2.807e+7` for these runs). |
| `r_range` | *(none)* | `[r_min, r_max]` radial trim [R☉]. Trims the dense base and caps integration depth. |
| `nghost` | `auto` | Ghost cells to strip; `auto` reads the file attributes. |
| `density_units` | `auto` | `auto` \| `mass_cgs` \| `number_cgs`. |
| `temperature_units` | `auto` | `auto` \| `K` \| `MK` \| `log10`. |
| `angle_units` | `auto` | `auto` \| `rad` \| `deg`. |
| `mu`, `m_p` | `0.6`, `1.6726e-24` | Mean molecular weight and proton mass, for mass → number density. |
| `datasets` | *(none)* | Override ARMS dataset keys, e.g. `{rho: vars/rho_e}`. |

Always read the `cube.summary()` a run prints first: it shows the detected
`case`/`fmt` and the resulting `ne` / `T` ranges. A wrong `ne_scale` or axis guess
shows up there before it shows up in an image.

## Image and intensity scale

| Key | Default | Meaning |
|---|---|---|
| `image.Npix` | `256` | Pixels per side. Cost scales as Npix². |
| `image.Rmax` | `1.5` | Field-of-view half-width [R☉]. ≈1.5 for a disk, ≈5 for a coronagraph. |
| `scale` | `log` | `log` \| `linear`. Rejected at load time if it's anything else. |
| `vlim` | *(none)* | `[lo, hi]` colorbar limits in **linear** data units (DN/s for EUV, dimensionless for pB, σ for NRGF). Omitted → 1–99 percentile of each image. |
| `vlim_log` | *(none)* | The same limits written as log₁₀ exponents: `[0.5, 4.0]` ≡ `vlim: [3.1623, 10000]`. Ignored if `vlim` is set. |
| `output_dir` | `out` | Where PNGs are written. |

`Rmax` is the *image window*; `r_range` is *how much of the cube* is integrated.
Anything past the cube's outer radius renders empty.

## EUV — `rrt-euv`

| Key | Default | Meaning |
|---|---|---|
| `response` | *(none)* | AIA response `.npz`. Without it you get raw column EM, not DN/s. |
| `wavelength` | `171` | One channel: 94, 131, 171, 193, 211, 304, 335. |
| `wavelengths` | *(none)* | A **list** → all channels in one grid figure per view, instead of one PNG per channel. Overrides `wavelength`. |

### `panel:` — multi-channel grid only

| Key | Default | Meaning |
|---|---|---|
| `ncols` | `3` | Columns in the grid. |
| `dynamic_range` | `1.0e+8` | Decades below each panel's own peak, when no explicit limits are given. Lower = more contrast. |
| `vlim` | *(none)* | `[lo, hi]` for every panel, or a per-channel map `{94: [lo, hi], 171: […]}`. Channels left out fall back to `dynamic_range`. |
| `vlim_log` | *(none)* | The same as log₁₀ exponents — the form AIA limits are usually quoted in. |
| `cbar` | `log10` | `log10` → ticks read `-1.00 … 3.50`; `decades` → ticks read `10⁻¹ … 10³·⁵`. |
| `cbar_ticks` | `6` | Number of labels per colorbar in `log10` mode. |
| `facecolor` | `black` | Page background. Ticks, labels and the title flip to black on a light background. Panel interiors stay black. |
| `mask_disk` | `false` | `true` hides on-disk pixels (coronagraph style). |

Precedence for limits: `panel.vlim` → `panel.vlim_log` → top-level `vlim` /
`vlim_log` → per-panel `dynamic_range`.

An isothermal cube integrates **once** and scales each channel by `R_λ(T_iso)`; a
real temperature field integrates once per channel.

## White light — `rrt-wl`

| Key | Default | Meaning |
|---|---|---|
| `white_light.Rocc` | `1.0` | Occulter radius [R☉]; pixels inside are blanked. |
| `white_light.U_LIMB` | `0.63` | Limb-darkening coefficient *u*. |
| `white_light.SIGMA_T` | `7.95e-26` | Thomson constant [cm²] — the classical electron radius squared, which is what the van de Hulst / Billings A, B factors expect. |
| `white_light.nrgf` | `false` | `true` → Normalised Radial Graded Filter, `(pixel − μ_r)/σ_r` per annulus. |
| `white_light.radial_power` | `0` | `k` → plot `pB·ρᵏ`, cancelling the steep radial falloff so streamers survive a linear stretch. `k ≈ 3` suits 1–3 R☉. Ignored when `nrgf: true`. |
| `white_light.cmap` | `Oranges_r` | Any matplotlib colormap; `gray` gives the classic coronagraph look. |

pB is **dimensionless** (`[cm⁻³]·[cm²]·[cm]` cancels) — a brightness ratio, not a
per-length quantity. NRGF output is signed, so it is always drawn on a linear
scale regardless of `scale:`, and its `vlim` is in units of σ (e.g. `[-2, 4]`).

## LOS diagnostics — `rrt-los`

| Key | Default | Meaning |
|---|---|---|
| `los.probes` | `[[0, 0]]` | List of `[X, Y]` impact points on the sky [R☉]. |

`b = √(X²+Y²)` must be inside the cube's outer radius, and for a partial domain
(a wedge) the ray has to actually cross it — otherwise the probe is reported as
having zero kept samples and skipped. `b < 1` is an on-disk probe, clipped at the
photosphere; `b > 1` sees the full chord.

## `views:` — one image per entry

Each entry needs a `label` (used in the output filename) plus one of:

```yaml
views:
  # explicit angles in the simulation frame
  - {label: Side view, phi_obs_deg: -70, B0_deg: 0}

  # sub-Earth viewpoint from a time (Carrington-framed data, e.g. MAS/PSI):
  # L0 and B0 come from sunpy
  - {label: earth, obs_time: '2024-03-30T20:49:00'}

  # ...with an override: same instant, but looking from near the pole
  - {label: pole, obs_time: '2024-03-30T20:49:00', B0_deg: 90}

  # true top-down view, image rolled so up_lon_deg points up
  # (defaults to the sub-Earth L0 of obs_time)
  - {label: earth_top, obs_time: '2024-03-30T20:49:00', pole: north}
  - {label: feature_top, pole: north, up_lon_deg: 209}
```

| Key | Meaning |
|---|---|
| `label` | Name in the figure title and the output filename. |
| `phi_obs_deg` | Observer longitude in the simulation frame [deg]. |
| `B0_deg` | Observer heliographic latitude [deg]. |
| `obs_time` | UTC time → sub-Earth `L0`/`B0` via sunpy. Explicit angles override either. |
| `phi_offset_deg` | Added to `L0`, to absorb a φ zero-point offset. |
| `pole` | `north` \| `south` — a true top-down view. |
| `up_lon_deg` | Which Carrington longitude points up in a `pole` view. |

`make_observer(phi, ±90)` ignores the longitude and pins the image roll to
Carrington 0°, which is why `pole:` exists: it sets the roll explicitly so a pole
view lines up with the matching Earth view.

## Full example

```yaml
data:
  density: data/rho_bin.h5
  temperature: data/t_bin.h5
  fmt: mas
  ne_scale: 1.0e+8
  t_scale: 2.807e+7
  r_range: [1.03, 5.0]
response: data/aia_temp_response_chiantifix.npz
wavelength: 171
scale: log
vlim_log: [0.5, 4.0]
image: {Npix: 400, Rmax: 1.5}
views:
  - {label: earth,     obs_time: '2024-03-30T20:49:00'}
  - {label: earth_top, obs_time: '2024-03-30T20:49:00', pole: north}
los: {probes: [[1.5, 0.3], [0.5, 0.0], [2.5, 0.0]]}
output_dir: out/my_run
```

HDF5 layout spec: [`DATA_FORMAT.md`](DATA_FORMAT.md) · method: [`THEORY.md`](THEORY.md) · API: [`API.md`](API.md)
