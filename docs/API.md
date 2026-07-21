# API reference

Curated public surface. Import from the top level (`from rrt import …`) or the
submodule. Full signatures live in the source docstrings.

## `rrt.io` — the unified loader

```python
load_mhd(density, temperature=None, *,
         input_case="auto", fmt="auto", nghost="auto",
         density_units="auto", temperature_units="auto", angle_units="auto",
         t_iso=1.0e6, ne_scale=1.0, t_scale=1.0, mu=0.6, m_p=1.6726e-24,
         r_range=None, datasets=None) -> MHDCube
```
`density` is a path or `(path, dataset_key)`. `temperature` is a path (→ separate
case), a dataset key (→ single case) or `None`. See
[DATA_FORMAT.md](DATA_FORMAT.md).

```python
MHDCube(r, theta, phi, ne, T, meta)      # dataclass
  .with_isothermal(t_iso) -> MHDCube
  .summary() -> str
detect_format(path) -> "arms" | "mas"
load_response(path, wavelength=171) -> (T_resp, R_resp)   # re-exported
```

## `rrt.observer`

```python
make_observer(phi_obs_deg, B0_deg) -> (n_obs, e_los, x_img, y_img)
image_grid(rmax, npix) -> (Xg, Yg, Rperp)

# time-based observer (Carrington-framed data, e.g. MAS/PSI) — needs sunpy
sub_earth_angles(obs_time) -> (L0_deg, B0_deg)          # sub-Earth L0, B0
make_observer_from_time(obs_time, phi_offset_deg=0.0)   # → make_observer(L0, B0)
view_angles(view_dict) -> (phi_obs_deg, B0_deg)         # resolves obs_time or explicit
```
`obs_time` is a UTC string/`astropy.time.Time`; `L0` is the sub-Earth Carrington
longitude (use as `phi_obs_deg` for MAS cubes), `B0` the heliographic latitude.

## `rrt.euv`

```python
run_siddon(Xg, Yg, x_img, y_img, e_los,
           r_ori, theta_ori, phi_ori, ne_grid, T_grid, T_resp, R_resp,
           Npix=256, Rmax=1.5, WAVELENGTH=171, OBS_TIME="",
           Rsun_cm=6.96e10, R_body=1.0, plot=True, save=True) -> ndarray  # DN/s/pix
siddon_integrate(...)   # the numba kernel (advanced use)
```
A flat response (`rrt.response.FLAT_T/FLAT_R`) returns column `EM = ∫ nₑ² dℓ`.

## `rrt.wl`

```python
run_siddon_pB(Xg, Yg, x_img, y_img, e_los,
              r_ori, theta_ori, phi_ori, ne_grid,
              Npix=256, Rmax=5.0, Rocc=1.0, OBS_TIME="",
              Rsun_cm=6.96e10, R_body=1.0, U_LIMB=0.63,
              SIGMA_T=7.95e-26, plot=True, save=True,   # r_e^2 (Billings/vdH)
              log_limits=None) -> ndarray   # pB [cm^-1]
siddon_integrate_pB(...)
```

## `rrt.los`

```python
sample_along_los(Xp, Yp, x_img, y_img, e_los,
                 r_ori, theta_ori, phi_ori, ne_grid, T_grid,
                 T_resp=None, R_resp=None, Rsun_cm=6.96e10,
                 r_occ=1.0, tol=1e-6) -> dict
plot_los_profile(samples, title=None) -> (fig, axes)
plot_los_voxel_proof(Xp, Yp, ..., samples=None, deg=True,
                     savepath=None, show=True) -> verdict dict
analyze_los(Xp, Yp, ..., T_resp=None, R_resp=None,
            show_profile=True, show_voxel_proof=True, show=True) -> samples
```

## `rrt.geometry` (shared core)

```python
make_edges(centres) -> edges
prepare_domain(r_ori, theta_ori, phi_ori, tol=1e-6) -> dict
  # keys: r_edges, theta_edges, phi_edges, theta_cross, phi_cross,
  #       full_theta, full_phi, phi_span, r_min, r_max
_radial_intersections / _theta_intersections / _phi_intersections   # njit helpers
```

## `rrt.response`

```python
load_response(path, wavelength=171) -> (T_resp, R_resp)
available_channels(path) -> [int]
response_at(T, T_resp, R_resp)          # log-log, scalar or array
isothermal_response(t_iso, T_resp, R_resp) -> float
FLAT_T, FLAT_R                           # flat unit response (column EM)
```

## `rrt.viz`

```python
aia_cmap(wavelength)
draw_wireframe_sphere(ax, n_obs, x_img, y_img, ...)
nrgf(img, Xg, Yg, r_inner=1.0, r_outer=None, n_radial_bins=120, ...)
plot_map(img, rmax, *, cmap, log, mask_disk, occulter, cbar_label, title, vlim, wireframe, ...)
make_movie(frame_paths, out_file, fps=4, frame_dir=None) -> path
```

## `rrt.config` / `rrt.cli`

```python
load_config(path) -> Config
load_cube_from_config(cfg) -> MHDCube
# console scripts (see examples/configs/):
#   rrt-euv config.yaml   rrt-wl config.yaml   rrt-los config.yaml
```

## `rrt.synthdata`

```python
spherical_grid(domain, nr, nth, nph, r_range) -> (r, theta, phi)
analytic_ne(r, theta, phi, n0=1e8, r0=1.0, p=2.0, aniso=False)
analytic_T(r, theta, phi, t0=1.5e6, gradient=False)
make_cube(domain, case, ...) -> dict
write_arms(path, r, theta, phi, ne, T=None, *, nghost=0, ...)
write_mas(path, r, theta, phi, field, *, ne_scale=1.0)
write_response(path, channels=(171, 193, 211))
# domains: "wedge" | "polar_cap" | "full"
```
