# rrt — Rectilinear Ray Tracing of MHD cubes into synthetic observables

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn a 3-D MHD cube of the solar corona into **synthetic images an instrument
would see**. Given electron density and temperature on a spherical
`(r, θ, φ)` grid, `rrt` integrates along every image-plane pixel's line of sight
(the analytic **Siddon** ray–cell method) to produce:

- **EUV / SDO-AIA** maps — `I_λ = ∫ nₑ² R_λ(T) dℓ`
- **White-light pB** — Thomson scattering (van de Hulst / Billings), optional NRGF
- **Line-of-sight diagnostics** — walk one ray and verify the algorithm

Any domain (wedge, polar cap, full shell, full disk) and both file layouts
(**ARMS** and **MAS/PSI**) are detected automatically.

---

## Contents
1. [Setup from scratch (nothing installed)](#1-setup-from-scratch-nothing-installed)
2. [Check it works (no data needed)](#2-check-it-works-no-data-needed)
3. [Run on YOUR data](#3-run-on-your-data)
4. [Config reference — what to change](#4-config-reference--what-to-change)
5. [Tips & gotchas](#5-tips--gotchas)

---

## 1. Setup from scratch (nothing installed)

You need **conda** (for Python + the scientific libraries) and **git** (to get
the code). If you already have conda, skip to step 1.3.

### 1.1 Install Miniconda

Pick your platform, paste into a terminal:

**macOS — Apple Silicon (M1/M2/M3):**
```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init zsh
```
**macOS — Intel:** replace `arm64` with `x86_64` above.
**Linux:** replace `MacOSX-arm64` with `Linux-x86_64` above and `zsh` with `bash`.

Close and reopen the terminal so `conda` is on your PATH (the prompt shows `(base)`).

### 1.2 Get the code

```bash
git clone https://github.com/Anshika62744/rrt.git
cd rrt
```
(No git? Download the repo as a ZIP from GitHub, unzip it, and `cd` into the folder.)

### 1.3 Create the environment

```bash
conda env create -f environment.yml
conda activate rrt
```

This builds an environment named `rrt` with numpy, numba, h5py, matplotlib,
sunpy, astropy, and installs this package (so the `rrt-euv` / `rrt-wl` /
`rrt-los` commands become available). You only run `conda activate rrt` **once
per terminal window**.

> ⚠️ Do **not** paste a command that has a `#` comment on it — zsh will treat the
> comment as arguments and error. Copy the plain command only.

---

## 2. Check it works (no data needed)

The package ships a generator that writes small synthetic cubes, so you can run
the whole pipeline before touching real data:

```bash
python examples/generate_synthetic_data.py
rrt-euv examples/configs/euv_fulldisk_synth.yaml
rrt-wl  examples/configs/wl_fulldisk_synth.yaml
rrt-los examples/configs/los_fulldisk_synth.yaml
```

Each command prints a summary and writes PNGs into `out/…/`. Open one:

```bash
open out/euv_fulldisk_synth/euv_171_equatorial.png     # macOS  (Linux: xdg-open)
```

If those images appear, your install is good.

---

## 3. Run on YOUR data

The **only** thing you change per dataset is a small **YAML config file**. Copy a
template, edit the `data:` block, run one command. Below are the two common cases.

### Case A — one file, density only (→ isothermal temperature)

e.g. an ARMS snapshot like `mhd_data_0220.h5` that has density but no temperature.

Create `my_run.yaml`:
```yaml
data:
  density: /full/path/to/mhd_data_0220.h5   # ← your file
  input_case: auto                          # no temperature in file → isothermal
  t_iso: 1.0e+6                             # isothermal coronal temperature [K]
response: data/aia_temp_response_chiantifix.npz
wavelength: 171
image: {Npix: 400, Rmax: 1.5}
views:
  - {label: Side view, phi_obs_deg: -70, B0_deg: 0}
output_dir: out/my_run
```
Run:
```bash
rrt-euv my_run.yaml        # → out/my_run/euv_171_side_view.png
```

### Case B — two files, separate density + temperature (MAS/PSI)

e.g. `rho_bin.h5` + `t_bin.h5` in code units.

```yaml
data:
  density: /full/path/to/rho_bin.h5         # ← your density file
  temperature: /full/path/to/t_bin.h5       # ← your temperature file
  fmt: mas
  ne_scale: 1.0e+8                          # code-unit → cm^-3   (your normalization)
  t_scale: 2.807e+7                         # code-unit → K       (your normalization)
  r_range: [1.03, 5.0]                      # trim dense base / limit depth
response: data/aia_temp_response_chiantifix.npz
wavelength: 171
image: {Npix: 300, Rmax: 1.5}
views:
  - {label: sub-Earth, obs_time: '2024-03-30T20:49:00'}   # MAS is Carrington-framed
output_dir: out/my_run
```
Run:
```bash
rrt-euv my_run.yaml
```

### Case C — one file holding both density and temperature

```yaml
data:
  density: /full/path/to/both.h5
  temperature: vars/T       # dataset name inside the same file (or leave out to auto-find)
```

### The three products — same config, different command

| you want… | command | config changes |
|---|---|---|
| **EUV / AIA** map | `rrt-euv my_run.yaml` | `wavelength`, `image` |
| **White-light pB** | `rrt-wl my_run.yaml` | use `Rmax: 5.0`; add a `white_light:` block |
| **LOS diagnostics** | `rrt-los my_run.yaml` | add a `los: {probes: [[X, Y], …]}` block |

White-light block:
```yaml
image: {Npix: 400, Rmax: 5.0}
white_light: {Rocc: 1.0, U_LIMB: 0.63, SIGMA_T: 7.95e-26, nrgf: false}
```
LOS block (X, Y are impact points on the sky in R_sun):
```yaml
image: {Npix: 256, Rmax: 3.0}
los: {probes: [[1.5, 0.3], [2.5, 0.0]]}
```

Outputs always go to the `output_dir` you set. Ready-made templates for both real
datasets and all three products live in [`examples/configs/`](examples/configs/).

---

## 4. Config reference — what to change

```yaml
data:
  density: PATH            # required. Path to the density HDF5 file.
  temperature: PATH|KEY    # optional. Another file (separate case), a dataset
                           #   name like "vars/T" (single-file case), or omit
                           #   entirely → isothermal t_iso.
  input_case: auto         # auto | separate | single | density_only
  fmt: auto                # auto | arms | mas
  t_iso: 1.0e+6            # isothermal fallback temperature [K]
  ne_scale: 1.0            # multiply density by this (MAS code units → cm^-3)
  t_scale: 1.0             # multiply temperature by this (MAS code units → K)
  r_range: [1.03, 5.0]     # optional [r_min, r_max] radial trim [R_sun]
  nghost: auto             # auto (read from file attrs) | an integer
  density_units: auto      # auto | mass_cgs | number_cgs
  temperature_units: auto  # auto | K | MK | log10
  angle_units: auto        # auto | rad | deg
response: data/aia_temp_response_chiantifix.npz   # AIA table (EUV & LOS only)
wavelength: 171            # 94, 131, 171, 193, 211, 304, 335
wavelengths: [171, 193]    # optional; a LIST renders every channel into ONE
                           #   grid figure per view (instead of one PNG each)
image:
  Npix: 400                # pixels per side (higher = slower, sharper)
  Rmax: 1.5                # field-of-view half-width [R_sun] (≈1.5 disk, ≈5 coronagraph)
vlim: [1.0e-1, 1.0e+4]     # optional colorbar limits [DN/s, or cm^-1 for pB];
                           #   omit for the 1–99 percentile of each image
panel: {ncols: 3, dynamic_range: 1.0e+8, mask_disk: false, vlim: null}
                           # multi-channel grid only: layout, decades below each
                           #   panel's peak, and an optional [lo, hi] or
                           #   {wavelength: [lo, hi]} colorbar override
views:                     # one image per entry
  - {label: Side, phi_obs_deg: -70, B0_deg: 0}     # explicit angles, OR…
  - {label: sub-Earth, obs_time: '2024-03-30T20:49:00'}   # sub-Earth L0/B0 (MAS)
white_light: {Rocc: 1.0, U_LIMB: 0.63, SIGMA_T: 7.95e-26, nrgf: false}
los: {probes: [[1.5, 0.3]]}
output_dir: out/my_run     # where the PNGs are written
```

Full HDF5 format spec: [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).
API & method: [`docs/API.md`](docs/API.md), [`docs/THEORY.md`](docs/THEORY.md).

---

## 5. Tips & gotchas

- **Always read the printed `cube.summary()`** at the top of a run — check the
  detected `case`, `fmt`, and the `ne` / `T` ranges *before* trusting the image.
  A wrong `ne_scale` or axis guess shows up there first.
- **RMAX / r_range**: `Rmax` is the image window; `r_range` is how much of the
  cube is used. For an on-disk AIA look use `Rmax ≈ 1.5`; for white light use
  `Rmax ≈ 5`. Anything past your data's outer radius is just empty (black).
- **Raw EM looks like a saturated blob?** That's `∫ nₑ²` dominated by the dense
  base. A real AIA image (with a `response:` and a temperature field) weights by
  `R_λ(T)` and shows loops instead. Keeping `r_range: [1.03, …]` also helps.
- **zsh comments**: never paste a line containing `#` — run `setopt
  interactive_comments` once if you want them, or just omit comments.
- **First call is slow (~5–10 s)**: numba is JIT-compiling; later calls are fast.

---

## License

[MIT](LICENSE) © Anshika Singh. If you use this, please cite it (`CITATION.cff`).
