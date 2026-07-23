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
1. [Install](#1-install)
2. [Get the data](#2-get-the-data)
3. [Run a test case](#3-run-a-test-case)
4. [The notebooks](#4-the-notebooks)
5. [Run on YOUR data](#5-run-on-your-data)
6. [Config reference](#6-config-reference)
7. [Tips & gotchas](#7-tips--gotchas)

---

## 1. Install

You need **conda** (for Python + the scientific libraries) and **git** (to get
the code). If you already have conda, skip to 1.3.

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
sunpy, astropy and jupyter, and installs this package in editable mode — so the
`rrt-euv` / `rrt-wl` / `rrt-los` commands become available. You only run
`conda activate rrt` **once per terminal window**.

Prefer pip? From the repo root, in a fresh virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
```

### 1.4 Check the install

```bash
rrt-euv --help
python -c "import rrt; print(rrt.__file__)"
```

The path printed must be the `src/rrt/__init__.py` **inside this repo**. If it
points somewhere else, another copy of `rrt` is installed and shadowing this one
— re-run `pip install -e .` from the repo root.

> ⚠️ Do **not** paste a command that has a `#` comment on it — zsh will treat the
> comment as arguments and error. Copy the plain command only.

---

## 2. Get the data

The MHD cubes are too large for git, so they live in a Google Drive folder:
**download links, file sizes and md5 checksums are in
[`data/README.md`](data/README.md)**.

Put the files in this repo's `data/` folder, keeping their names exactly:

```
data/
  aia_temp_response_chiantifix.npz   ← ships with the repo
  rho_bin.h5   t_bin.h5              ← case 1   (35 MB each)
  mhd_data_0220.h5                   ← case 2   (3.2 GB)
  rhoC_02.h5   t02.h5                ← case 3   (42 MB each)
```

The configs reference these paths, so a renamed download (`rho_bin (1).h5`) will
fail with `FileNotFoundError`. You only need the files for the case you want to
run — the cases are independent.

| Case | Data | Kind |
|---|---|---|
| **case 1** | `rho_bin.h5` + `t_bin.h5` | MAS/PSI full shell, density **and** temperature (code units) |
| **case 2** | `mhd_data_0220.h5` | ARMS 90° wedge, density only → isothermal |
| **case 3** | `rhoC_02.h5` + `t02.h5` | MAS/PSI full shell, a second snapshot |

---

## 3. Run a test case

One command per product per case, all from the repo root:

| | case 1 | case 2 | case 3 |
|---|---|---|---|
| **EUV / AIA** | `rrt-euv examples/configs/case1_euv.yaml` | `rrt-euv examples/configs/case2_euv.yaml` | `rrt-euv examples/configs/case3_euv.yaml` |
| **White light** | `rrt-wl examples/configs/case1_wl.yaml` | `rrt-wl examples/configs/case2_wl.yaml` | `rrt-wl examples/configs/case3_wl.yaml` |
| **LOS** | `rrt-los examples/configs/case1_los.yaml` | `rrt-los examples/configs/case2_los.yaml` | `rrt-los examples/configs/case3_los.yaml` |

Each command prints the loaded `cube.summary()`, then a line per view, and writes
PNGs into the config's `output_dir` (`out/case1_euv/`, `out/case2_wl/`, …). Open
one:

```bash
open out/case1_euv/euv_171_earth.png     # macOS  (Linux: xdg-open)
```

**The command chooses the product, not the filename.** `rrt-euv` on a `_wl`
config will not complain — it just renders a meaningless EUV map, because every
key it needs has a default. Match the command to the config.

### All AIA channels in one figure

[`examples/configs/case1_euv_channels.yaml`](examples/configs/case1_euv_channels.yaml)
renders 94, 131, 171, 193, 211 and 335 Å as a single grid figure instead of six
separate PNGs:

```bash
rrt-euv examples/configs/case1_euv_channels.yaml   # → out/case1_euv_channels/
```

The only difference from a normal EUV config is the plural key —
`wavelengths: [94, 131, …]` instead of `wavelength: 171` — plus a `panel:` block
for the layout and per-channel colorbar limits. Any EUV config can be switched
this way.

### What the LOS diagnostics are for

`rrt-los` doesn't make a picture of the Sun — it **verifies the ray tracer**. For
each probe pixel it walks a single line of sight and produces two figures:

- a **profile**: `r(s)`, segment length `Δs`, `nₑ`, `T` and the per-segment
  contribution along the ray, with the total Σ annotated;
- a **voxel proof**: the r/θ/φ staircase with the assigned cell shaded per
  segment, plus a **PASS/CHECK verdict** — PASS means every kept sample lies
  inside the cell the integrator actually read, and the geometry residuals
  (closest approach vs impact parameter, spherical↔Cartesian round-trip) are
  ~machine zero.

Probes are `[X, Y]` impact points on the sky in R☉. For a partial domain like
case 2's wedge, a probe must lie where the wedge actually is — probes that miss
it are reported and skipped.

---

## 4. The notebooks

[`examples/notebooks/`](examples/notebooks/) has the same three products with the
figures inline. Each loads the *same YAML the CLI does*, so notebook and command
line can't drift, and each switches datasets with one line:

```python
CASE = 'case1'      # 'case1' | 'case2' | 'case3'
```

| Notebook | Shows |
|---|---|
| [`01_euv_three_viewpoints.ipynb`](examples/notebooks/01_euv_three_viewpoints.ipynb) | AIA maps per viewpoint, including the exact pole-on `earth_top` view that YAML can only approximate |
| [`02_white_light_pB.ipynb`](examples/notebooks/02_white_light_pB.ipynb) | Raw pB next to the NRGF-filtered version |
| [`03_los_diagnostics.ipynb`](examples/notebooks/03_los_diagnostics.ipynb) | Per-probe profile + voxel proof, then a cross-check that the LOS sum reproduces `run_siddon` for the same pixel to ~machine precision |
| [`04_time_evolution_movie.ipynb`](examples/notebooks/04_time_evolution_movie.ipynb) | Frames → MP4/GIF via `rrt.viz.make_movie` |

Register the environment as a kernel once, then open the notebook and pick it:

```bash
conda activate rrt
python -m ipykernel install --user --name rrt --display-name "Python (rrt)"
jupyter lab examples/notebooks/01_euv_three_viewpoints.ipynb
```

Notebook figures are written alongside the CLI's with an `nb_` prefix, so the two
never overwrite each other.

---

## 5. Run on YOUR data

Start from the shipped config closest to your input shape and edit the `data:`
block — that's the only part that changes per dataset.

```bash
cp examples/configs/case1_euv.yaml my_run.yaml
```

**Two files, separate density + temperature** (MAS/PSI — like case 1 and 3):

```yaml
data:
  density: /full/path/to/rho.h5
  temperature: /full/path/to/t.h5
  fmt: mas
  ne_scale: 1.0e+8         # code-unit → cm^-3   (your normalization)
  t_scale: 2.807e+7        # code-unit → K       (your normalization)
  r_range: [1.03, 5.0]     # trim dense base / limit depth
```

**One file, density only** → isothermal (ARMS — like case 2):

```yaml
data:
  density: /full/path/to/mhd_data_0220.h5
  input_case: auto         # no temperature in file → isothermal
  t_iso: 1.0e+6            # isothermal coronal temperature [K]
```

**One file holding both:**

```yaml
data:
  density: /full/path/to/both.h5
  temperature: vars/T      # dataset name inside the same file (omit to auto-find)
```

Then run whichever product you want:

```bash
rrt-euv my_run.yaml        # needs response: + wavelength:
rrt-wl  my_run.yaml        # add a white_light: block; try Rmax: 5.0
rrt-los my_run.yaml        # add los: {probes: [[1.5, 0.3], [2.5, 0.0]]}
```

---

## 6. Config reference

Common keys:

```yaml
data:
  density: PATH            # required
  temperature: PATH|KEY    # another file, a dataset key, or omit → isothermal
  fmt: auto                # auto | arms | mas
  ne_scale: 1.0            # code units → cm^-3
  t_scale: 1.0             # code units → K
  r_range: [1.03, 5.0]     # radial trim [R_sun]
response: data/aia_temp_response_chiantifix.npz   # EUV & LOS only
wavelength: 171            # 94, 131, 171, 193, 211, 304, 335
image: {Npix: 400, Rmax: 1.5}
scale: log                 # log | linear
vlim: [1.0e-1, 1.0e+4]     # colorbar limits (linear values); omit → percentiles
views:
  - {label: Side, phi_obs_deg: -70, B0_deg: 0}
  - {label: earth, obs_time: '2024-03-30T20:49:00'}
white_light: {Rocc: 1.0, U_LIMB: 0.63, nrgf: false}
los: {probes: [[1.5, 0.3]]}
output_dir: out/my_run
```

**Every key, with defaults and the multi-channel / pole-view / white-light
options: [`docs/CONFIG.md`](docs/CONFIG.md).**

HDF5 layout spec: [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).
API & method: [`docs/API.md`](docs/API.md), [`docs/THEORY.md`](docs/THEORY.md).

---

## 7. Tips & gotchas

- **Always read the printed `cube.summary()`** at the top of a run — check the
  detected `case`, `fmt`, and the `ne` / `T` ranges *before* trusting the image.
  A wrong `ne_scale` or axis guess shows up there first.
- **The command picks the product.** A config has no idea which of the three it
  is for; running the wrong command produces a wrong figure, not an error.
- **RMAX / r_range**: `Rmax` is the image window; `r_range` is how much of the
  cube is used. For an on-disk AIA look use `Rmax ≈ 1.5`; for white light use
  `Rmax ≈ 5`. Anything past your data's outer radius is just empty (black).
- **Raw EM looks like a saturated blob?** That's `∫ nₑ²` dominated by the dense
  base. A real AIA image (with a `response:` and a temperature field) weights by
  `R_λ(T)` and shows loops instead. Keeping `r_range: [1.03, …]` also helps.
- **Colorbar limits are linear**, even on a log scale — use `vlim_log: [0.5, 4.0]`
  if you'd rather type exponents.
- **Raw white light looks like a featureless glow.** That's real: pB spans ~5
  decades over 1–6 R☉. Use `nrgf: true`, or `radial_power: 3` with `scale: linear`,
  to see streamers.
- **zsh comments**: never paste a line containing `#` — run `setopt
  interactive_comments` once if you want them, or just omit comments.
- **First call is slow (~5–10 s)**: numba is JIT-compiling; later calls are fast.

---

## License

[MIT](LICENSE) © Anshika Singh. If you use this, please cite it (`CITATION.cff`).
