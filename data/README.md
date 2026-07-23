# Data

This directory holds the inputs the forward model integrates over. The MHD cubes
are **not** tracked in git (see `.gitignore`) — they are far too large for a
GitHub repo — so download them and drop them in this folder before running the
example configs or the notebooks. Only the small AIA response table ships with
the repo.

| File | Tracked? | Size | md5 | Used by |
|---|:---:|---:|---|---|
| `aia_temp_response_chiantifix.npz` | ✅ | 8 KB | — | every EUV / LOS config |
| `rho_bin.h5` | ❌ | 35 MB | `eb59f3b644312ac231d6a12a8e208a15` | case 1 |
| `t_bin.h5` | ❌ | 35 MB | `190586bd36999b2c6fe22d209ac5e8ad` | case 1 |
| `mhd_data_0220.h5` | ❌ | 3.2 GB | `8197c456bcfef4faf30ac825be939eae` | case 2 |
| `rhoC_02.h5` | ❌ | 42 MB | `a6d7d9876544944efd436b07cea1820a` | case 3 |
| `t02.h5` | ❌ | 42 MB | `659caa6ac5621cf3321c23340b3a2e47` | case 3 |

**Keep the filenames exactly as listed** — the configs in `examples/configs/`
reference them by these paths (`data/rho_bin.h5`, …). If you rename a download,
update the `data:` block of the configs that use it.

## Download

| Case | Files | Data | Link |
|---|---|---|---|
| **case 1** | `rho_bin.h5` + `t_bin.h5` | MAS/PSI full shell, density + temperature in code units | _(link to be added)_ |
| **case 2** | `mhd_data_0220.h5` | ARMS 90° wedge, density only (→ isothermal) | _(link to be added)_ |
| **case 3** | `rhoC_02.h5` + `t02.h5` | MAS/PSI full shell, second snapshot | <https://drive.google.com/drive/folders/1BFf-Grq2p3pRjBgORyAR9gTpcW-48ri4?usp=share_link> |

Google Drive shows a virus-scan interstitial for files over ~100 MB, so a plain
`curl`/`wget` on a share link returns an HTML page instead of the file. Either
download through a browser, or use [`gdown`](https://github.com/wkentaro/gdown),
which handles the confirmation token:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1BFf-Grq2p3pRjBgORyAR9gTpcW-48ri4" -O data/
```

Verify a download before trusting an image — a truncated cube produces a
plausible but wrong figure rather than an error:

```bash
md5 data/rhoC_02.h5        # macOS      (Linux: md5sum)
```

## What each case runs

Once the files are in place, from the repo root:

```bash
rrt-euv examples/configs/case1_euv.yaml     # also case1_wl / case1_los
rrt-euv examples/configs/case2_euv.yaml     # wedge, density-only
rrt-euv examples/configs/case3_euv.yaml
```

The notebooks in `examples/notebooks/` read the same configs — set
`CASE = 'case1' | 'case2' | 'case3'` in the first code cell.

No data at all? `python examples/generate_synthetic_data.py` writes small
synthetic cubes into `examples/data/`, and the `*_synth.yaml` configs run the
whole pipeline on those.

## Expected HDF5 layouts

Both layouts are auto-detected by `rrt.io.load_mhd` (`fmt: auto`); the `fmt:` key
in a config only forces the choice.

**MAS / PSI** (cases 1 and 3) — one field per file:

| Key | Shape | Meaning |
|---|---|---|
| `Data` | `(nphi, ntheta, nr)` | the field (density or temperature) |
| `dim1` | `(nphi,)` | azimuth φ [rad] |
| `dim2` | `(ntheta,)` | polar angle θ [rad] |
| `dim3` | `(nr,)` | heliocentric radius r [R_sun] |

Values are in code units; `ne_scale` and `t_scale` in the config convert them
(`1.0e+8` cm⁻³ and `2.807e+7` K for these runs). The loader transposes `Data` to
`(nr, ntheta, nphi)` and forces every axis ascending.

**ARMS** (case 2) — coordinates and variables in one file:

| Key | Shape | Meaning |
|---|---|---|
| `coords/r`, `coords/theta`, `coords/phi` | 1-D | grid axes (ghost cells included) |
| `vars/rho` | `(nphi, ntheta, nr)` | density [cm⁻³], with an `axis_order` attribute |

Ghost cells are read from the file attributes (`nghost`) and trimmed. There is no
temperature field, so the config supplies `t_iso`.

Full spec, including the unit/axis auto-detection rules and every override:
[`../docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md).

## Where the data comes from

The MAS cubes are **Predictive Science Inc.** model runs, distributed through the
PSI MHDweb portal: <https://www.predsci.com/mhdweb/data_access.php>. The ARMS
wedge is an adaptively refined MHD snapshot interpolated onto a spherical grid.

## AIA response table

`aia_temp_response_chiantifix.npz` is a NumPy archive with a `logte` grid and one
`resp{λ}` curve per channel (94, 131, 171, 193, 211, 304, 335), giving
DN cm⁵ s⁻¹ pixel⁻¹. `rrt.response.load_response` interpolates log₁₀R vs log₁₀T.
