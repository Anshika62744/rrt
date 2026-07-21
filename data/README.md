# Data

This directory holds the inputs the forward model integrates over. The large
MHD cubes are **not** tracked in git (see `.gitignore`) — they are too big for
a GitHub repo and should be distributed as a [GitHub Release asset], a
[Zenodo record] (recommended — gives a citable DOI for the paper), or via
[Git LFS]. Drop them in this folder before running the notebook.

## Files used by the model

| File | Tracked in git? | Description |
|------|:---------------:|-------------|
| `aia_temp_response_chiantifix.npz` | ✅ yes (small) | SDO/AIA temperature response tables `T_resp`, `R_resp` (CHIANTI-corrected). This is the response table used throughout the notebook. |
| `rho.h5`                       | ❌ no | 3D electron-density cube from the MHD model. |
| `t.h5`                           | ❌ no | 3D temperature cube from the MHD model. |

## Expected HDF5 layout (`rho.h5`, `t.h5`)

Each cube is read in `EUV_Synthetic.ipynb` with these dataset keys:

| Key    | Shape        | Meaning |
|--------|--------------|---------|
| `Data` | `(nphi, ntheta, nr)` | the field (density or temperature) |
| `dim1` | `(nphi,)`    | azimuth φ [radians] |
| `dim2` | `(ntheta,)`  | polar angle θ [radians] |
| `dim3` | `(nr,)`      | heliocentric radius r [R_sun] |

The notebook transposes `Data` to `(nr, ntheta, nphi)` and applies unit
conversions (`NE_UNIT`, MK→K). Adjust `DATA_KEY` / `PHI_KEY` / `THETA_KEY` /
`R_KEY` and the unit factors in the "File paths" cell if your cubes differ.

## AIA response table (`*.npz`)

Loaded as a NumPy `.npz` archive providing the temperature grid and the
per-channel response curve used for `R(T)` interpolation during EUV synthesis.


## Test Data for trial run : https://drive.google.com/drive/folders/1BFf-Grq2p3pRjBgORyAR9gTpcW-48ri4?usp=sharing
## Where to get the data

The 3D MHD cubes come from the **Predictive Science Inc. (PSI) MAS model**
runs, distributed through the PSI MHDweb data portal:

**https://www.predsci.com/mhdweb/data_access.php**

Browse to the relevant Carrington rotation / model run and download the
density and temperature cubes:

| PSI field | Save as | Used for |
|-----------|---------|----------|
| `rho` (electron density) | `data/rho.h5` | `NE_PATH` in the notebook |
| `t` (temperature)        | `data/t.h5`   | `TEMP_PATH` in the notebook |

Place both files in this `data/` directory. If your download has different
filenames, either rename them to match the table above or update `NE_PATH` /
`TEMP_PATH` in the "File paths" cell of `EUV_Synthetic.ipynb`.

[GitHub Release asset]: https://docs.github.com/en/repositories/releasing-projects-on-github
[Zenodo record]: https://zenodo.org/
[Git LFS]: https://git-lfs.com/
