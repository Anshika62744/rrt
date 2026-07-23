# Contributing to `rrt`

Thanks for helping improve the synthetic-observable pipeline. This is a small
research package; the bar is *correctness first*, then clarity.

## Development setup

```bash
git clone <your-fork-url> rrt && cd rrt
conda env create -f environment.yml && conda activate rrt   # or a venv (see README)
pip install -e .
python examples/generate_synthetic_data.py                  # no proprietary files needed
rrt-euv examples/configs/euv_fulldisk_synth.yaml            # smoke check
```

Tested on Python 3.11; supports 3.10–3.12.

## The one rule that matters: preserve the validated numerics

The geometry core (`rrt.geometry`) and the two integrator kernels
(`rrt.euv.siddon_integrate`, `rrt.wl.siddon_integrate_pB`) are **validated**. If
you refactor any of them:

1. Prove correctness against the **defining equation**: each intersection must
   back-substitute into its surface to ~machine zero.
2. The LOS diagnostic and the integrators must agree on **cell assignment**:
   digitise into EDGES with `searchsorted(edges, x, "right") - 1`. Do not change
   this on one side only — the LOS Σ must reproduce `run_siddon` for the same
   pixel to ~machine precision. `examples/notebooks/03_los_diagnostics.ipynb`
   runs exactly that cross-check, and `plot_los_voxel_proof` returns a PASS/CHECK
   verdict plus the geometry residuals.
3. If you change a result on purpose, snapshot the old output and record the new
   numbers in the PR description.

A change to the numerics without evidence that it still reproduces the defining
equation will not be merged.

## Style & scope

- Prefer clear, documented functions over notebook-embedded logic. Notebooks
  import from the package; they do not redefine it.
- The shared geometry lives in **one** place (`rrt.geometry`). Don't reintroduce a
  second copy in `euv` / `wl` / `los`.
- New data quirks belong in `rrt.io.load_mhd` behind an auto-detect + override,
  demonstrated on a cube built with `rrt.synthdata`.
- Large/binary data is never committed (see `.gitignore`); add a synthetic
  generator instead.

## Demonstrating a change

`rrt.synthdata` builds cubes in-memory (wedge, polar cap, full shell, full disk;
density-only or with temperature), so a change can be shown to work without
shipping data files. Add new domains/cases there rather than attaching files, and
include the `cube.summary()` and the relevant figure or LOS verdict in the PR.

## Reporting issues

Include: the loader `cube.meta` (or `cube.summary()`), the config/call you used,
and — if a numerical result looks wrong — a minimal `rrt.synthdata` cube that
reproduces it.
