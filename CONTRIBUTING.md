# Contributing to `rrt`

Thanks for helping improve the synthetic-observable pipeline. This is a small
research package; the bar is *correctness first*, then clarity.

## Development setup

```bash
git clone <your-fork-url> rrt && cd rrt
conda env create -f environment.yml && conda activate rrt   # or a venv (see README)
pip install -e ".[test]"
pytest        # runs entirely on generated synthetic data — no proprietary files
```

Tested on Python 3.11; CI runs 3.10–3.12.

## The one rule that matters: preserve the validated numerics

The geometry core (`rrt.geometry`) and the two integrator kernels
(`rrt.euv.siddon_integrate`, `rrt.wl.siddon_integrate_pB`) are **validated**. If
you refactor any of them:

1. Prove correctness against the **defining equation**: each intersection
   back-substitutes into its surface to ~machine zero
   (`tests/test_geometry_intersections.py`). Keep those residual tests green.
2. The LOS diagnostic and the integrators must agree on **cell assignment**:
   digitise into EDGES with `searchsorted(edges, x, "right") - 1`. Do not change
   this on one side only — `tests/test_los_integrator_agreement.py` guards it
   (the LOS Σ must reproduce `run_siddon` to `rtol=1e-9`).
3. If you change a result on purpose, snapshot the old output and add a
   regression test pinning the new numbers.

A change to the numerics without a ground-truth/regression test will not be
merged.

## Style & scope

- Prefer clear, documented functions over notebook-embedded logic. Notebooks
  import from the package; they do not redefine it.
- The shared geometry lives in **one** place (`rrt.geometry`). Don't reintroduce a
  second copy in `euv` / `wl` / `los`.
- New data quirks belong in `rrt.io.load_mhd` behind an auto-detect + override,
  with a test in `tests/test_loader_cases.py` using a `rrt.synthdata` fixture.
- Large/binary data is never committed (see `.gitignore`); add a synthetic
  generator instead.

## Adding a test

All fixtures generate data in-memory or into `tmp_path` via `rrt.synthdata`. Add
new domains/cases there rather than shipping files. Run `pytest -q` before opening
a PR; the suite is fast (~15 s).

## Reporting issues

Include: the loader `cube.meta` (or `cube.summary()`), the config/call you used,
and — if a numerical result looks wrong — a minimal `rrt.synthdata` cube that
reproduces it.
