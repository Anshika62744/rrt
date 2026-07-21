"""
Unified-loader tests: the three input cases and every auto-detection, exercised
against synthetic HDF5 files written by rrt.synthdata (ARMS and MAS layouts).
"""
import h5py
import numpy as np
import pytest

from rrt import synthdata
from rrt.io import load_mhd, detect_format


@pytest.fixture
def cube_arrays():
    r, th, ph = synthdata.spherical_grid("full", nr=8, nth=10, nph=12, r_range=(1.0, 3.0))
    ne = synthdata.analytic_ne(r, th, ph, n0=1.0e8)
    T = synthdata.analytic_T(r, th, ph, t0=1.5e6)
    return r, th, ph, ne, T


# ── the three input cases ────────────────────────────────────────────────────

def test_case_single_file(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "single.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T)
    cube = load_mhd(str(p))
    assert cube.meta["input_case"] == "single"
    assert cube.ne.shape == ne.shape
    np.testing.assert_allclose(cube.ne, ne, rtol=1e-4)
    np.testing.assert_allclose(cube.T, T, rtol=1e-3)


def test_case_separate_files(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    dp, tp = tmp_path / "rho.h5", tmp_path / "t.h5"
    synthdata.write_arms(dp, r, th, ph, ne)              # density only
    synthdata.write_arms(tp, r, th, ph, np.ones_like(ne), T=T)  # T lives in vars/T
    cube = load_mhd(str(dp), temperature=str(tp))
    assert cube.meta["input_case"] == "separate"
    np.testing.assert_allclose(cube.T, T, rtol=1e-3)


def test_case_density_only_isothermal(tmp_path, cube_arrays):
    r, th, ph, ne, _ = cube_arrays
    p = tmp_path / "rho_only.h5"
    synthdata.write_arms(p, r, th, ph, ne)
    cube = load_mhd(str(p), t_iso=2.0e6)
    assert cube.meta["input_case"] == "density_only"
    assert np.allclose(cube.T, 2.0e6)


# ── auto-detection features ──────────────────────────────────────────────────

def test_ghost_strip(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "ghosts.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T, nghost=2)
    cube = load_mhd(str(p))
    assert cube.meta["nghost"] == 2
    assert cube.r.size == r.size            # ghosts removed
    np.testing.assert_allclose(cube.ne, ne, rtol=1e-4)


def test_ghost_override(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "ghosts.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T, nghost=2)
    cube = load_mhd(str(p), nghost=0)       # force: keep the ghost padding
    assert cube.r.size == r.size + 4


def test_density_mass_to_number(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "mass.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T, ne_is_mass=True)  # stored g/cm^3
    cube = load_mhd(str(p))
    assert cube.meta["density"] == "mass→number"
    np.testing.assert_allclose(cube.ne, ne, rtol=1e-4)


@pytest.mark.parametrize("units", ["MK", "log10", "K"])
def test_temperature_units(tmp_path, cube_arrays, units):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / f"T_{units}.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T, temperature_units=units)
    cube = load_mhd(str(p))
    np.testing.assert_allclose(cube.T, T, rtol=1e-3)


def test_angles_deg_to_rad(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "deg.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T, angle_units="deg")
    cube = load_mhd(str(p))
    assert cube.theta.max() <= np.pi + 1e-6
    np.testing.assert_allclose(cube.theta.max(), th.max(), rtol=1e-5)


def test_axis_order_gives_rtp(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "order.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T)
    cube = load_mhd(str(p))
    assert cube.meta["axis_perm"] == (2, 1, 0)
    # analytic ne ∝ r^-2 → strictly decreasing along axis 0 (r)
    prof = cube.ne[:, 0, 0]
    assert np.all(np.diff(prof) < 0)


def test_axis_order_fallback_when_ambiguous(tmp_path):
    """No axis_order attr + equal dims → (2,1,0) fallback with a recorded warning."""
    r, th, ph = synthdata.spherical_grid("wedge", nr=9, nth=9, nph=9)
    ne = synthdata.analytic_ne(r, th, ph)
    p = tmp_path / "ambig.h5"
    synthdata.write_arms(p, r, th, ph, ne)
    with h5py.File(p, "r+") as f:            # strip the axis_order hints
        del f.attrs["axis_order"]
        del f["vars/rho"].attrs["axis_order"]
    cube = load_mhd(str(p))
    assert cube.meta["axis_source"] == "(2,1,0) fallback"
    assert any("ambiguous" in w for w in cube.meta.get("warnings", []))


def test_mas_format(tmp_path):
    r, th, ph = synthdata.spherical_grid("full", nr=8, nth=10, nph=12, r_range=(1.0, 10.0))
    ne = synthdata.analytic_ne(r, th, ph, n0=1.0e8)
    p = tmp_path / "mas.h5"
    synthdata.write_mas(p, r, th, ph, ne, ne_scale=1.0e8)   # code units
    assert detect_format(str(p)) == "mas"
    cube = load_mhd(str(p), ne_scale=1.0e8, t_iso=1.2e6)     # undo code units
    assert cube.meta["fmt"] == "mas"
    assert cube.ne.shape == (r.size, th.size, ph.size)
    np.testing.assert_allclose(cube.ne, ne, rtol=1e-6)
    prof = cube.ne[:, 0, 0]
    assert np.all(np.diff(prof) < 0)


def test_radial_clip(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "clip.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T)
    cube = load_mhd(str(p), r_range=(1.5, 2.5))
    assert cube.r.min() >= 1.5 and cube.r.max() <= 2.5
    assert cube.ne.shape[0] == cube.r.size


def test_summary_runs(tmp_path, cube_arrays):
    r, th, ph, ne, T = cube_arrays
    p = tmp_path / "s.h5"
    synthdata.write_arms(p, r, th, ph, ne, T=T)
    s = load_mhd(str(p)).summary()
    assert "MHDCube" in s and "ne" in s
