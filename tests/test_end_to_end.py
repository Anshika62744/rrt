"""
End-to-end regression on synthetic data:

* EUV and white-light renders produce sane, positive maps;
* the three Domain_Check correctness checks (φ-decomposition, photospheric-clip,
  azimuthal-symmetry) hold as assertions; and
* the rrt-euv / rrt-wl / rrt-los console drivers run from a config and write files.
"""
import numpy as np
import pytest

from rrt import synthdata
from rrt.euv import run_siddon
from rrt.wl import run_siddon_pB
from rrt.response import FLAT_T, FLAT_R
from rrt.observer import image_grid   # returns (Xg, Yg, Rperp)

from conftest import observer         # returns (x_img, y_img, e_los)


def _em(Xg, Yg, view, r, th, ph, ne, T, npix):
    x_img, y_img, e_los = view
    return run_siddon(Xg, Yg, x_img, y_img, e_los, r, th, ph, ne, T,
                      FLAT_T, FLAT_R, Npix=npix, Rmax=3.0, plot=False, save=False)


@pytest.fixture(scope="module")
def view():
    return observer(-70.0, 0.0)       # (x_img, y_img, e_los)


@pytest.fixture(scope="module")
def full():
    r, th, ph = synthdata.spherical_grid("full", nr=10, nth=12, nph=16, r_range=(1.0, 3.0))
    ne = synthdata.analytic_ne(r, th, ph, n0=1.0e8, aniso=True)
    T = synthdata.analytic_T(r, th, ph, t0=1.5e6)
    return r, th, ph, ne, T


# ── basic sanity ─────────────────────────────────────────────────────────────

def test_euv_map_is_positive(full, view):
    r, th, ph, ne, T = full
    Xg, Yg = image_grid(3.0, 40)[:2]
    em = _em(Xg, Yg, view, r, th, ph, ne, T, 40)
    assert np.isfinite(em).all()
    assert (em > 0).sum() > 0


def test_wl_offlimb_positive(full, view):
    r, th, ph, ne, T = full
    Xg, Yg, Rperp = image_grid(3.0, 40)
    x_img, y_img, e_los = view
    pb = run_siddon_pB(Xg, Yg, x_img, y_img, e_los, r, th, ph, ne,
                       Npix=40, Rmax=3.0, plot=False, save=False)
    assert (pb[Rperp > 1.05] > 0).sum() > 0


# ── Check 1 — φ-decomposition: sum of contiguous φ sub-wedges == whole ───────

def test_phi_decomposition(full, view):
    r, th, ph, ne, T = full
    Xg, Yg = image_grid(3.0, 48)[:2]
    whole = _em(Xg, Yg, view, r, th, ph, ne, T, 48)
    nph = ph.size
    parts = np.zeros_like(whole)
    for k in range(4):
        s = slice(k * nph // 4, (k + 1) * nph // 4)
        parts += _em(Xg, Yg, view, r, th, ph[s],
                     np.ascontiguousarray(ne[:, :, s]),
                     np.ascontiguousarray(T[:, :, s]), 48)
    denom = np.abs(whole).sum()
    rel = np.abs(parts - whole).sum() / denom
    assert rel < 2e-2, f"phi decomposition rel.diff={rel:.2e}"


# ── Check 2 — photospheric clip: EM jumps UP crossing the limb ───────────────

def test_photospheric_clip_limb_jump(full, view):
    r, th, ph, ne, T = full
    Xg, Yg, Rperp = image_grid(1.5, 64)
    em = _em(Xg, Yg, view, r, th, ph, ne, T, 64)
    ring_in = em[(Rperp > 0.85) & (Rperp < 0.99)]
    ring_out = em[(Rperp > 1.01) & (Rperp < 1.15)]
    ring_in = ring_in[ring_in > 0]
    ring_out = ring_out[ring_out > 0]
    assert ring_in.size and ring_out.size
    assert ring_out.mean() / ring_in.mean() > 1.0


# ── Check 3 — azimuthal symmetry: pB constant on a circle of fixed b ─────────

def test_azimuthal_symmetry(full, view):
    r, th, ph, ne, T = full
    x_img, y_img, e_los = view
    ne_sph = np.broadcast_to(ne.mean(axis=(1, 2))[:, None, None], ne.shape).copy()
    pa = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    for b in (1.5, 2.0):
        Xb = (b * np.cos(pa)).reshape(6, 6)
        Yb = (b * np.sin(pa)).reshape(6, 6)
        img = run_siddon_pB(Xb, Yb, x_img, y_img, e_los, r, th, ph, ne_sph,
                            Npix=6, Rmax=3.0, plot=False, save=False)
        v = img.ravel()
        spread = v.std() / v.mean()
        assert spread < 5e-3, f"b={b}: rel.spread={spread:.2e}"


# ── CLI drivers run from a config and write files ────────────────────────────

@pytest.fixture
def wedge_files(tmp_path):
    r, th, ph = synthdata.spherical_grid("wedge", nr=8, nth=10, nph=12, r_range=(1.0, 3.0))
    ne = synthdata.analytic_ne(r, th, ph, n0=1.0e8)
    dpath = tmp_path / "rho.h5"
    synthdata.write_arms(dpath, r, th, ph, ne)          # density-only ARMS
    rpath = tmp_path / "resp.npz"
    synthdata.write_response(rpath, channels=(171,))
    return tmp_path, dpath, rpath


def _write_yaml(path, text):
    path.write_text(text)
    return str(path)


def test_cli_euv(wedge_files):
    tmp, dpath, rpath = wedge_files
    from rrt.cli import main_euv
    cfg = _write_yaml(tmp / "euv.yaml", f"""
data: {{density: {dpath}, input_case: auto, t_iso: 1.0e6}}
response: {rpath}
wavelength: 171
image: {{Npix: 24, Rmax: 3.0}}
views:
  - {{label: Side, phi_obs_deg: -70, B0_deg: 0}}
output_dir: {tmp / 'out'}
""")
    assert main_euv([cfg]) == 0
    outs = list((tmp / "out").glob("euv_*.png"))
    assert len(outs) == 1


def test_cli_wl(wedge_files):
    tmp, dpath, rpath = wedge_files
    from rrt.cli import main_wl
    cfg = _write_yaml(tmp / "wl.yaml", f"""
data: {{density: {dpath}, input_case: auto}}
image: {{Npix: 24, Rmax: 3.0}}
views:
  - {{label: Side, phi_obs_deg: -70, B0_deg: 0}}
white_light: {{Rocc: 1.0, nrgf: false}}
output_dir: {tmp / 'outwl'}
""")
    assert main_wl([cfg]) == 0
    assert list((tmp / "outwl").glob("wl_pb_*.png"))


def test_cli_los(wedge_files):
    tmp, dpath, rpath = wedge_files
    from rrt.cli import main_los
    cfg = _write_yaml(tmp / "los.yaml", f"""
data: {{density: {dpath}, input_case: auto, t_iso: 1.0e6}}
response: {rpath}
wavelength: 171
image: {{Npix: 24, Rmax: 3.0}}
views:
  - {{label: Side, phi_obs_deg: -70, B0_deg: 0}}
los: {{probes: [[1.3, 0.0]]}}
output_dir: {tmp / 'outlos'}
""")
    assert main_los([cfg]) == 0
    assert list((tmp / "outlos").glob("los_*.png"))
