"""
The LOS walker must agree with the production integrator.

Because rrt.los now (a) splits rays on the SAME shared geometry surfaces and
(b) interpolates R(T) log-log exactly like rrt.euv, the per-segment Σ along one
ray must reproduce run_siddon for that pixel. An off-limb pixel is used so the
photospheric clip is a no-op and the whole chord is integrated on both sides.
"""
import numpy as np
import pytest

from rrt.euv import run_siddon
from rrt.los import sample_along_los, plot_los_voxel_proof

# A curved response so the log-log R(T) path is genuinely exercised.
RESP_T = np.array([1.0e4, 1.0e6, 3.0e6, 1.0e7])
RESP_R = np.array([1.0e-30, 2.0e-27, 5.0e-28, 1.0e-28])

# Off-limb impact point (b > 1) that crosses a full-domain cube.
XP, YP = 1.5, 0.3


def _integrator_pixel(cube, view, T_resp, R_resp):
    x_img, y_img, e_los = view
    Xg = np.array([[XP]]); Yg = np.array([[YP]])
    img = run_siddon(Xg, Yg, x_img, y_img, e_los,
                     cube["r"], cube["theta"], cube["phi"],
                     cube["ne"], cube["T"], T_resp, R_resp,
                     Npix=1, Rmax=3.0, plot=False, save=False)
    return float(img[0, 0])


def test_los_matches_integrator_curved_response(full_cube, view):
    s = sample_along_los(XP, YP, *view,
                         full_cube["r"], full_cube["theta"], full_cube["phi"],
                         full_cube["ne"], full_cube["T"],
                         T_resp=RESP_T, R_resp=RESP_R)
    assert s["kept"].any()
    assert not s["on_disk"]                       # off-limb: no clip
    los_total = s["I_total"]
    integ = _integrator_pixel(full_cube, view, RESP_T, RESP_R)
    assert integ > 0
    np.testing.assert_allclose(los_total, integ, rtol=1e-9)


def test_los_matches_integrator_flat_em(full_cube, view, flat_response):
    T_resp, R_resp = flat_response
    s = sample_along_los(XP, YP, *view,
                         full_cube["r"], full_cube["theta"], full_cube["phi"],
                         full_cube["ne"], full_cube["T"],
                         T_resp=T_resp, R_resp=R_resp)
    integ = _integrator_pixel(full_cube, view, T_resp, R_resp)
    np.testing.assert_allclose(s["I_total"], integ, rtol=1e-9)


def test_voxel_proof_passes(full_cube, view):
    verdict = plot_los_voxel_proof(
        XP, YP, *view,
        full_cube["r"], full_cube["theta"], full_cube["phi"],
        full_cube["ne"], full_cube["T"], show=False)
    assert verdict["all_inside"]
    assert verdict["pass"]
    assert verdict["roundtrip_err"] < 1e-9


def test_los_zero_coverage_raises(wedge_cube, view):
    # A pixel far off the wedge contributes nothing → the voxel proof refuses.
    with pytest.raises(ValueError, match="0 kept samples"):
        plot_los_voxel_proof(
            8.0, 8.0, *view,
            wedge_cube["r"], wedge_cube["theta"], wedge_cube["phi"],
            wedge_cube["ne"], wedge_cube["T"], show=False)
