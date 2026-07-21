"""
Observer geometry from an observation time (sub-Earth L0/B0 via sunpy) and the
config view resolver.
"""
import numpy as np
import pytest

from rrt.observer import view_angles, make_observer


def test_view_angles_explicit():
    assert view_angles({"phi_obs_deg": -70, "B0_deg": 12}) == (-70.0, 12.0)


def test_view_angles_obs_time():
    sunpy = pytest.importorskip("sunpy")  # core dep, but skip cleanly if absent
    phi, b0 = view_angles({"obs_time": "2024-05-08T14:09:00"})
    # sub-Earth values for this UTC instant (deterministic, no network)
    assert phi == pytest.approx(332.23, abs=0.5)
    assert b0 == pytest.approx(-3.35, abs=0.3)


def test_view_angles_phi_offset():
    pytest.importorskip("sunpy")
    base, _ = view_angles({"obs_time": "2024-05-08T14:09:00"})
    off, _ = view_angles({"obs_time": "2024-05-08T14:09:00", "phi_offset_deg": 10})
    assert off == pytest.approx(base + 10.0, abs=1e-6)


def test_time_observer_is_valid_basis():
    pytest.importorskip("sunpy")
    phi, b0 = view_angles({"obs_time": "2024-05-08T14:09:00"})
    n_obs, e_los, x_img, y_img = make_observer(phi, b0)
    # orthonormal image basis, e_los = -n_obs
    for v in (n_obs, x_img, y_img):
        assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(e_los, -n_obs)
    assert abs(np.dot(x_img, y_img)) < 1e-9
    assert abs(np.dot(x_img, n_obs)) < 1e-9
