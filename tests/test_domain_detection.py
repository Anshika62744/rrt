"""
Domain-detection tests for prepare_domain across wedge / polar-cap / full-shell.

Verifies the flags each geometry triggers and the cross-surface reductions the
integrators rely on: a full φ grid snaps the seam shut and drops the duplicate
half-plane; θ edges at 0/π (degenerate cones) are dropped from theta_cross.
"""
import numpy as np

from rrt.geometry import prepare_domain
from rrt import synthdata


def _dom(domain, **kw):
    r, th, ph = synthdata.spherical_grid(domain, **kw)
    return prepare_domain(r, th, ph)


def test_wedge_is_partial():
    d = _dom("wedge")
    assert d["full_theta"] is False
    assert d["full_phi"] is False
    # a wedge keeps every edge as a splitting surface
    assert len(d["phi_cross"]) == len(d["phi_edges"])


def test_polar_cap():
    d = _dom("polar_cap")
    assert d["full_theta"] is False   # theta only 2°–40°
    assert d["full_phi"] is True      # phi spans full 2π
    assert np.isclose(d["phi_span"], 2.0 * np.pi)
    # periodic phi → duplicate seam half-plane dropped
    assert len(d["phi_cross"]) == len(d["phi_edges"]) - 1


def test_full_shell():
    d = _dom("full")
    assert d["full_theta"] is True
    assert d["full_phi"] is True
    assert np.isclose(d["phi_span"], 2.0 * np.pi)
    # theta edges reach 0 and pi
    assert d["theta_edges"][0] <= 1e-6
    assert d["theta_edges"][-1] >= np.pi - 1e-6
    # the two degenerate polar cones (sinθ≈0) are dropped from theta_cross
    assert len(d["theta_cross"]) == len(d["theta_edges"]) - 2


def test_full_phi_seam_snapped():
    d = _dom("full")
    span = d["phi_edges"][-1] - d["phi_edges"][0]
    assert np.isclose(span, 2.0 * np.pi)


def test_phi_must_increase():
    # A descending phi (not fixed by the caller) must raise a clear error.
    r, th, ph = synthdata.spherical_grid("wedge")
    import pytest
    with pytest.raises(ValueError, match="phi centres must increase"):
        prepare_domain(r, th, ph[::-1])


def test_edges_bracket_centres():
    r, th, ph = synthdata.spherical_grid("full")
    d = prepare_domain(r, th, ph)
    assert d["r_edges"][0] <= r[0] <= d["r_edges"][-1]
    assert d["r_edges"][0] >= 0.0  # radial floor clamp
