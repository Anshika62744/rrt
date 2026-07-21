"""
Ground-truth (residual) tests for the boundary-intersection helpers.

Each returned parameter ``s`` is back-substituted into the surface it is meant to
lie on; the residual must be ~machine zero. This is the defining-equation proof
the plan calls for: if the algebra were wrong, the residual would be O(1).
"""
import numpy as np

from rrt.geometry import (
    _radial_intersections,
    _theta_intersections,
    _phi_intersections,
    make_edges,
)

# A generic, non-degenerate ray: origin r0 and unit direction e.
R0 = np.array([0.3, -1.2, 0.7])
E = np.array([0.2, 0.9, -0.35])
E = E / np.linalg.norm(E)


def _point(s):
    return R0 + s * E


def test_radial_residual():
    shells = np.array([0.5, 1.0, 1.7, 2.5])
    out = _radial_intersections(*R0, *E, shells)
    assert len(out) > 0
    for s in out:
        r = np.linalg.norm(_point(s))
        # s solves |r(s)| = one of the shells
        assert np.min(np.abs(r - shells)) < 1e-9


def test_theta_residual():
    thetas = np.array([0.6, 1.1, 2.0])
    out = _theta_intersections(*R0, *E, thetas)
    assert len(out) > 0
    for s in out:
        p = _point(s)
        r = np.linalg.norm(p)
        cos_th = p[2] / r
        # s solves z² = cos²θ · r²  ⇔  |cosθ_point| == cos(one bound)
        assert np.min(np.abs(np.abs(cos_th) - np.abs(np.cos(thetas)))) < 1e-9


def test_phi_residual():
    phis = np.array([-0.4, 0.7, 1.9])
    out = _phi_intersections(*R0, *E, phis)
    assert len(out) > 0
    for s in out:
        p = _point(s)
        # s solves the half-plane y·cosφ = x·sinφ for one φ bound
        res = np.abs(p[1] * np.cos(phis) - p[0] * np.sin(phis))
        assert np.min(res) < 1e-9


def test_radial_count_for_crossing_ray():
    # A ray that passes inside a shell crosses it exactly twice (near & far).
    r0 = np.array([0.0, 0.0, 0.0]); e = np.array([1.0, 0.0, 0.0])
    out = _radial_intersections(*r0, *e, np.array([2.0]))
    assert len(out) == 2
    assert np.allclose(sorted(out), [-2.0, 2.0])


def test_make_edges_basic():
    c = np.array([1.0, 2.0, 3.0, 4.0])
    e = make_edges(c)
    # n+1 edges, monotonic, centres bracketed
    assert e.size == c.size + 1
    assert np.all(np.diff(e) > 0)
    assert np.allclose(e, [0.5, 1.5, 2.5, 3.5, 4.5])


def test_make_edges_nonuniform():
    c = np.array([1.0, 1.5, 3.0])
    e = make_edges(c)
    assert e.size == 4
    assert np.all(np.diff(e) > 0)
    # interior edges are midpoints
    assert np.isclose(e[1], 1.25)
    assert np.isclose(e[2], 2.25)
