"""
rrt.geometry
────────────
The shared geometry core for every rrt integrator. Previously this code was
duplicated verbatim in the EUV and white-light modules and re-implemented a
third time inside the LOS diagnostics; it now lives here **once** and is imported
by ``rrt.euv``, ``rrt.wl`` and ``rrt.los``.

Contents
    _radial_intersections   ray × radial shells       (quadratic in s)
    _theta_intersections    ray × polar cones         (quadratic in s)
    _phi_intersections      ray × azimuthal planes     (linear in s)
    make_edges              cell centres → cell-boundary edges
    prepare_domain          domain detection (wedge / cap / shell / disk)

The three ``_*_intersections`` helpers are ``@njit`` so they can be called both
from the parallel Numba kernels (``rrt.euv``/``rrt.wl``) and from the pure-Python
LOS walker (``rrt.los``) — a single implementation with no drift.

Cell assignment convention (used by every consumer): digitise into EDGES with
``searchsorted(edges, x, side='right') - 1``. The LOS diagnostics and the
integrators MUST agree on this, so it is defined and exercised here.
"""

import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Radial, Polar and Azimuthal BOUNDARY INTERSECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@njit
def _radial_intersections(rx, ry, rz, ex, ey, ez, r_shells):
    """
    |r(s)|² = ri²  →  s² + b·s + c = 0
    Returns list of s-values where ray crosses each radial shell.
    """
    b   = 2.0*(rx*ex + ry*ey + rz*ez)
    rp2 = rx*rx + ry*ry + rz*rz
    out = []
    for ri in r_shells:
        disc = b*b - 4.0*(rp2 - ri*ri)
        if disc < 0:
            continue
        sq = disc**0.5
        out.append((-b - sq)*0.5)
        out.append((-b + sq)*0.5)
    return out


@njit
def _theta_intersections(rx, ry, rz, ex, ey, ez, theta_bounds):
    """
    Cone surface z(s)² = cos²(θ)|r(s)|²  →  quadratic in s.
    Returns list of s-values where ray crosses each polar cone.
    """
    rdote = rx*ex + ry*ey + rz*ez
    r2    = rx*rx + ry*ry + rz*rz

    out = []
    for th in theta_bounds:
        cos2 = np.cos(th)**2
        a =  ez*ez - cos2
        b =  2.0*(rz*ez - cos2*rdote)
        c =  rz*rz - cos2*r2
        if abs(a) < 1e-14:
            if abs(b) > 1e-14:
                out.append(-c/b)
            continue
        disc = b*b - 4.0*a*c
        if disc < 0:
            continue
        sq = disc**0.5
        out.append((-b - sq)/(2.0*a))
        out.append((-b + sq)/(2.0*a))
    return out


@njit
def _phi_intersections(rx, ry, rz, ex, ey, ez, phi_bounds):
    """
    Half-plane y(s)cos(φ) = x(s)sin(φ)  →  linear in s.
    Returns list of s-values where ray crosses each azimuthal plane.
    """
    out = []
    for ph in phi_bounds:
        cosph = np.cos(ph)
        sinph = np.sin(ph)
        denom = ey*cosph - ex*sinph
        if abs(denom) < 1e-14:
            continue
        out.append((rx*sinph - ry*cosph)/denom)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: cell-centre → cell-edge conversion
# ─────────────────────────────────────────────────────────────────────────────

def make_edges(centres):
    """Convert 1-D array of cell centres to cell-boundary edges."""
    mid = 0.5*(centres[:-1] + centres[1:])
    lo  = centres[0]  - 0.5*(centres[1]  - centres[0])
    hi  = centres[-1] + 0.5*(centres[-1] - centres[-2])
    return np.concatenate([[lo], mid, [hi]])


def prepare_domain(r_ori, theta_ori, phi_ori, tol=1e-6):
    """
    Build cell edges and detect the angular extent of the domain.

    Works for any spherical sub-domain: a narrow wedge, a polar cap, a full
    2pi shell, or a complete disk. The returned flags let the integrator skip
    the wedge-rejection tests (and drop the degenerate cone / duplicate
    half-plane boundaries) whenever a coordinate is complete or periodic.

    Returns a dict with
        r_edges, theta_edges, phi_edges : cell boundaries
        theta_cross, phi_cross          : boundary surfaces to split rays on
        full_theta, full_phi            : coordinate spans its full range
        phi_span                        : phi_edges[-1] - phi_edges[0]
        r_min, r_max                    : radial extent of the domain
    """
    r_edges = make_edges(np.asarray(r_ori, dtype=np.float64))
    r_edges[0] = max(r_edges[0], 0.0)

    theta_edges = np.clip(make_edges(np.asarray(theta_ori, dtype=np.float64)),
                          0.0, np.pi)

    # Unwrap first, so a wedge straddling +-pi becomes monotonic.
    phi_c = np.unwrap(np.asarray(phi_ori, dtype=np.float64))
    if phi_c[-1] < phi_c[0]:
        raise ValueError('phi centres must increase after np.unwrap; reverse '
                         'phi_ori and the last axis of the data grids')
    phi_edges = make_edges(phi_c)

    phi_span = float(phi_edges[-1] - phi_edges[0])
    full_phi = bool(phi_span >= 2.0*np.pi - tol)
    if full_phi:
        # Snap the seam shut so round-off cannot drop samples there.
        phi_span = 2.0*np.pi
        phi_edges[-1] = phi_edges[0] + 2.0*np.pi

    full_theta = bool(theta_edges[0] <= tol and theta_edges[-1] >= np.pi - tol)

    # Surfaces used only for splitting rays into segments.
    # A theta edge at 0 or pi is the polar axis, not a cone: the quadratic is
    # degenerate there and yields no real segment boundary.
    theta_cross = theta_edges[np.abs(np.sin(theta_edges)) > tol]
    # On a periodic phi grid the first and last edge are the same half-plane.
    phi_cross = phi_edges[:-1] if full_phi else phi_edges

    return {
        'r_edges':     np.ascontiguousarray(r_edges),
        'theta_edges': np.ascontiguousarray(theta_edges),
        'phi_edges':   np.ascontiguousarray(phi_edges),
        'theta_cross': np.ascontiguousarray(theta_cross),
        'phi_cross':   np.ascontiguousarray(phi_cross),
        'full_theta':  full_theta,
        'full_phi':    full_phi,
        'phi_span':    phi_span,
        'r_min':       float(r_edges[0]),
        'r_max':       float(r_edges[-1]),
    }
