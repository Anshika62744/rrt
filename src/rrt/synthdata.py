"""
rrt.synthdata
─────────────
Generate small, self-contained synthetic MHD cubes so the tests and examples run
**without** any proprietary data. Two things are provided:

* in-memory grid + analytic fields (``spherical_grid``, ``analytic_ne``,
  ``analytic_T``, ``make_cube``) — used directly by the numerical tests, and
* HDF5 writers in both supported layouts (``write_arms``, ``write_mas``) that
  reproduce the messy-data features the loader must cope with: ghost cells,
  various density/temperature unit encodings, axis orders and angle units.

The analytic density is spherically symmetric by default,
``n_e(r) = n0 · (r0 / r)^p`` with ``p = 2`` — a plane-parallel hydrostatic-ish
falloff. That symmetry is what the azimuthal-symmetry correctness check relies on;
pass ``aniso=True`` for a mild latitude/longitude modulation when you want
structure in the image.

Domain presets (drive ``prepare_domain``'s flags):

===========  ==========  ========  =========
preset       full_theta  full_phi  shape
===========  ==========  ========  =========
``wedge``    False       False     narrow θ & φ wedge
``polar_cap` False       True      θ∈(0,40°], full 2π φ
``full``     True        True      full shell / disk
===========  ==========  ========  =========
"""

from __future__ import annotations

import numpy as np

D2R = np.pi / 180.0

# Angular extents for each domain preset: (theta_lo, theta_hi, phi_lo, phi_hi) [deg]
_DOMAIN_DEG = {
    "wedge":     (70.0, 110.0, -15.0, 15.0),
    "polar_cap": (2.0, 40.0, None, None),   # phi spans full 2π (set below)
    "full":      (0.0, 180.0, None, None),  # theta full, phi full
}


def spherical_grid(domain="wedge", nr=10, nth=12, nph=14, r_range=(1.0, 3.0)):
    """Return 1-D ascending cell-centre arrays ``(r, theta, phi)`` for a preset.

    ``r`` in R_sun, ``theta``/``phi`` in radians. φ is built on ``[0, 2π)`` for
    the periodic presets so ``prepare_domain`` reports ``full_phi=True``.
    """
    if domain not in _DOMAIN_DEG:
        raise ValueError(f"unknown domain {domain!r}; choose from {list(_DOMAIN_DEG)}")
    th_lo, th_hi, ph_lo, ph_hi = _DOMAIN_DEG[domain]

    r = np.linspace(r_range[0], r_range[1], nr)
    theta = np.linspace(th_lo * D2R, th_hi * D2R, nth)

    if ph_lo is None:  # periodic full-2π φ (endpoint excluded so the seam closes)
        phi = np.linspace(0.0, 2.0 * np.pi, nph, endpoint=False)
    else:
        phi = np.linspace(ph_lo * D2R, ph_hi * D2R, nph)
    return r, theta, phi


def analytic_ne(r, theta, phi, n0=1.0e8, r0=1.0, p=2.0, aniso=False):
    """Analytic electron density ``n0·(r0/r)^p`` on the ``(r,θ,φ)`` grid [cm⁻³]."""
    R = np.asarray(r, float)[:, None, None]
    ne = n0 * (r0 / R) ** p * np.ones((r.size, theta.size, phi.size))
    if aniso:
        TH = np.asarray(theta, float)[None, :, None]
        PH = np.asarray(phi, float)[None, None, :]
        ne = ne * (1.0 + 0.3 * np.cos(TH) ** 2 + 0.2 * np.cos(PH))
    return np.ascontiguousarray(ne, dtype=np.float64)


def analytic_T(r, theta, phi, t0=1.5e6, gradient=False):
    """Analytic temperature field [K]. Constant ``t0`` unless ``gradient``."""
    T = np.full((r.size, theta.size, phi.size), float(t0))
    if gradient:
        R = np.asarray(r, float)[:, None, None]
        T = t0 * (R[0] / R) ** 0.3 * np.ones_like(T)
    return np.ascontiguousarray(T, dtype=np.float64)


def make_cube(domain="wedge", case="density_only", *, nr=10, nth=12, nph=14,
              r_range=(1.0, 3.0), n0=1.0e8, t0=1.5e6, aniso=False):
    """Convenience: build a full in-memory cube as a dict.

    ``case``: ``"density_only"`` → ``T`` is None; ``"single"``/``"separate"`` →
    a real temperature field is included (the loader distinguishes *where* T comes
    from; the array itself is the same).
    """
    r, theta, phi = spherical_grid(domain, nr, nth, nph, r_range)
    ne = analytic_ne(r, theta, phi, n0=n0, aniso=aniso)
    T = None if case == "density_only" else analytic_T(r, theta, phi, t0=t0)
    return dict(r=r, theta=theta, phi=phi, ne=ne, T=T,
                domain=domain, case=case)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writers — reproduce the two real file layouts and their quirks
# ─────────────────────────────────────────────────────────────────────────────

def _add_ghost(centres, nghost):
    """Extend a 1-D centre array by ``nghost`` uniformly-spaced cells each side."""
    if nghost <= 0:
        return centres
    step = centres[1] - centres[0]
    lo = centres[0] + step * np.arange(-nghost, 0)
    hi = centres[-1] + step * np.arange(1, nghost + 1)
    return np.concatenate([lo, centres, hi])


def write_arms(path, r, theta, phi, ne, T=None, *, nghost=0,
               density_units="cm^-3", temperature_units="K",
               angle_units="rad", ne_is_mass=False, mu=0.6, m_p=1.6726e-24):
    """Write an ARMS-style file: ``coords/{r,theta,phi}`` + ``vars/{rho[,T]}``.

    Data cubes are stored in ``(phi, theta, r)`` order with an ``axis_order``
    root attribute, matching the real ARMS export. Ghost cells (if requested)
    are added on every axis and flagged via ``ghost_cells_included``/``nghost``.
    """
    import h5py

    rc = _add_ghost(r, nghost)
    tc = _add_ghost(theta, nghost)
    pc = _add_ghost(phi, nghost)

    def _embed(field):
        """Place the physical cube inside a ghost-padded (phi,theta,r) array."""
        cube = np.transpose(field, (2, 1, 0))  # (r,θ,φ) → (φ,θ,r)
        if nghost <= 0:
            return np.ascontiguousarray(cube, dtype=np.float32)
        full = np.zeros((pc.size, tc.size, rc.size), dtype=np.float32)
        full[nghost:-nghost, nghost:-nghost, nghost:-nghost] = cube
        return full

    rho = ne * (mu * m_p) if ne_is_mass else ne
    tc_out = np.degrees(tc) if angle_units == "deg" else tc
    pc_out = np.degrees(pc) if angle_units == "deg" else pc

    with h5py.File(path, "w") as f:
        f.attrs["axis_order"] = "(phi, theta, r, var)"
        f.attrs["ghost_cells_included"] = bool(nghost > 0)
        f.attrs["nghost"] = int(nghost)
        g = f.create_group("coords")
        g.create_dataset("r", data=rc.astype(np.float64)).attrs["units"] = "R_sun"
        g.create_dataset("theta", data=tc_out.astype(np.float64)).attrs["units"] = angle_units
        g.create_dataset("phi", data=pc_out.astype(np.float64)).attrs["units"] = angle_units
        v = f.create_group("vars")
        d = v.create_dataset("rho", data=_embed(rho))
        d.attrs["units"] = "g cm^-3" if ne_is_mass else density_units
        d.attrs["axis_order"] = "(phi, theta, r)"
        if T is not None:
            Tw = _encode_temperature(T, temperature_units)
            dt = v.create_dataset("T", data=_embed(Tw))
            dt.attrs["units"] = temperature_units
            dt.attrs["axis_order"] = "(phi, theta, r)"


def write_mas(path, r, theta, phi, field, *, ne_scale=1.0):
    """Write a MAS/PSI-style file: one ``Data`` cube + ``dim1/dim2/dim3`` scales.

    Mimics the real PSI export: no attributes, code units. ``Data`` is stored in
    ``(r, theta, phi)`` order with ``dim1=r, dim2=theta, dim3=phi``. The loader
    identifies axes from coordinate ranges, so use an ``r`` range whose maximum
    exceeds 2π (as real coronal-scale MAS grids do) to keep that heuristic valid.
    """
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("dim1", data=np.asarray(r, np.float64))
        f.create_dataset("dim2", data=np.asarray(theta, np.float64))
        f.create_dataset("dim3", data=np.asarray(phi, np.float64))
        f.create_dataset("Data", data=np.ascontiguousarray(
            field / ne_scale, dtype=np.float64))


def write_response(path, channels=(171, 193, 211), logte=None):
    """Write a tiny AIA-style response ``.npz`` (``logte`` + ``resp{λ}``).

    The curves are smooth synthetic bumps peaked near log10 T ≈ 6 — enough to
    exercise the log-log R(T) interpolation without shipping the real table.
    """
    if logte is None:
        logte = np.linspace(4.0, 8.0, 61)
    out = {"logte": logte.astype(np.float32)}
    for i, ch in enumerate(channels):
        peak = 5.8 + 0.15 * i
        curve = 1.0e-27 * np.exp(-0.5 * ((logte - peak) / 0.25) ** 2) + 1.0e-32
        out[f"resp{ch}"] = curve.astype(np.float64)
    np.savez(path, **out)
    return path


def _encode_temperature(T, units):
    """Encode a Kelvin field into the requested on-disk unit convention."""
    u = units.lower()
    if u in ("mk", "megakelvin"):
        return (T / 1.0e6).astype(np.float64)
    if "log" in u:
        return np.log10(T).astype(np.float64)
    return T.astype(np.float64)  # kelvin
