"""
rrt.observer
────────────
Observer geometry: the image-plane basis and line-of-sight vector for a
viewpoint, and the sky-plane pixel grid. Harvested from the (identical) copies
that were duplicated across the driver notebooks.

A viewpoint may be given as explicit ``(phi_obs_deg, B0_deg)`` or, for
Carrington-framed data (e.g. MAS/PSI), derived from an observation time via
:func:`sub_earth_angles` (sub-Earth L0/B0 from sunpy).
"""

from __future__ import annotations

import numpy as np


def make_observer(phi_obs_deg: float, B0_deg: float):
    """Return ``(n_obs, e_los, x_img, y_img)`` for a viewpoint.

    Parameters
    ----------
    phi_obs_deg : observer longitude in the simulation frame [deg] (0=+x, 90=+y).
    B0_deg      : observer heliographic latitude [deg] (0=equator, 90=north pole).

    ``n_obs`` points from Sun to observer; ``e_los = -n_obs`` is the look
    direction. The image basis uses a pole-safe up-vector so a top-down view does
    not degenerate.
    """
    phi = np.deg2rad(phi_obs_deg)
    B0 = np.deg2rad(B0_deg)
    n_obs = np.array([np.cos(B0) * np.cos(phi),
                      np.cos(B0) * np.sin(phi),
                      np.sin(B0)])
    n_obs /= np.linalg.norm(n_obs)
    e_los = -n_obs
    up = np.array([1.0, 0.0, 0.0]) if abs(B0_deg) > 89.99 else np.array([0.0, 0.0, 1.0])
    x_img = np.cross(up, n_obs); x_img /= np.linalg.norm(x_img)
    y_img = np.cross(n_obs, x_img); y_img /= np.linalg.norm(y_img)
    return n_obs, e_los, x_img, y_img


def sub_earth_angles(obs_time):
    """Sub-Earth ``(L0_deg, B0_deg)`` at ``obs_time`` via sunpy.

    ``L0`` is the Carrington longitude of the sub-Earth point and ``B0`` the
    heliographic latitude. For **Carrington-framed** data (e.g. MAS/PSI cubes,
    whose φ axis *is* Carrington longitude), ``L0`` is the observer longitude to
    pass as ``phi_obs_deg`` and ``B0`` the latitude.

    Parameters
    ----------
    obs_time : str or astropy.time.Time
        e.g. ``"2024-05-08T14:09:00"`` (UTC).
    """
    from astropy.time import Time
    import astropy.units as u
    from sunpy.coordinates import sun

    t = obs_time if hasattr(obs_time, "jd") else Time(obs_time, scale="utc")
    L0 = float(sun.L0(t).to(u.deg).value)
    B0 = float(sun.B0(t).to(u.deg).value)
    return L0, B0


def make_observer_from_time(obs_time, phi_offset_deg=0.0):
    """``make_observer`` driven by an observation time (sub-Earth L0/B0).

    ``phi_offset_deg`` is added to L0 to absorb any zero-point offset between the
    data's φ axis and Carrington longitude (0 for a standard Carrington cube).
    """
    L0, B0 = sub_earth_angles(obs_time)
    return make_observer(L0 + phi_offset_deg, B0)


def view_angles(view):
    """Resolve a config ``view`` dict to ``(phi_obs_deg, B0_deg)``.

    Accepts either explicit ``phi_obs_deg`` / ``B0_deg``, or an ``obs_time``
    (→ sub-Earth L0/B0 via sunpy, with an optional ``phi_offset_deg``). Explicit
    ``phi_obs_deg`` / ``B0_deg`` override the time-derived values when both given.
    """
    if view.get("obs_time"):
        L0, B0 = sub_earth_angles(view["obs_time"])
        phi = float(view.get("phi_obs_deg", L0 + float(view.get("phi_offset_deg", 0.0))))
        b0 = float(view.get("B0_deg", B0))
        return phi, b0
    return float(view["phi_obs_deg"]), float(view["B0_deg"])


def image_grid(rmax: float, npix: int):
    """Square sky-plane pixel grid spanning ``[-rmax, rmax]`` [R_sun].

    Returns ``(Xg, Yg, Rperp)`` where ``Rperp = hypot(Xg, Yg)`` is the impact
    parameter of each pixel.
    """
    xs = np.linspace(-rmax, rmax, npix)
    Xg, Yg = np.meshgrid(xs, xs, indexing="xy")
    return Xg, Yg, np.hypot(Xg, Yg)
