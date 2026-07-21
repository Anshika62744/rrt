"""
Shared pytest fixtures. All test data is generated in-memory (or into a tmp dir)
by :mod:`rrt.synthdata` — no proprietary files are ever needed.
"""
import matplotlib
matplotlib.use("Agg")  # headless: never open a GUI window during tests

import numpy as np
import pytest

from rrt import synthdata


# ─────────────────────────────────────────────────────────────────────────────
# Observer geometry (a tiny, self-contained copy of the make_observer logic; the
# packaged rrt.observer arrives in Phase 2 step 6 and is tested there).
# ─────────────────────────────────────────────────────────────────────────────

def observer(phi_obs_deg=-70.0, B0_deg=0.0):
    """Return (x_img, y_img, e_los) for a viewpoint. Pole-safe up-vector."""
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
    return x_img, y_img, e_los


def image_grid(rmax, npix):
    xs = np.linspace(-rmax, rmax, npix)
    Xg, Yg = np.meshgrid(xs, xs, indexing="xy")
    return Xg, Yg


@pytest.fixture(scope="session")
def view():
    """A single equatorial viewpoint shared across tests."""
    return observer(phi_obs_deg=-70.0, B0_deg=0.0)


@pytest.fixture
def small_grid():
    """Small image grid factory: small_grid(rmax=3.0, npix=6) -> (Xg, Yg)."""
    return image_grid


@pytest.fixture(scope="session")
def wedge_cube():
    return synthdata.make_cube("wedge", "single", nr=8, nth=10, nph=12,
                               r_range=(1.0, 3.0))


@pytest.fixture(scope="session")
def full_cube():
    return synthdata.make_cube("full", "single", nr=8, nth=12, nph=16,
                               r_range=(1.0, 3.0))


@pytest.fixture(scope="session")
def flat_response():
    """Flat unit response → run_siddon returns pure column EM = ∫ n_e² dℓ."""
    return np.array([1.0e3, 1.0e9]), np.array([1.0, 1.0])
