"""
rrt.euv
───────
Ray-cell (Siddon) intersection integrator for synthetic EUV emission.

Column EM = ∫ n_e² dℓ, then I_λ = R_λ(T) · (EM path), isothermal or with a real
temperature field. Any SDO/AIA channel is supported via the response table
``(T_resp, R_resp)``.

The boundary-intersection geometry, ``make_edges`` and ``prepare_domain`` are
imported from :mod:`rrt.geometry` — the single shared core used by the EUV,
white-light and LOS-diagnostic paths alike. 

Domain-agnostic: works for a wedge, a polar cap, a full shell or a full disk.

Example
-------
    from rrt.euv import run_siddon
    EUV_image = run_siddon(
        Xg, Yg, x_img, y_img, e_los,
        r_ori, theta_ori, phi_ori,
        ne_grid, T_grid,
        T_resp, R_resp,
        Npix=516, Rmax=1.5, WAVELENGTH=171,  # 94, 211, 304, 131, 335
        OBS_TIME='2024-05-08T14:09:00',
    )
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit, prange

from rrt.geometry import (
    _radial_intersections,
    _theta_intersections,
    _phi_intersections,
    make_edges,        # noqa: F401  (re-exported for backward compatibility)
    prepare_domain,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE INTEGRATOR  (Numba JIT, parallelised over pixels)
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True)
def siddon_integrate(
    X_flat, Y_flat,           # (Npix²,)  flat pixel coords in R_sun
    x_img, y_img, e_los,      # (3,)      observer geometry
    r_edges, theta_edges, phi_edges,   # cell boundary edges
    theta_cross, phi_cross,   # boundary surfaces to split rays on
    r_cen,  theta_cen,  phi_cen,       # cell centre coords
    ne_data, T_data,          # (nr, ntheta, nphi)  MHD fields
    T_resp,  R_resp,          # (nresp,)  AIA response function
    Rsun_cm,                  # float     R_sun in cm
    r_min, r_max,             # float     valid domain bounds
    r_occ,                    # float     opaque body radius (photosphere)
    full_theta, full_phi,     # bool      domain complete in theta / phi
    phi_span                  # float     phi_edges[-1] - phi_edges[0]
):
    """
    Rectilinear Ray-cell integrator.
    Returns flat array of I_EUV values (DN/s/pixel) for every pixel.

    Domain-agnostic: works for a wedge, a polar cap, a full shell or a full
    disk. The angular rejection tests are skipped for whichever coordinate
    the caller reports as complete.
    """
    Npix2  = X_flat.shape[0]
    result = np.zeros(Npix2, dtype=np.float64)
    nresp  = T_resp.shape[0]
    nr     = r_cen.shape[0]
    nt     = theta_cen.shape[0]
    np_    = phi_cen.shape[0]

    for idx in prange(Npix2):
        Xp = X_flat[idx];  Yp = Y_flat[idx]

        # Sky-plane position vector (Cartesian, R_sun)
        rx = Xp*x_img[0] + Yp*y_img[0]
        ry = Xp*x_img[1] + Yp*y_img[1]
        rz = Xp*x_img[2] + Yp*y_img[2]
        ex = e_los[0];  ey = e_los[1];  ez = e_los[2]

        # ── LOS bounds from the impact parameter ──────────────────────────────
        # r_0 ⊥ ê_los → |r(s)|² = b² + s², so the ray is inside the outer
        # domain shell for |s| ≤ sqrt(r_max² - b²).
        # On-disk pixels (b ≤ r_occ) have their LOS intercepted by the
        # photosphere, which is opaque to EUV: stop at s = -sqrt(r_occ² - b²),
        # the crossing on the observer's side, so the far side is never
        # integrated. Off-limb pixels see the full chord.
        b2 = Xp*Xp + Yp*Yp  # impact parameter squared
        if b2 >= r_max*r_max:
            continue        # ray misses the domain entirely
        s_out = (r_max*r_max - b2)**0.5

        s_lo = -s_out
        if b2 <= r_occ*r_occ:
            s_hi = -(r_occ*r_occ - b2)**0.5
        else:
            s_hi = s_out

        # ── Collect all boundary crossings ────────────────────────────────────
        crossings = [s_lo, s_hi]

        for s in _radial_intersections(rx, ry, rz, ex, ey, ez, r_edges):
            if s_lo < s < s_hi:
                crossings.append(s)
        for s in _theta_intersections(rx, ry, rz, ex, ey, ez, theta_cross):
            if s_lo < s < s_hi:
                crossings.append(s)
        for s in _phi_intersections(rx, ry, rz, ex, ey, ez, phi_cross):
            if s_lo < s < s_hi:
                crossings.append(s)

        # ── Sort ──────────────────────────────────────────────────────────────
        carr = np.array(crossings)
        carr.sort()

        # ── Walk segments ─────────────────────────────────────────────────────
        I_pix = 0.0
        for k in range(len(carr) - 1):
            s0 = carr[k];   s1 = carr[k+1]
            ds = s1 - s0
            if ds <= 0:
                continue

            # Midpoint
            sm  = 0.5*(s0 + s1)
            pmx = rx + sm*ex
            pmy = ry + sm*ey
            pmz = rz + sm*ez

            # Scalar radius
            r_m = (pmx*pmx + pmy*pmy + pmz*pmz)**0.5
            if r_m < r_min or r_m > r_max or r_m < r_occ:
                continue

            # Spherical coords
            th_m = np.arccos(max(-1.0, min(1.0, pmz/r_m)))
            ph_m = np.arctan2(pmy, pmx)

            # Reject samples outside the domain. A complete coordinate needs
            # no test — every sample is inside it by construction.
            if not full_theta:
                if th_m < theta_edges[0] or th_m > theta_edges[-1]:
                    continue
            # Fold phi into [phi_edges[0], phi_edges[0] + 2pi); for a partial
            # domain anything past the span falls in the gap.
            dph = (ph_m - phi_edges[0]) % (2.0*np.pi)
            if (not full_phi) and dph > phi_span:
                continue
            ph_m = phi_edges[0] + dph


            # Nearest-cell lookup (avoids interpolation artifacts)
            ir = np.searchsorted(r_edges,     r_m,  side='right') - 1
            it = np.searchsorted(theta_edges, th_m, side='right') - 1
            ip = np.searchsorted(phi_edges,   ph_m, side='right') - 1
            if ir < 0:    ir = 0
            if it < 0:    it = 0
            if ip < 0:    ip = 0
            if ir >= nr:  ir = nr  - 1
            if it >= nt:  it = nt  - 1
            if ip >= np_: ip = np_ - 1


            ne_val = ne_data[ir, it, ip]
            T_val  = T_data[ir,  it, ip]

            # R(T) via log-log linear interpolation on the response table.
            # (linear-linear interpolation undersamples the steep hot peaks of
            #  the 94 and 131 A channels; interpolating log10(R) vs log10(T) is
            #  far more accurate for a response that spans many decades.)
            if T_val <= T_resp[0]:
                R_val = R_resp[0]
            elif T_val >= T_resp[nresp - 1]:
                R_val = R_resp[nresp - 1]
            else:
                j = np.searchsorted(T_resp, T_val)
                logT_lo = np.log10(T_resp[j-1]);  logT_hi = np.log10(T_resp[j])
                logT    = np.log10(T_val)
                # guard against zero response entries before taking the log
                R_lo = R_resp[j-1] if R_resp[j-1] > 0.0 else 1e-40
                R_hi = R_resp[j]   if R_resp[j]   > 0.0 else 1e-40
                logR_lo = np.log10(R_lo);  logR_hi = np.log10(R_hi)
                frac    = (logT - logT_lo) / (logT_hi - logT_lo)
                R_val   = 10.0**(logR_lo + frac*(logR_hi - logR_lo))

            # Accumulate
            I_pix += ne_val**2 * R_val * ds * Rsun_cm

        result[idx] = I_pix

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def run_siddon(
    Xg, Yg,
    x_img, y_img, e_los,
    r_ori, theta_ori, phi_ori,
    ne_grid, T_grid,
    T_resp, R_resp,
    Npix=256, Rmax=1.5,
    WAVELENGTH=171,
    OBS_TIME='',
    Rsun_cm=6.96e10,
    R_body=1.0,
    plot=True,
    save=True
):
    """
    Full Siddon EUV pipeline.

    Parameters
    ----------
    Xg, Yg         : 2D image-plane pixel grids (R_sun)
    x_img, y_img   : image-plane basis vectors
    e_los          : LOS unit vector (Earth → Sun)
    r_ori          : 1D radial cell centres (R_sun)
    theta_ori      : 1D polar cell centres (rad)
    phi_ori        : 1D azimuthal cell centres (rad)
    ne_grid        : 3D density array (nr, ntheta, nphi)  [cm⁻³]
    T_grid         : 3D temperature array  (nr, ntheta, nphi)  [K]
    T_resp, R_resp : AIA temperature response arrays
    Npix           : image size (pixels per side)
    Rmax           : image half-width (R_sun)
    WAVELENGTH     : AIA channel (Å)
    OBS_TIME       : observation time string (for plot title)
    Rsun_cm        : R_sun in cm
    R_body         : radius of the opaque body that clips the LOS (R_sun).
                     1.0 = photosphere; 0.0 disables the clip.
    plot           : whether to show/save the image
    save           : whether to save PNG

    Returns
    -------
    EUV_image : 2D array (Npix, Npix)  [DN/s/pixel]
    """

    # ── Build cell edges and detect the domain extent ─────────────────────────
    dom = prepare_domain(r_ori, theta_ori, phi_ori)
    print(f"Domain: r=[{dom['r_min']:.3f}, {dom['r_max']:.3f}] R_sun, "
          f"theta {'full' if dom['full_theta'] else 'partial'}, "
          f"phi {'full' if dom['full_phi'] else 'partial'} "
          f"({np.degrees(dom['phi_span']):.1f} deg)")

    # ── Flatten pixel grid ────────────────────────────────────────────────────
    X_flat = Xg.flatten().astype(np.float64)
    Y_flat = Yg.flatten().astype(np.float64)

    args = (
        x_img.astype(np.float64),
        y_img.astype(np.float64),
        e_los.astype(np.float64),
        dom['r_edges'], dom['theta_edges'], dom['phi_edges'],
        dom['theta_cross'], dom['phi_cross'],
        np.ascontiguousarray(r_ori,     dtype=np.float64),
        np.ascontiguousarray(theta_ori, dtype=np.float64),
        np.ascontiguousarray(phi_ori,   dtype=np.float64),
        np.ascontiguousarray(ne_grid,   dtype=np.float64),
        np.ascontiguousarray(T_grid,    dtype=np.float64),
        np.ascontiguousarray(T_resp,    dtype=np.float64),
        np.ascontiguousarray(R_resp,    dtype=np.float64),
        Rsun_cm,
        dom['r_min'], dom['r_max'], float(R_body),
        dom['full_theta'], dom['full_phi'], dom['phi_span'],
    )

    # ── JIT warm-up ───────────────────────────────────────────────────────────
    print('Warming up Numba JIT (first call compiles, ~10 s)...')
    _ = siddon_integrate(X_flat[:4], Y_flat[:4], *args)

    # ── Full integration ──────────────────────────────────────────────────────
    print(f'Running Siddon integration ({Npix}×{Npix} pixels)...')
    t0 = time.time()
    result_flat = siddon_integrate(X_flat, Y_flat, *args)
    dt = time.time() - t0

    EUV_image = result_flat.reshape(Npix, Npix)
    nonzero   = EUV_image[EUV_image > 0]
    if nonzero.size == 0:
        print(f'Done in {dt:.1f} s  |  image is entirely zero — check that the '
              f'domain overlaps the field of view')
    else:
        print(f'Done in {dt:.1f} s  |  '
              f'I_min={nonzero.min():.2e}  I_max={EUV_image.max():.2e} DN/s/pix')

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        EUV_plot = EUV_image.copy().astype(float)
        EUV_plot[EUV_plot <= 0] = np.nan
        vmin = np.nanpercentile(EUV_plot, 1)
        vmax = np.nanpercentile(EUV_plot, 99)

        cmap = plt.get_cmap(f'sdoaia{WAVELENGTH}').copy()
        cmap.set_bad('black')

        fig, ax = plt.subplots(figsize=(6.5, 6))
        im = ax.imshow(
            EUV_plot, origin='lower',
            extent=[-Rmax, Rmax, -Rmax, Rmax],
            cmap=cmap,
            norm=LogNorm(vmin=max(vmin, 1e-30), vmax=vmax),
            interpolation='bilinear'
        )
        fig.colorbar(im, ax=ax, pad=0.02, label='DN / s / pixel')
        #ax.set_xlabel(r'Solar West  [$R_\odot$]')
        #ax.set_ylabel(r'Solar North [$R_\odot$]')
        ax.set_title(f'Synthetic AIA {WAVELENGTH} Å  ({OBS_TIME})')
        ax.set_aspect('equal')
        plt.tight_layout()

        if save:
            fname = f'synthetic_EUV_{WAVELENGTH}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Figure saved → {fname}')
        plt.show()

    return EUV_image
