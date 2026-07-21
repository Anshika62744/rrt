"""
rrt.wl
──────
Ray-cell (Siddon) intersection integrator for synthetic white-light polarised
brightness (pB) via Thomson scattering (van de Hulst / Billings coefficients).

pB(pixel) = ∫ n_e(r) · K(r) · dℓ,  with  K(r) = σ_e · [(1-u)·A(r) + u·B(r)].

The boundary-intersection geometry, ``make_edges`` and ``prepare_domain`` are
imported from :mod:`rrt.geometry` — the single shared core. The integrator kernel
below is byte-for-byte the validated original; only the source of the geometry
helpers changed (import instead of an in-file copy).

Domain-agnostic: works for a wedge, a polar cap, a full shell or a full disk.

Example
-------
    from rrt.wl import run_siddon_pB
    pB_image = run_siddon_pB(
        Xg, Yg, x_img, y_img, e_los,
        r_ori, theta_ori, phi_ori,
        ne_grid,
        Npix=516, Rmax=6.0,
        OBS_TIME='2024-05-07T14:09:00',
    )
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # noqa: F401  (kept for API parity)
from numba import njit, prange

from rrt.geometry import (
    _radial_intersections,
    _theta_intersections,
    _phi_intersections,
    make_edges,        # noqa: F401  (re-exported for backward compatibility)
    prepare_domain,
)


# ─────────────────────────────────────────────────────────────────────────────
# WHITE LIGHT pB — THOMSON SCATTERING KERNEL  (van de Hulst / Billings)
# ─────────────────────────────────────────────────────────────────────────────

@njit
def _billings_ABCD(r):
    """
    Van de Hulst / Billings geometry factors for Thomson scattering.
    r  : 3-D distance from Sun centre  [R_sun]
    Returns A, B  (the two pB-relevant terms)
    """
    sinO  = 1.0 / r                         # sin(Omega) = R_sun / r
    if sinO >= 1.0:
        sinO = 1.0 - 1e-12
    cosO  = (1.0 - sinO*sinO)**0.5
    # avoid log(0)
    arg   = (1.0 + sinO) / cosO
    if arg <= 0.0:
        arg = 1e-30
    lnval = np.log(arg)

    A = cosO * sinO * sinO
    B = -0.125 * (
        1.0 - 3.0*sinO*sinO
        - (cosO*cosO / sinO) * (1.0 + 3.0*sinO*sinO) * lnval
    )
    return A, B


@njit
def _pB_kernel(r, U_LIMB, SIGMA_T):
    """
    Polarised brightness kernel K(r):
        K(r) = SIGMA_T * [ (1 - u)*A(r) + u*B(r) ]
    Units: cm²  (scattering constant times dimensionless geometry)

    SIGMA_T here is the coronal Thomson-scattering constant that multiplies the
    van de Hulst / Billings geometric factors A, B — the standard value is the
    classical electron radius squared r_e² ≈ 7.95e-26 cm². (The A, B factors
    already carry the angular dependence, so the total cross-section
    σ_T = (8π/3) r_e² = 6.65e-25 would double-count the 8π/3 integration.)
    """
    A, B = _billings_ABCD(r)
    return SIGMA_T * ((1.0 - U_LIMB)*A + U_LIMB*B)


# ─────────────────────────────────────────────────────────────────────────────
# CORE pB SIDDON INTEGRATOR  (Numba JIT, parallelised over pixels)
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True)
def siddon_integrate_pB(
    X_flat, Y_flat,                        # (Npix²,)  sky-plane pixels [R_sun]
    x_img, y_img, e_los,                   # (3,)      observer geometry
    r_edges, theta_edges, phi_edges,       # cell boundary edges
    theta_cross, phi_cross,                # boundary surfaces to split rays on
    r_cen,  theta_cen,  phi_cen,           # cell centre coords
    ne_data,                               # (nr, ntheta, nphi) [cm⁻³]
    Rsun_cm,                               # float  R_sun in cm
    U_LIMB,                                # float  limb-darkening coefficient
    SIGMA_T,                               # float  Thomson cross-section [cm²]
    r_min, r_max,                          # float  valid domain bounds [R_sun]
    r_occ,                                 # float  opaque body radius [R_sun]
    full_theta, full_phi,                  # bool   domain complete in theta/phi
    phi_span                               # float  phi_edges[-1] - phi_edges[0]
):
    """
    Siddon ray-cell integrator for polarised brightness (white light).

    Integrand:  pB(pixel) = ∫ ne(r) · K(r) · ds
    where K(r) = SIGMA_T * [(1-u)*A(r) + u*B(r)]  [cm²]

    Units: [cm⁻³] × [cm²] × [cm] = [cm⁻¹]
    (multiply by (Rsun_cm / 1AU_cm)² outside to get MSB if needed)
    """
    Npix2  = X_flat.shape[0]
    result = np.zeros(Npix2, dtype=np.float64)
    nr     = r_cen.shape[0]
    nt     = theta_cen.shape[0]
    np_    = phi_cen.shape[0]

    for idx in prange(Npix2):
        Xp = X_flat[idx];  Yp = Y_flat[idx]

        # Sky-plane position vector  (Cartesian, R_sun)
        rx = Xp*x_img[0] + Yp*y_img[0]
        ry = Xp*x_img[1] + Yp*y_img[1]
        rz = Xp*x_img[2] + Yp*y_img[2]
        ex = e_los[0];  ey = e_los[1];  ez = e_los[2]

        # ── LOS bounds from the impact parameter ──────────────────────────────
        # Same geometry as the EUV kernel: the photosphere is opaque in white
        # light too, so on-disk rays stop at the near-side crossing and never
        # sample the far side. Off-limb rays take the full chord.
        b2 = Xp*Xp + Yp*Yp
        if b2 >= r_max*r_max:
            continue
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
        pB_pix = 0.0
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

            # 3-D radius at midpoint
            r_m = (pmx*pmx + pmy*pmy + pmz*pmz)**0.5

            # Skip: outside domain, inside the opaque body
            if r_m < r_min or r_m > r_max or r_m < r_occ:
                continue

            th_m = np.arccos(max(-1.0, min(1.0, pmz / r_m)))
            ph_m = np.arctan2(pmy, pmx)                       # (-π, π]

            # Reject samples outside the domain. A complete coordinate needs
            # no test — every sample is inside it by construction.
            if not full_theta:
                if th_m < theta_edges[0] or th_m > theta_edges[-1]:
                    continue
            dph = (ph_m - phi_edges[0]) % (2.0*np.pi)
            if (not full_phi) and dph > phi_span:
                continue
            ph_m = phi_edges[0] + dph

            # nearest-cell lookup via EDGES (guaranteed in-range now)
            ir = np.searchsorted(r_edges,     r_m,  side='right') - 1
            it = np.searchsorted(theta_edges, th_m, side='right') - 1
            ip = np.searchsorted(phi_edges,   ph_m, side='right') - 1
            if ir < 0:    ir = 0
            if it < 0:    it = 0
            if ip < 0:    ip = 0
            if ir >= nr:  ir = nr  - 1
            if it >= nt:  it = nt  - 1
            if ip >= np_: ip = np_ - 1

            ne_val = ne_data[ir, it, ip]          # cm⁻³

            # Thomson scattering kernel at this 3-D radius
            K_val  = _pB_kernel(r_m, U_LIMB, SIGMA_T)   # cm²

            # Accumulate:  ne [cm⁻³] × K [cm²] × ds [cm]
            pB_pix += ne_val * K_val * ds * Rsun_cm

        result[idx] = pB_pix

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL WRAPPER for white light pB
# ─────────────────────────────────────────────────────────────────────────────

def run_siddon_pB(
    Xg, Yg,
    x_img, y_img, e_los,
    r_ori, theta_ori, phi_ori,
    ne_grid,
    Npix=256, Rmax=5.0,
    Rocc=1.0,
    OBS_TIME='',
    Rsun_cm=6.96e10,
    R_body=1.0,
    U_LIMB=0.63,
    SIGMA_T=7.95e-26,   # cm²  = classical electron radius squared r_e²; the
                        # standard van de Hulst/Billings coronal-pB constant.
    plot=True,
    save=True,
    log_limits=None              # (log10_min, log10_max) or None for auto
):
    """
    Siddon white light polarised brightness pipeline.

    Integrand:   pB = ∫ ne · K(r) · ds
    K(r) = SIGMA_T * [(1-U_LIMB)*A(r) + U_LIMB*B(r)]

    Parameters
    ----------
    Xg, Yg         : 2D image-plane pixel grids (R_sun)
    x_img, y_img   : image-plane basis vectors
    e_los          : LOS unit vector (Earth → Sun)
    r_ori          : 1D radial cell centres (R_sun)
    theta_ori      : 1D polar cell centres (rad)
    phi_ori        : 1D azimuthal cell centres (rad)
    ne_grid        : 3D density array (nr, ntheta, nphi)  [cm⁻³]
    Npix           : pixels per side
    Rmax           : image half-width  (R_sun)  — use 5–6 for coronagraph FOV
    Rocc           : occulter radius   (R_sun)  — 1.0 = solar limb.
                     Instrumental: masks the image, does not affect the LOS.
    OBS_TIME       : observation time string (for plot title)
    Rsun_cm        : R_sun in cm
    R_body         : radius of the opaque body that clips the LOS (R_sun).
                     1.0 = photosphere; 0.0 disables the clip.
    U_LIMB         : limb-darkening coefficient (default 0.63)
    SIGMA_T        : coronal Thomson-scattering constant in cm² multiplying the
                     Billings A, B factors. Default 7.95e-26 = classical electron
                     radius squared r_e² (the literature-standard value). Use the
                     total cross-section 6.6524587158e-25 only if your A, B
                     definitions do NOT already carry the angular integration.
    plot           : show plot
    save           : save PNG
    log_limits     : (log10_min, log10_max) for colorbar, or None for auto

    Returns
    -------
    pB_image : 2D array (Npix, Npix)  [cm⁻¹]
    """

    # ── Build cell edges and detect the domain extent ─────────────────────────
    dom = prepare_domain(r_ori, theta_ori, phi_ori)
    print(f"Domain: r=[{dom['r_min']:.3f}, {dom['r_max']:.3f}] R_sun, "
          f"theta {'full' if dom['full_theta'] else 'partial'}, "
          f"phi {'full' if dom['full_phi'] else 'partial'} "
          f"({np.degrees(dom['phi_span']):.1f} deg)")

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
        Rsun_cm, U_LIMB, SIGMA_T,
        dom['r_min'], dom['r_max'], float(R_body),
        dom['full_theta'], dom['full_phi'], dom['phi_span'],
    )

    # ── JIT warm-up ───────────────────────────────────────────────────────────
    print('Warming up Numba JIT for pB (first call compiles, ~10 s)...')
    _ = siddon_integrate_pB(X_flat[:4], Y_flat[:4], *args)

    # ── Full integration ──────────────────────────────────────────────────────
    print(f'Running Siddon pB integration ({Npix}×{Npix} pixels)...')
    t0 = time.time()
    result_flat = siddon_integrate_pB(X_flat, Y_flat, *args)
    dt = time.time() - t0

    pB_image = result_flat.reshape(Npix, Npix)
    nonzero  = pB_image[pB_image > 0]
    if nonzero.size == 0:
        print(f'Done in {dt:.1f} s  |  image is entirely zero — check that the '
              f'domain overlaps the field of view')
    else:
        print(f'Done in {dt:.1f} s  |  '
              f'pB_min={nonzero.min():.2e}  pB_max={pB_image.max():.2e} cm⁻¹')

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        from matplotlib.patches import Circle
        from matplotlib.colors import Normalize

        log_img = np.log10(pB_image.copy().astype(float))
        log_img[~np.isfinite(log_img)] = np.nan

        # Mask occulter region
        Rperp = np.sqrt(Xg**2 + Yg**2)
        log_img[Rperp < Rocc] = np.nan

        if log_limits is not None:
            log_min, log_max = log_limits
        else:
            log_min = np.nanpercentile(log_img, 1)
            log_max = np.nanpercentile(log_img, 99)

        cmap = plt.get_cmap('gray').copy()
        cmap.set_bad('black')

        fig, ax = plt.subplots(figsize=(6.5, 6))
        im = ax.imshow(
            log_img, origin='lower',
            extent=[-Rmax, Rmax, -Rmax, Rmax],
            cmap=cmap,
            norm=Normalize(vmin=log_min, vmax=log_max),
            interpolation='bilinear'
        )

        # Black occulter disk
        ax.add_patch(Circle((0, 0), Rocc,
                            facecolor='black', edgecolor='black', zorder=5))
        # Solar limb outline
        ax.add_patch(Circle((0, 0), 1.0,
                            facecolor='none', edgecolor='yellow',
                            lw=0.8, zorder=6))

        fig.colorbar(im, ax=ax, pad=0.02,
                     label=r'$\log_{10}$ pB  [cm$^{-1}$]')
        #ax.set_xlabel(r'Solar West  [$R_\odot$]')
        #ax.set_ylabel(r'Solar North [$R_\odot$]')
        #ax.set_title(f'Synthetic White Light pB — Siddon  ({OBS_TIME})')
        ax.set_xlim(-Rmax, Rmax)
        ax.set_ylim(-Rmax, Rmax)
        ax.set_aspect('equal')
        plt.tight_layout()

        if save:
            fname = 'synthetic_pB_siddon.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Figure saved → {fname}')
        plt.show()

    return pB_image
