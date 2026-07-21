"""
rrt.los
───────
Single line-of-sight diagnostics for the Siddon ray tracer — the "does the
algorithm work" check.

Domain-agnostic (wedge / polar cap / full shell / full disk). The walker mirrors
the production integrators (:mod:`rrt.euv`, :mod:`rrt.wl`): it splits rays on the
**same** non-degenerate boundary surfaces via the shared
:mod:`rrt.geometry` intersection helpers, rejects samples outside the domain only
where a coordinate is partial, folds φ into the domain window, and assigns cells
by digitising into EDGES (``searchsorted(edges, x, "right") - 1``) — the identical
rule the integrators use, so the plotted voxel bands are the cells actually
sampled. R(T) uses the **log-log** interpolation of :mod:`rrt.response`, matching
``rrt.euv`` exactly (previously this diagnostic used a linear interp).

  • sample_along_los()      — walk one ray, return per-segment (s_mid, ds, r,
                              θ, φ, n_e, T, cell indices, optional I) plus the
                              domain edges/flags it used.
  • plot_los_profile()      — stacked r(s), Δs, n_e, T, per-segment integrand.
  • plot_los_voxel_proof()  — r / θ / φ staircase vs s with the assigned-voxel
                              band shaded per segment, plus a PASS/CHECK verdict.
  • analyze_los()           — walk once, draw both.

Usage
─────
    from rrt.los import (sample_along_los, plot_los_profile,
                         plot_los_voxel_proof, analyze_los)

    s = sample_along_los(Xp, Yp, x_img, y_img, e_los,
                         r_ori, theta_ori, phi_ori, ne_grid, T_grid,
                         T_resp=T_resp, R_resp=R_resp)
    plot_los_profile(s)
    plot_los_voxel_proof(Xp, Yp, x_img, y_img, e_los,
                         r_ori, theta_ori, phi_ori, ne_grid, T_grid, samples=s)

    # or, in one call:
    analyze_los(Xp, Yp, x_img, y_img, e_los,
                r_ori, theta_ori, phi_ori, ne_grid, T_grid,
                T_resp=T_resp, R_resp=R_resp)
"""

import numpy as np
import matplotlib.pyplot as plt

from rrt.geometry import (
    make_edges,            # noqa: F401  (re-exported for parity/back-compat)
    prepare_domain,
    _radial_intersections,
    _theta_intersections,
    _phi_intersections,
)
from rrt.response import response_at


# ─────────────────────────────────────────────────────────────────────────────
# Single-ray walker  (domain-agnostic, edge-digitised assignment)
# ─────────────────────────────────────────────────────────────────────────────

def sample_along_los(
    Xp, Yp,
    x_img, y_img, e_los,
    r_ori, theta_ori, phi_ori,
    ne_grid, T_grid,
    T_resp=None, R_resp=None,
    Rsun_cm=6.96e10,
    r_occ=1.0,
    tol=1e-6,
):
    """
    Walk one Siddon ray and return per-segment diagnostic arrays.

    Domain-agnostic: handles wedge, polar cap, full shell and full disk via the
    shared :func:`rrt.geometry.prepare_domain`. Cell assignment digitises into
    EDGES to match the production integrators, so ``samples['ir/it/ip']`` are the
    voxels actually read.

    Returns
    -------
    dict of length-nseg arrays (nseg = len(crossings)-1):
        s_mid, ds                segment midpoint and length [R_sun]
        r, theta, phi            spherical coords at midpoint (phi folded into
                                 the domain window [phi_edges[0], +2π))
        ne, T                    field values at midpoint (NaN where skipped)
        ir, it, ip               assigned cell indices (-1 where skipped)
        kept                     bool mask of contributing segments
        crossings                the sorted s-values
        r_edges, theta_edges, phi_edges, full_theta, full_phi, phi_span
                                 the domain description used (for the voxel proof)
        ray_origin, e_los, Smax, Xp, Yp
        s_front, s_stop, on_disk, kept_after_clip
        R_at_T, I_per_seg, I_total, I_total_clipped   (if T_resp & R_resp given)
    """
    dom = prepare_domain(r_ori, theta_ori, phi_ori, tol)
    r_edges     = dom['r_edges']
    theta_edges = dom['theta_edges']
    phi_edges   = dom['phi_edges']
    theta_cross = dom['theta_cross']
    phi_cross   = dom['phi_cross']
    full_theta  = dom['full_theta']
    full_phi    = dom['full_phi']
    phi_span    = dom['phi_span']
    r_min, r_max = dom['r_min'], dom['r_max']

    # ── Ray geometry + photospheric clip (unchanged EUV physics) ────────────
    Smax = float(r_ori.max())
    b2      = Xp * Xp + Yp * Yp
    on_disk = b2 <= 1.0
    s_front = -np.sqrt(1.0 - b2) if on_disk else None
    s_stop  = s_front if on_disk else Smax

    rx = Xp * x_img[0] + Yp * y_img[0]
    ry = Xp * x_img[1] + Yp * y_img[1]
    rz = Xp * x_img[2] + Yp * y_img[2]
    ex, ey, ez = float(e_los[0]), float(e_los[1]), float(e_los[2])

    # ── Boundary crossings — via the SAME shared geometry helpers the
    # integrators use, so the split surfaces are guaranteed identical ───────
    crossings = [-Smax, Smax]
    for s in _radial_intersections(rx, ry, rz, ex, ey, ez, r_edges):
        if -Smax < s < Smax:
            crossings.append(s)
    for s in _theta_intersections(rx, ry, rz, ex, ey, ez, theta_cross):
        if -Smax < s < Smax:
            crossings.append(s)
    for s in _phi_intersections(rx, ry, rz, ex, ey, ez, phi_cross):
        if -Smax < s < Smax:
            crossings.append(s)

    crossings = np.sort(np.array(crossings))
    nseg      = len(crossings) - 1

    s_mid = 0.5 * (crossings[:-1] + crossings[1:])
    ds    = np.diff(crossings)
    r_arr  = np.zeros(nseg)
    th_arr = np.zeros(nseg)
    ph_arr = np.zeros(nseg)
    ne_arr = np.full(nseg, np.nan)
    T_arr  = np.full(nseg, np.nan)
    ir_arr = np.full(nseg, -1, dtype=np.int32)
    it_arr = np.full(nseg, -1, dtype=np.int32)
    ip_arr = np.full(nseg, -1, dtype=np.int32)
    kept   = np.zeros(nseg, dtype=bool)

    nr, nt, np_ = r_ori.size, theta_ori.size, phi_ori.size
    twopi = 2.0 * np.pi

    # ── Walk segments (mirrors the inner loop of siddon_integrate) ──────────
    for k in range(nseg):
        sm = s_mid[k]
        pmx = rx + sm * ex
        pmy = ry + sm * ey
        pmz = rz + sm * ez
        rm  = np.sqrt(pmx * pmx + pmy * pmy + pmz * pmz)

        # spherical coords for EVERY segment (continuous curves for the plot);
        # phi is folded into the domain window so it never jumps at the seam.
        th  = np.arccos(np.clip(pmz / rm, -1.0, 1.0))
        dph = (np.arctan2(pmy, pmx) - phi_edges[0]) % twopi
        ph  = phi_edges[0] + dph
        r_arr[k] = rm
        th_arr[k] = th
        ph_arr[k] = ph

        # gating for contribution: skip zero-length segments and anything
        # outside the domain / inside the opaque body (matches the integrator)
        if ds[k] <= 0:
            continue
        if rm < r_min or rm > r_max or rm < r_occ:
            continue
        if (not full_theta) and (th < theta_edges[0] or th > theta_edges[-1]):
            continue
        if (not full_phi) and (dph > phi_span):
            continue

        # nearest-cell lookup by digitising into EDGES
        ir = np.searchsorted(r_edges,     rm, side='right') - 1
        it = np.searchsorted(theta_edges, th, side='right') - 1
        ip = np.searchsorted(phi_edges,   ph, side='right') - 1
        ir = min(max(ir, 0), nr  - 1)
        it = min(max(it, 0), nt  - 1)
        ip = min(max(ip, 0), np_ - 1)

        ne_arr[k] = ne_grid[ir, it, ip]
        T_arr[k]  = T_grid[ir,  it, ip]
        ir_arr[k] = ir
        it_arr[k] = it
        ip_arr[k] = ip
        kept[k]   = True

    out = dict(
        s_mid=s_mid, ds=ds, crossings=crossings,
        r=r_arr, theta=th_arr, phi=ph_arr,
        ne=ne_arr, T=T_arr, kept=kept,
        ir=ir_arr, it=it_arr, ip=ip_arr,
        r_edges=r_edges, theta_edges=theta_edges, phi_edges=phi_edges,
        full_theta=full_theta, full_phi=full_phi, phi_span=phi_span,
        ray_origin=np.array([rx, ry, rz]),
        e_los=np.array([ex, ey, ez]),
        Smax=Smax, Xp=Xp, Yp=Yp,
    )

    # ── Optional per-segment integrand ─────────────────────────────────────
    # R(T) via LOG-LOG interpolation — identical to the rrt.euv kernel, so the
    # per-segment sum below reproduces run_siddon for the same pixel.
    if T_resp is not None and R_resp is not None:
        R_at_T    = response_at(T_arr, T_resp, R_resp)
        I_per_seg = np.where(kept, ne_arr ** 2 * R_at_T * ds * Rsun_cm, 0.0)
        out['R_at_T']    = R_at_T
        out['I_per_seg'] = I_per_seg
        out['I_total']   = float(np.nansum(I_per_seg))

    # ── Photospheric-clip annotations ──────────────────────────────────────
    kept_after_clip = kept & (s_mid <= s_stop)
    out['s_front']         = s_front
    out['s_stop']          = s_stop
    out['on_disk']         = on_disk
    out['kept_after_clip'] = kept_after_clip
    if 'I_per_seg' in out:
        out['I_total_clipped'] = float(
            np.nansum(np.where(kept_after_clip, out['I_per_seg'], 0.0))
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-panel profile plot  (r, Δs, n_e, T, per-segment integrand)
# ─────────────────────────────────────────────────────────────────────────────

def plot_los_profile(samples, title=None):
    """
    Stack of panels: r(s), Δs, n_e, T, [contribution per segment if available].
    All panels share the s-axis so you can read off cell boundaries vertically.
    """
    s    = samples['s_mid']
    ds   = samples['ds']
    r    = samples['r']
    ne   = samples['ne']
    T    = samples['T']
    kept = samples['kept']

    has_int = 'I_per_seg' in samples
    nrows   = 5 if has_int else 4

    fig, axes = plt.subplots(nrows, 1, sharex=True,
                             figsize=(8.5, 1.7 * nrows + 0.6))

    # Panel 1 ── radial position along the ray
    axes[0].plot(s, r, '-', lw=0.7, color='0.55')
    axes[0].plot(s[kept], r[kept], 'o', ms=2.5, color='C0')
    axes[0].axhline(1.0, color='gray', ls=':', lw=0.7)
    axes[0].set_ylabel(r'$r(s)$  [$R_\odot$]')
    axes[0].grid(alpha=0.3)
    axes[0].text(0.99, 0.05, r'dotted: $r=1\,R_\odot$',
                 transform=axes[0].transAxes, ha='right', va='bottom',
                 fontsize=8, color='gray')

    # Panel 2 ── segment length (stem plot — discrete by nature)
    axes[1].vlines(s, 0, ds, lw=0.5, color='C1', alpha=0.7)
    axes[1].plot(s, ds, '.', ms=2, color='C1')
    axes[1].set_ylabel(r'$\Delta s$  [$R_\odot$]')
    axes[1].grid(alpha=0.3)

    # Panel 3 ── electron density (log)
    ne_plot = np.where(kept, ne, np.nan)
    axes[2].semilogy(s, ne_plot, 'o-', ms=2.5, lw=0.6, color='C2')
    axes[2].set_ylabel(r'$n_e$  [cm$^{-3}$]')
    axes[2].grid(alpha=0.3, which='both')

    # Panel 4 ── temperature (log)
    T_plot = np.where(kept, T, np.nan)
    axes[3].semilogy(s, T_plot, 'o-', ms=2.5, lw=0.6, color='C3')
    axes[3].set_ylabel(r'$T$  [K]')
    axes[3].grid(alpha=0.3, which='both')

    # Panel 5 ── per-segment integrand contribution (only if response given)
    if has_int:
        I = samples['I_per_seg']
        I_plot = np.where(I > 0, I, np.nan)
        axes[4].vlines(s, np.nanmin(I_plot[I_plot > 0]) if np.any(I_plot > 0) else 1e-30,
                       I_plot, lw=0.6, color='C4', alpha=0.75)
        axes[4].set_yscale('log')
        axes[4].set_ylabel('per-segment\n[DN/s/pix]')
        axes[4].grid(alpha=0.3, which='both')
        # Σ summary — show the post-clip total (which is what the plot displays)
        if 'I_total_clipped' in samples and samples.get('s_front') is not None:
            txt = f"Σ = {samples['I_total_clipped']:.2e} DN/s/pix"
        else:
            txt = f"Σ = {samples['I_total']:.2e} DN/s/pix"
        axes[4].text(0.99, 0.92, txt,
                     transform=axes[4].transAxes, ha='right', va='top',
                     fontsize=9, color='C4')

    # ── Photospheric clip: crop the view to what the patched integrator
    # actually integrates. For off-limb pixels this is a no-op (s_stop = Smax).
    s_stop = samples.get('s_stop', samples['Smax'])
    s_min  = -samples['Smax']
    for ax in axes:
        ax.set_xlim(s_min, s_stop)

    axes[-1].set_xlabel(r'LOS parameter  $s$  [$R_\odot$]')

    if title is None:
        title = (f"LOS through  (X={samples['Xp']:+.3f}, "
                 f"Y={samples['Yp']:+.3f}) $R_\\odot$    |    "
                 f"{int(kept.sum())} of {len(s)} segments contribute")
    fig.suptitle(title, fontsize=10, y=0.995)
    plt.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Voxel-assignment staircase  (r / theta / phi variation vs s)
# ─────────────────────────────────────────────────────────────────────────────

def _assigned_edges(edges, idx, n_cen):
    """(edge_lo, edge_hi) for the cell each idx points to, from precomputed edges."""
    idx = np.clip(idx, 0, n_cen - 1)
    return edges[idx], edges[idx + 1]


def plot_los_voxel_proof(
    Xp, Yp,
    x_img, y_img, e_los,
    r_ori, theta_ori, phi_ori,
    ne_grid, T_grid,
    samples=None,        # precomputed sample_along_los() dict; walk if None
    deg=True,            # plot theta, phi in degrees
    savepath=None,
    show=True,
):
    """r/theta/phi staircase proof that the ray reads the correct voxel.

    Uses the SAME edges and cell indices the walker (and hence the integrator)
    used, so a kept coordinate lies inside its shaded band by construction — for
    a wedge or a full disk alike. phi is plotted in the folded domain window, so
    there is no artificial 2π jump at the seam."""

    b = np.hypot(Xp, Yp)
    if samples is None:
        s = sample_along_los(Xp, Yp, x_img, y_img, e_los,
                             r_ori, theta_ori, phi_ori, ne_grid, T_grid)
    else:
        s = samples

    smid = s['s_mid']
    r    = s['r']
    th   = s['theta']
    ph   = s['phi']
    ds   = s['ds']
    k    = s['kept']

    if k.sum() == 0:
        raise ValueError(
            f'Pixel (X={Xp}, Y={Yp}) has 0 kept samples — its LOS never enters '
            f'the domain. Pick a pixel whose line of sight crosses the grid.')

    # ── Assigned cell boundaries — exactly the edges the walker digitised ────
    r_lo,  r_hi  = _assigned_edges(s['r_edges'],     s['ir'], r_ori.size)
    th_lo, th_hi = _assigned_edges(s['theta_edges'], s['it'], theta_ori.size)
    ph_lo, ph_hi = _assigned_edges(s['phi_edges'],   s['ip'], phi_ori.size)

    # ── The claim, as a number: every kept coord inside its assigned box? ────
    eps = 1e-9
    in_r  = (r  >= r_lo  - eps) & (r  <= r_hi  + eps)
    in_th = (th >= th_lo - eps) & (th <= th_hi + eps)
    in_ph = (ph >= ph_lo - eps) & (ph <= ph_hi + eps)
    inside = in_r & in_th & in_ph
    frac_inside = inside[k].mean() * 100.0
    all_inside  = bool(inside[k].all())

    # ── Supporting geometric lemmas ──────────────────────────────────────────
    A = np.polyfit(smid, r**2, 2);  s0 = -A[1] / (2 * A[0])
    min_r = np.nanmin(r)                                  # discrete (for reporting)
    r2_vertex = A[2] - A[1]**2 / (4.0 * A[0])             # continuous min of the fit
    t1 = abs(np.sqrt(max(r2_vertex, 0.0)) - b) < 1e-6     # closest approach == b
    res_r = np.nanmax(np.abs(r - np.sqrt(b**2 + (smid - s0)**2)));  t2 = res_r < 1e-9
    rx, ry, rz = s['ray_origin'];  ex, ey, ez = s['e_los']
    x = r*np.sin(th)*np.cos(ph); y = r*np.sin(th)*np.sin(ph); z = r*np.cos(th)
    rt = np.sqrt((x-(rx+smid*ex))**2 + (y-(ry+smid*ey))**2 + (z-(rz+smid*ez))**2)
    res_rt = np.nanmax(rt[k]);  t3 = res_rt < 1e-9

    # ── Plot ──────────────────────────────────────────────────────────────────
    conv = np.degrees if deg else (lambda v: v)
    ang_unit = 'deg' if deg else 'rad'

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8.5, 9.5))

    panels = [
        (ax[0], r,        r_lo,        r_hi,        r'$r$ [$R_\odot$]',        False),
        (ax[1], conv(th), conv(th_lo), conv(th_hi), rf'$\theta$ [{ang_unit}]', True),
        (ax[2], conv(ph), conv(ph_lo), conv(ph_hi), rf'$\phi$ [{ang_unit}]',   True),
    ]

    kidx = np.where(k)[0]
    s_kept = smid[k]
    z_s0, z_s1 = s_kept.min(), s_kept.max()
    s_pad = 0.05 * (z_s1 - z_s0 + 1e-9)

    def _draw(a, coord, lo, hi):
        for n in kidx:
            a.fill_between([smid[n] - ds[n]/2, smid[n] + ds[n]/2],
                           [lo[n], lo[n]], [hi[n], hi[n]],
                           color='C2', alpha=0.30, linewidth=0, zorder=1)
        a.plot(smid, coord, '-', color='0.7', lw=0.8, zorder=2,
               label='coordinate along ray')
        a.plot(smid[k], coord[k], 'o', ms=4, color='C0', zorder=3,
               label='contributing sample')

    for a, coord, lo, hi, ylab, is_ang in panels:
        _draw(a, coord, lo, hi)
        a.set_ylabel(ylab)
        a.grid(alpha=0.25)
        y_lo = min(lo[k].min(), coord[k].min())
        y_hi = max(hi[k].max(), coord[k].max())
        y_pad = 0.08 * (y_hi - y_lo + 1e-9)
        a.set_ylim(y_lo - y_pad, y_hi + y_pad)
        a.set_xlim(z_s0 - s_pad, z_s1 + s_pad)

    ax[0].axhline(1.0, color='red', ls='--', lw=1.0, zorder=2, label=r'$r=1$')
    ax[0].axhline(b,   color='gray', ls=':', lw=1.0, zorder=2,
                  label=f'b = {b:.3f}')

    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, fontsize=8, loc='upper right', ncol=2)

    ax[2].set_xlabel(r'LOS position $s$ [$R_\odot$]')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f'Saved -> {savepath}')
    if show:
        plt.show()

    return {
        'b': b, 'n_kept': int(k.sum()),
        'frac_inside_pct': frac_inside, 'all_inside': all_inside,
        'min_r': min_r, 'resid_quadratic': res_r, 'roundtrip_err': res_rt,
        'pass': all_inside and t1 and t2 and t3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: walk once, draw both diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def analyze_los(
    Xp, Yp,
    x_img, y_img, e_los,
    r_ori, theta_ori, phi_ori,
    ne_grid, T_grid,
    T_resp=None, R_resp=None,
    Rsun_cm=6.96e10,
    deg=True,
    show_profile=True,
    show_voxel_proof=True,
    profile_title=None,
    voxel_savepath=None,
    show=True,
):
    """
    Walk one LOS a single time and produce the per-segment profile and the
    r/theta/phi voxel-staircase from the SAME samples.

    Returns the samples dict (with the voxel-proof verdict merged in under the
    key 'voxel_proof' when that panel is drawn).
    """
    samples = sample_along_los(
        Xp, Yp, x_img, y_img, e_los,
        r_ori, theta_ori, phi_ori, ne_grid, T_grid,
        T_resp=T_resp, R_resp=R_resp, Rsun_cm=Rsun_cm,
    )

    if show_profile:
        plot_los_profile(samples, title=profile_title)

    if show_voxel_proof:
        verdict = plot_los_voxel_proof(
            Xp, Yp, x_img, y_img, e_los,
            r_ori, theta_ori, phi_ori, ne_grid, T_grid,
            samples=samples, deg=deg, savepath=voxel_savepath, show=show,
        )
        samples['voxel_proof'] = verdict

    return samples
