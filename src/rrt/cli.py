"""
Console entry points: ``rrt-euv``, ``rrt-wl``, ``rrt-los``.

Each reads a YAML/JSON config (see :mod:`rrt.config`), loads the cube via
:func:`rrt.io.load_mhd`, and renders the requested product for every configured
view into ``output_dir``.

    rrt-euv config.yaml     # synthetic AIA map(s)
    rrt-wl  config.yaml     # white-light pB map(s) (+ optional NRGF)
    rrt-los config.yaml     # single-LOS diagnostics + coverage map
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from rrt.observer import view_angles, observer_from_view


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()


def _parse(argv, prog, desc):
    p = argparse.ArgumentParser(prog=prog, description=desc)
    p.add_argument("config", help="path to a YAML/JSON config file")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def _setup(argv, prog, desc):
    """Shared preamble: headless backend, parse, load config + cube + geometry."""
    import matplotlib
    matplotlib.use("Agg")  # drivers only ever save files
    import matplotlib.pyplot as plt  # noqa: F401

    from rrt.config import load_config, load_cube_from_config
    from rrt.observer import image_grid, make_observer

    args = _parse(argv, prog, desc)
    cfg = load_config(args.config)
    cube = load_cube_from_config(cfg)
    print(cube.summary())
    os.makedirs(cfg.output_dir, exist_ok=True)
    Xg, Yg, Rperp = image_grid(cfg.Rmax, cfg.Npix)
    return cfg, cube, Xg, Yg, Rperp, make_observer


# ─────────────────────────────────────────────────────────────────────────────

def main_euv(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    from rrt.euv import run_siddon
    from rrt.response import load_response, isothermal_response, FLAT_T, FLAT_R
    from rrt.viz import aia_cmap, plot_map

    from rrt.viz import channel_grid

    cfg, cube, Xg, Yg, _, make_observer = _setup(argv, "rrt-euv",
                                                 "Synthetic EUV/AIA maps")

    isothermal = cube.meta.get("input_case") == "density_only"
    if len(cfg.wavelengths) > 1:
        return _euv_channel_grid(cfg, cube, Xg, Yg, make_observer, isothermal,
                                 run_siddon, load_response, isothermal_response,
                                 channel_grid, FLAT_T, FLAT_R, plt)

    if cfg.response:
        T_resp, R_resp = load_response(cfg.response, cfg.wavelength)
    else:
        T_resp, R_resp = FLAT_T, FLAT_R  # pure column EM

    written = []
    for v in cfg.views:
        n_obs, e_los, x_img, y_img = observer_from_view(v)
        # For an isothermal corona render EM (flat response) then scale by R(T_iso)
        em = run_siddon(Xg, Yg, x_img, y_img, e_los,
                        cube.r, cube.theta, cube.phi, cube.ne, cube.T,
                        FLAT_T if isothermal and cfg.response else T_resp,
                        FLAT_R if isothermal and cfg.response else R_resp,
                        Npix=cfg.Npix, Rmax=cfg.Rmax, WAVELENGTH=cfg.wavelength,
                        plot=False, save=False)
        if isothermal and cfg.response:
            t_iso = float(np.median(cube.T))
            em = em * isothermal_response(t_iso, T_resp, R_resp)
        fig, _, _ = plot_map(em, cfg.Rmax, cmap=aia_cmap(cfg.wavelength),
                             log=(cfg.scale == "log"),
                             vlim=(tuple(cfg.vlim) if cfg.vlim else None),
                             cbar_label=f"AIA {cfg.wavelength} Å  [DN/s]",
                             title=f"{v['label']} — AIA {cfg.wavelength} Å")
        out = os.path.join(cfg.output_dir, f"euv_{cfg.wavelength}_{_slug(v['label'])}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")
    print(f"rrt-euv: {len(written)} image(s) → {cfg.output_dir}")
    return 0


def _euv_channel_grid(cfg, cube, Xg, Yg, make_observer, isothermal,
                      run_siddon, load_response, isothermal_response,
                      channel_grid, FLAT_T, FLAT_R, plt) -> int:
    """Multi-channel mode: one grid figure per view (config key ``wavelengths:``).

    An isothermal cube needs a single EM integration — the channels then differ
    only by the scalar R_λ(T_iso). A real temperature field needs one integration
    per channel, since R_λ(T) weights each voxel differently.
    """
    if not cfg.response:
        print("rrt-euv: 'wavelengths:' needs a response table (set response: …)")
        return 2

    written = []
    for v in cfg.views:
        n_obs, e_los, x_img, y_img = observer_from_view(v)
        geom = (f"pole {v['pole']}" if v.get("pole")
                else "φ = {:.1f}°, B₀ = {:.1f}°".format(*view_angles(v)))
        images = {}

        if isothermal:
            t_iso = float(np.median(cube.T))
            em = run_siddon(Xg, Yg, x_img, y_img, e_los,
                            cube.r, cube.theta, cube.phi, cube.ne, cube.T,
                            FLAT_T, FLAT_R, Npix=cfg.Npix, Rmax=cfg.Rmax,
                            plot=False, save=False)
            for w in cfg.wavelengths:
                T_resp, R_resp = load_response(cfg.response, w)
                images[w] = em * isothermal_response(t_iso, T_resp, R_resp)
                print(f"  {w} Å: max {images[w].max():.3e} DN/s/pix")
        else:
            for w in cfg.wavelengths:
                T_resp, R_resp = load_response(cfg.response, w)
                images[w] = run_siddon(Xg, Yg, x_img, y_img, e_los,
                                       cube.r, cube.theta, cube.phi, cube.ne, cube.T,
                                       T_resp, R_resp, Npix=cfg.Npix, Rmax=cfg.Rmax,
                                       WAVELENGTH=w, plot=False, save=False)
                print(f"  {w} Å: max {images[w].max():.3e} DN/s/pix")

        src = os.path.basename(str(cfg.data.get("density", "")))
        temp = (f"isothermal  T = {float(np.median(cube.T)):.1e} K" if isothermal
                else f"T = {cube.T.min():.2e} – {cube.T.max():.2e} K")
        panel = cfg.raw.get("panel", {}) or {}
        # colorbar: panel.vlim ({λ: [lo, hi]} or [lo, hi]) > top-level vlim >
        # per-panel auto limits spanning panel.dynamic_range below each peak
        pv = panel.get("vlim")
        if pv is None and panel.get("vlim_log") is not None:   # log10 exponents
            pl = panel["vlim_log"]
            pv = ({int(k): [10.0 ** float(a), 10.0 ** float(b)] for k, (a, b) in pl.items()}
                  if isinstance(pl, dict) else [10.0 ** float(x) for x in pl])
        if pv is None:
            pv = cfg.vlim
        if isinstance(pv, dict):
            pv = {int(k): (float(a), float(b)) for k, (a, b) in pv.items()}
        elif pv is not None:
            pv = (float(pv[0]), float(pv[1]))
        fig, _ = channel_grid(images, cfg.Rmax,
                              ncols=int(panel.get("ncols", 3)),
                              dynamic_range=float(panel.get("dynamic_range", 1.0e8)),
                              vlim=pv, log=(cfg.scale == "log"),
                              cbar=str(panel.get("cbar", "log10")),
                              cbar_ticks=int(panel.get("cbar_ticks", 6)),
                              facecolor=str(panel.get("facecolor", "black")),
                              mask_disk=bool(panel.get("mask_disk", False)),
                              suptitle=(f"Synthetic AIA — {os.path.splitext(src)[0]}  |  "
                                        f"{v['label']}: {geom}  |  {temp}"))
        out = os.path.join(cfg.output_dir, f"euv_channels_{_slug(v['label'])}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")
    print(f"rrt-euv: {len(written)} channel-grid figure(s) → {cfg.output_dir}")
    return 0


def main_wl(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    from rrt.wl import run_siddon_pB
    from rrt.viz import nrgf, plot_map, radial_flatten

    cfg, cube, Xg, Yg, _, make_observer = _setup(argv, "rrt-wl",
                                                 "White-light polarised-brightness maps")

    written = []
    for v in cfg.views:
        n_obs, e_los, x_img, y_img = observer_from_view(v)
        pb = run_siddon_pB(Xg, Yg, x_img, y_img, e_los,
                           cube.r, cube.theta, cube.phi, cube.ne,
                           Npix=cfg.Npix, Rmax=cfg.Rmax, Rocc=cfg.Rocc,
                           U_LIMB=cfg.U_LIMB, SIGMA_T=cfg.SIGMA_T,
                           plot=False, save=False)
        # radial_power k renders pB·ρ^k — flattens the ρ^-3…-5 falloff so the
        # outer corona survives a linear stretch (0 = off; ignored under NRGF)
        k = float(cfg.white_light.get("radial_power", 0.0))
        if cfg.use_nrgf:
            img, tag = nrgf(pb, Xg, Yg, r_inner=cfg.Rocc, r_outer=cfg.Rmax), "nrgf"
        elif k:
            img, tag = radial_flatten(pb, Xg, Yg, power=k), f"pb_r{k:g}"
        else:
            img, tag = pb, "pb"
        # NRGF is a signed σ-map, so it is always linear; raw pB follows cfg.scale
        fig, _, _ = plot_map(
            img, cfg.Rmax, cmap=str(cfg.white_light.get("cmap", "Oranges_r")),
            log=(cfg.scale == "log" and not cfg.use_nrgf),
            occulter=cfg.Rocc, vlim=(tuple(cfg.vlim) if cfg.vlim else None),
            # pB = ∫ n_e·K·dl is dimensionless ([cm^-3]·[cm^2]·[cm]); flattening
            # by ρ^k (ρ in R_sun) leaves it in units of R_sun^k
            cbar_label=("NRGF [σ]" if cfg.use_nrgf else
                        (rf"pB$\cdot\rho^{{{k:g}}}$  [R$_\odot^{{{k:g}}}$]" if k else "pB")),
            title=f"{v['label']} — white light ({tag})")
        out = os.path.join(cfg.output_dir, f"wl_{tag}_{_slug(v['label'])}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")
    print(f"rrt-wl: {len(written)} image(s) → {cfg.output_dir}")
    return 0


def main_los(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    from rrt.los import analyze_los
    from rrt.response import load_response

    cfg, cube, _, _, _, make_observer = _setup(argv, "rrt-los",
                                               "Single line-of-sight diagnostics")

    v = cfg.views[0]
    n_obs, e_los, x_img, y_img = observer_from_view(v)
    T_resp = R_resp = None
    if cfg.response:
        try:
            T_resp, R_resp = load_response(cfg.response, cfg.wavelength)
        except Exception as ex:
            print(f"  response unavailable ({ex}); intensity panel disabled")

    probes = cfg.los.get("probes") or [[0.0, 0.0]]
    written = 0
    for i, (X, Y) in enumerate(probes):
        try:
            samples = analyze_los(X, Y, x_img, y_img, e_los,
                                  cube.r, cube.theta, cube.phi, cube.ne, cube.T,
                                  T_resp=T_resp, R_resp=R_resp,
                                  show_profile=True, show_voxel_proof=True, show=False)
        except ValueError as ex:
            print(f"  probe ({X},{Y}) skipped — {ex}")
            continue
        # save whatever figures analyze_los produced
        for j, num in enumerate(plt.get_fignums()):
            fig = plt.figure(num)
            out = os.path.join(cfg.output_dir, f"los_probe{i}_{j}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            written += 1
        plt.close("all")
        vp = samples.get("voxel_proof", {})
        print(f"  probe ({X:+.2f},{Y:+.2f}): kept={int(samples['kept'].sum())} "
              f"verdict={'PASS' if vp.get('pass') else 'CHECK'}")
    print(f"rrt-los: {written} figure(s) → {cfg.output_dir}")
    return 0
