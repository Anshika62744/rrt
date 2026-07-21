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

from rrt.observer import view_angles


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

    cfg, cube, Xg, Yg, _, make_observer = _setup(argv, "rrt-euv",
                                                 "Synthetic EUV/AIA maps")

    isothermal = cube.meta.get("input_case") == "density_only"
    if cfg.response:
        T_resp, R_resp = load_response(cfg.response, cfg.wavelength)
    else:
        T_resp, R_resp = FLAT_T, FLAT_R  # pure column EM

    written = []
    for v in cfg.views:
        n_obs, e_los, x_img, y_img = make_observer(*view_angles(v))
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
        fig, _, _ = plot_map(em, cfg.Rmax, cmap=aia_cmap(cfg.wavelength), log=True,
                             cbar_label=f"AIA {cfg.wavelength} Å  [DN/s]",
                             title=f"{v['label']} — AIA {cfg.wavelength} Å")
        out = os.path.join(cfg.output_dir, f"euv_{cfg.wavelength}_{_slug(v['label'])}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")
    print(f"rrt-euv: {len(written)} image(s) → {cfg.output_dir}")
    return 0


def main_wl(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt
    from rrt.wl import run_siddon_pB
    from rrt.viz import nrgf, plot_map

    cfg, cube, Xg, Yg, _, make_observer = _setup(argv, "rrt-wl",
                                                 "White-light polarised-brightness maps")

    written = []
    for v in cfg.views:
        n_obs, e_los, x_img, y_img = make_observer(*view_angles(v))
        pb = run_siddon_pB(Xg, Yg, x_img, y_img, e_los,
                           cube.r, cube.theta, cube.phi, cube.ne,
                           Npix=cfg.Npix, Rmax=cfg.Rmax, Rocc=cfg.Rocc,
                           U_LIMB=cfg.U_LIMB, SIGMA_T=cfg.SIGMA_T,
                           plot=False, save=False)
        img = nrgf(pb, Xg, Yg, r_inner=cfg.Rocc, r_outer=cfg.Rmax) if cfg.use_nrgf else pb
        tag = "nrgf" if cfg.use_nrgf else "pb"
        fig, _, _ = plot_map(
            img, cfg.Rmax, cmap="Oranges_r", log=not cfg.use_nrgf,
            occulter=cfg.Rocc, cbar_label=("NRGF [σ]" if cfg.use_nrgf else r"pB [cm$^{-1}$]"),
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
    n_obs, e_los, x_img, y_img = make_observer(*view_angles(v))
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
