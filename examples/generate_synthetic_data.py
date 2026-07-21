"""
Generate small synthetic HDF5 cubes so the example configs run WITHOUT any
proprietary data:

    python examples/generate_synthetic_data.py
    rrt-euv examples/configs/euv_wedge_synth.yaml
    rrt-euv examples/configs/euv_fulldisk_synth.yaml
    rrt-wl  examples/configs/wl_fulldisk_synth.yaml
    rrt-los examples/configs/los_fulldisk_synth.yaml

Writes into examples/data/ (git-ignored). Files are a few MB each.
"""
import os

from rrt import synthdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) A 40°-ish wedge, density only (→ isothermal) — like the real ARMS file.
    r, th, ph = synthdata.spherical_grid("wedge", nr=48, nth=64, nph=48, r_range=(1.0, 3.0))
    ne = synthdata.analytic_ne(r, th, ph, n0=1.0e8, aniso=True)
    p1 = os.path.join(OUT, "synth_wedge.h5")
    synthdata.write_arms(p1, r, th, ph, ne, nghost=2)          # include ghost cells
    print("wrote", p1, "(ARMS wedge, density-only, 2 ghost cells)")

    # 2) A full disk with a real temperature field, single file.
    r, th, ph = synthdata.spherical_grid("full", nr=64, nth=96, nph=128, r_range=(1.0, 6.0))
    ne = synthdata.analytic_ne(r, th, ph, n0=1.0e8, aniso=True)
    T = synthdata.analytic_T(r, th, ph, t0=1.5e6, gradient=True)
    p2 = os.path.join(OUT, "synth_fulldisk.h5")
    synthdata.write_arms(p2, r, th, ph, ne, T=T)               # T in vars/T
    print("wrote", p2, "(ARMS full disk, density+temperature)")

    # 3) A small AIA-style response table.
    p3 = os.path.join(OUT, "synth_response.npz")
    synthdata.write_response(p3, channels=(171, 193, 211))
    print("wrote", p3, "(synthetic AIA response)")


if __name__ == "__main__":
    main()
