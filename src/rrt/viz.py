"""
rrt.viz
───────
Reusable plotting helpers shared by the drivers and example notebooks: the AIA
colormap lookup, the limb wireframe overlay, the NRGF radial filter, a single-map
plotter, and the 4-tier movie encoder.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle

# Register SDO/AIA colormaps if sunpy is available (optional).
try:
    from sunpy.visualization.colormaps import cm as _sdo_cm  # noqa: F401
    _HAVE_SUNPY = True
except Exception:  # pragma: no cover
    _HAVE_SUNPY = False


def aia_cmap(wavelength):
    """Return the SDO/AIA colormap for a channel, or 'inferno' as a fallback."""
    try:
        return plt.get_cmap(f"sdoaia{wavelength}").copy()
    except Exception:
        return plt.get_cmap("inferno").copy()


def draw_wireframe_sphere(ax, n_obs, x_img, y_img, R=1.0,
                          lon_step=30, lat_step=30, color="white", lw=0.5, alpha=0.5):
    """Overlay a solar-limb wireframe (visible meridians/parallels) on an image axis."""
    npts = 200
    for lon_deg in range(0, 360, lon_step):                 # meridians
        lon = np.deg2rad(lon_deg); lat = np.linspace(-np.pi / 2, np.pi / 2, npts)
        x3 = R * np.cos(lat) * np.cos(lon); y3 = R * np.cos(lat) * np.sin(lon); z3 = R * np.sin(lat)
        xi = x3 * x_img[0] + y3 * x_img[1] + z3 * x_img[2]
        yi = x3 * y_img[0] + y3 * y_img[1] + z3 * y_img[2]
        vis = x3 * n_obs[0] + y3 * n_obs[1] + z3 * n_obs[2]; xi[vis < 0] = np.nan
        ax.plot(xi, yi, color=color, lw=lw, alpha=alpha)
    for lat_deg in range(-90 + lat_step, 90, lat_step):     # parallels
        lat = np.deg2rad(lat_deg); lon = np.linspace(0, 2 * np.pi, npts)
        x3 = R * np.cos(lat) * np.cos(lon); y3 = R * np.cos(lat) * np.sin(lon)
        z3 = R * np.sin(lat) * np.ones_like(lon)
        xi = x3 * x_img[0] + y3 * x_img[1] + z3 * x_img[2]
        yi = x3 * y_img[0] + y3 * y_img[1] + z3 * y_img[2]
        vis = x3 * n_obs[0] + y3 * n_obs[1] + z3 * n_obs[2]; xi[vis < 0] = np.nan
        ax.plot(xi, yi, color=color, lw=lw, alpha=alpha)


def nrgf(img, Xg, Yg, r_inner=1.0, r_outer=None, n_radial_bins=120, min_pixels_per_bin=20):
    """Normalised Radial Graded Filter: ``(pixel − μ_r) / σ_r`` per radial annulus.

    Removes the steep radial falloff so streamers / CMEs stand out. Returns an
    array with NaN outside the mask or in under-populated annuli.
    """
    rho = np.sqrt(Xg ** 2 + Yg ** 2)
    if r_outer is None:
        r_outer = rho.max()
    out = np.full_like(img, np.nan, dtype=np.float64)
    valid = np.isfinite(img) & (img > 0) & (rho >= r_inner) & (rho <= r_outer)
    edges = np.linspace(r_inner, r_outer, n_radial_bins + 1)
    for i in range(n_radial_bins):
        in_bin = valid & (rho >= edges[i]) & (rho < edges[i + 1])
        if in_bin.sum() < min_pixels_per_bin:
            continue
        vals = img[in_bin]
        mu, sig = vals.mean(), vals.std()
        if sig <= 0:
            continue
        out[in_bin] = (img[in_bin] - mu) / sig
    return out


def radial_flatten(img, Xg, Yg, power=3.0, r_ref=1.0):
    """Multiply by ``(ρ/r_ref)**power`` to cancel the steep radial falloff.

    ``ρ = hypot(Xg, Yg)`` is the plane-of-sky impact parameter [R_sun]. White
    light drops roughly as ρ⁻³…ρ⁻⁵, so ``power≈3`` compresses several decades of
    dynamic range into one image and lets streamers stay visible far out —
    unlike :func:`nrgf`, it preserves the sign and the relative brightness
    *within* an annulus, it only rescales between annuli.
    """
    rho = np.maximum(np.hypot(np.asarray(Xg), np.asarray(Yg)), 1e-6)
    return np.asarray(img, dtype=float) * (rho / float(r_ref)) ** float(power)


def plot_map(img, rmax, *, cmap="viridis", log=True, mask_disk=False,
             occulter=None, cbar_label="", title="", vlim=None, wireframe=None,
             ax=None):
    """Plot one image-plane map with sensible defaults. Returns ``(fig, ax, im)``.

    ``wireframe`` may be ``(n_obs, x_img, y_img)`` to overlay the limb.
    ``occulter`` (R_sun) draws a filled black disk (coronagraph style).
    """
    img = np.asarray(img, dtype=float).copy()
    disp = img.copy()
    # Only a log scale needs non-positive pixels removed. A linear map may be a
    # signed quantity (e.g. an NRGF σ-map, where ~half the pixels are ≤ 0 by
    # construction); blanking those would delete half the image.
    if log:
        disp[disp <= 0] = np.nan
    Rperp = None
    if mask_disk or occulter is not None:
        xs = np.linspace(-rmax, rmax, img.shape[1])
        Xg, Yg = np.meshgrid(xs, xs, indexing="xy")
        Rperp = np.hypot(Xg, Yg)
    if mask_disk and Rperp is not None:
        disp[Rperp < 1.0] = np.nan
    if occulter is not None and Rperp is not None:
        disp[Rperp < occulter] = np.nan

    if vlim is None:
        pos = disp[np.isfinite(disp)]
        lo, hi = np.nanpercentile(pos, [1, 99]) if pos.size else (1e-30, 1.0)
    else:
        lo, hi = vlim
    norm = LogNorm(vmin=max(lo, 1e-30), vmax=hi) if log else Normalize(vmin=lo, vmax=hi)

    cm = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
    cm.set_bad("black")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = ax.figure
    im = ax.imshow(disp, origin="lower", extent=[-rmax, rmax, -rmax, rmax],
                   cmap=cm, norm=norm, interpolation="bilinear")
    if occulter is not None:
        ax.add_patch(Circle((0, 0), occulter, facecolor="black", edgecolor="black", zorder=5))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="0.6", lw=0.8, zorder=6))
    if wireframe is not None:
        draw_wireframe_sphere(ax, *wireframe, color="0.5")
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    if cbar_label:
        fig.colorbar(im, ax=ax, pad=0.02, label=cbar_label)
    if title:
        ax.set_title(title)
    return fig, ax, im


def channel_grid(images, rmax, *, ncols=3, dynamic_range=1e8, vlim=None, log=True,
                 cbar="log10", cbar_ticks=6, occulter=None, mask_disk=False,
                 suptitle="", panel_size=(4.8, 4.0), facecolor="black"):
    """One panel per AIA channel in a single dark figure. Returns ``(fig, axes)``.

    ``images`` maps wavelength → 2-D map (DN/s/pixel). By default each panel gets
    its own log colorbar spanning ``dynamic_range`` below that channel's peak,
    since the channels differ by orders of magnitude. Pass ``vlim=(lo, hi)`` for
    one fixed scale on every panel, or a ``{wavelength: (lo, hi)}`` dict for
    per-channel limits (channels left out fall back to ``dynamic_range``).

    ``cbar`` picks the tick style of a log scale: ``"log10"`` (default) plots
    log₁₀ values with evenly spaced numeric ticks, like a PSI/Predictive-Science
    frame; ``"decades"`` keeps a matplotlib LogNorm with 10ⁿ ticks. ``vlim`` is
    given in linear units either way.
    """
    ws = list(images)
    ncols = max(1, min(ncols, len(ws)))
    nrows = int(np.ceil(len(ws) / ncols))
    # text/ticks flip to black on a light page, white on a dark one
    from matplotlib.colors import to_rgb
    r_, g_, b_ = to_rgb(facecolor)
    fg = "black" if (0.299 * r_ + 0.587 * g_ + 0.114 * b_) > 0.5 else "white"

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.patch.set_facecolor(facecolor)

    Rperp = None
    if mask_disk or occulter is not None:
        xs = np.linspace(-rmax, rmax, np.asarray(images[ws[0]]).shape[1])
        Xg, Yg = np.meshgrid(xs, xs, indexing="xy")
        Rperp = np.hypot(Xg, Yg)

    for k, w in enumerate(ws):
        ax = axes[k // ncols][k % ncols]
        disp = np.asarray(images[w], dtype=float).copy()
        disp[disp <= 0] = np.nan
        if mask_disk and Rperp is not None:
            disp[Rperp < 1.0] = np.nan
        if occulter is not None and Rperp is not None:
            disp[Rperp < occulter] = np.nan

        dmax = np.nanmax(disp) if np.any(np.isfinite(disp)) else 1.0
        lim = vlim.get(w) if isinstance(vlim, dict) else vlim
        if lim is not None:
            vmin, vmax = float(lim[0]), float(lim[1])
        elif log:
            vmax, vmin = dmax, max(dmax / dynamic_range, 1e-30)
        else:
            vmax, vmin = dmax, 0.0
        cm = aia_cmap(w); cm.set_bad("black")

        # log10 display: plot the logged data on a linear norm, so the colorbar
        # reads -1.00 … 3.50 (PSI style) instead of 10^-1 … 10^3.5
        log10_mode = log and cbar == "log10"
        if log10_mode:
            data = np.log10(disp)
            lo, hi = np.log10(max(vmin, 1e-30)), np.log10(max(vmax, 1e-30))
            norm = Normalize(vmin=lo, vmax=hi)
        else:
            data = disp
            norm = (LogNorm(vmin=max(vmin, 1e-30), vmax=vmax) if log
                    else Normalize(vmin=vmin, vmax=vmax))
        im = ax.imshow(data, origin="lower", extent=[-rmax, rmax, -rmax, rmax],
                       cmap=cm, norm=norm, interpolation="bilinear")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046,
                          ticks=(np.linspace(lo, hi, int(cbar_ticks)) if log10_mode else None))
        if log10_mode:
            cb.ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(lo, hi, int(cbar_ticks))])
        cb.set_label((r"Log$_{10}$ DN s$^{-1}$ pix$^{-1}$" if log10_mode
                      else r"DN s$^{-1}$ pix$^{-1}$"), color=fg, fontsize=8)
        cb.ax.tick_params(colors=fg, labelsize=7)
        cb.outline.set_edgecolor(fg)

        ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="0.7", ls="--", lw=0.8))
        if occulter is not None:
            ax.add_patch(Circle((0, 0), occulter, facecolor="black", edgecolor="0.7",
                                ls="--", lw=0.8, zorder=5))
        # these sit on the black image area, so they stay white in both themes
        ax.text(0.04, 0.94, f"AIA {w} Å", transform=ax.transAxes, color="white",
                fontsize=12, fontweight="bold", va="top")
        ax.text(0.97, 0.03, f"max = {dmax:.2e}", transform=ax.transAxes, color="white",
                fontsize=8, ha="right", va="bottom")
        ax.set_facecolor("black")          # empty sky stays black in both themes
        ax.set_aspect("equal")
        ax.tick_params(colors=fg, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(fg if fg == "black" else "black")

    for k in range(len(ws), nrows * ncols):          # blank any unused cell
        axes[k // ncols][k % ncols].axis("off")

    if suptitle:
        fig.suptitle(suptitle, color=fg, fontsize=14)
    fig.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Movie encoder — 4-tier fallback: ffmpeg → imageio → opencv → GIF
# ─────────────────────────────────────────────────────────────────────────────

def make_movie(frame_paths, out_file, fps=4, frame_dir=None):
    """Assemble PNG frames into an MP4 (or GIF fallback). Returns the output path.

    Tries system ffmpeg, then imageio-ffmpeg, then opencv, then a Pillow GIF.
    ``frame_dir`` (defaults to the first frame's dir) holds the ffmpeg concat list.
    """
    frames = sorted(frame_paths)
    if not frames:
        raise RuntimeError("no frames to assemble")
    if frame_dir is None:
        frame_dir = os.path.dirname(frames[0]) or "."

    for method in (_try_ffmpeg, _try_imageio, _try_opencv):
        produced = method(frames, out_file, fps, frame_dir)
        if produced:
            return produced
    return _fallback_gif(frames, out_file, fps)


def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe, "system"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio_ffmpeg bundled"
    except ImportError:
        return None, None


def _try_ffmpeg(frames, out_file, fps, frame_dir):
    exe, source = _find_ffmpeg()
    if exe is None:
        return None
    try:
        list_file = os.path.join(frame_dir, "framelist.txt")
        with open(list_file, "w") as fh:
            for p in frames:
                fh.write(f"file '{os.path.abspath(p)}'\n")
                fh.write(f"duration {1/fps:.6f}\n")
            fh.write(f"file '{os.path.abspath(frames[-1])}'\n")
        cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_file]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return out_file
    except Exception:
        pass
    return None


def _try_imageio(frames, out_file, fps, frame_dir):
    try:
        import imageio, imageio_ffmpeg  # noqa: F401
        from PIL import Image as PILImage
        os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
        imgs = [np.array(PILImage.open(p).convert("RGB")) for p in frames]
        h, w = imgs[0].shape[:2]
        if h % 2 or w % 2:
            imgs = [np.array(PILImage.fromarray(f).resize((w + w % 2, h + h % 2))) for f in imgs]
        writer = imageio.get_writer(out_file, fps=fps, codec="libx264",
                                    macro_block_size=2,
                                    output_params=["-pix_fmt", "yuv420p", "-crf", "18"])
        for img in imgs:
            writer.append_data(img)
        writer.close()
        return out_file
    except Exception:
        return None


def _try_opencv(frames, out_file, fps, frame_dir):
    try:
        import cv2
        from PIL import Image as PILImage
        s = np.array(PILImage.open(frames[0]).convert("RGB")); h, w = s.shape[:2]
        vw = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in frames:
            vw.write(cv2.cvtColor(np.array(PILImage.open(p).convert("RGB")), cv2.COLOR_RGB2BGR))
        vw.release()
        return out_file
    except Exception:
        return None


def _fallback_gif(frames, out_file, fps):
    from PIL import Image as PILImage
    gif = out_file.replace(".mp4", ".gif")
    pil = [PILImage.open(p).convert("RGB") for p in frames]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0)
    return gif
