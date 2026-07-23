"""
rrt.io
──────
The unified data loader. One entry point, :func:`load_mhd`, reads a spherical MHD
cube from HDF5 and returns an :class:`MHDCube` with fields shaped ``(r, θ, φ)``,
n_e in cm⁻³ and T in Kelvin — ready for :func:`rrt.euv.run_siddon`,
:func:`rrt.wl.run_siddon_pB` and :func:`rrt.los.sample_along_los`.

It auto-detects (each override-able) the three **input cases**

  1. ``separate``     — density and temperature in two files,
  2. ``single``       — one file holding both,
  3. ``density_only`` — density alone → isothermal ``t_iso``,

the two **file layouts** (``arms``: ``coords/`` + ``vars/`` with attrs; ``mas``:
``Data`` + ``dim1/2/3``), and the messy-data concerns: ghost cells, density units
(mass CGS ↔ number), temperature units (K / MK / log10), axis order, angle units
(rad ↔ deg), and ascending-axis fixup. Domain detection itself is delegated to
:func:`rrt.geometry.prepare_domain` — it is not re-implemented here.

See ``docs/DATA_FORMAT.md`` for the expected datasets/attributes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from rrt.response import load_response  # noqa: F401  (re-exported convenience)

_TWO_PI = 2.0 * np.pi
_ARMS_KEYS = dict(r="coords/r", theta="coords/theta", phi="coords/phi", rho="vars/rho")
# candidate temperature datasets to probe for the "single file" case
_T_CANDIDATES = ("vars/T", "vars/temperature", "vars/Temp", "vars/te", "vars/temp")
# accepted temperature_units spellings → canonical mode. The docs quote the short
# forms ("K", "MK", "log10"); the long ones are the internal names.
_T_MODES = {"k": "kelvin", "kelvin": "kelvin",
            "mk": "megakelvin", "megakelvin": "megakelvin",
            "log10": "log10", "log": "log10"}


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MHDCube:
    r: np.ndarray            # (nr,)   ascending, R_sun
    theta: np.ndarray        # (nth,)  ascending, rad
    phi: np.ndarray          # (nph,)  ascending (unwrapped), rad
    ne: np.ndarray           # (nr, nth, nph)  cm^-3, contiguous
    T: np.ndarray | None     # (nr, nth, nph)  K, or None (before iso-fill)
    meta: dict = field(default_factory=dict)

    def with_isothermal(self, t_iso: float) -> "MHDCube":
        """Return a copy with T set to a constant ``t_iso`` [K]."""
        T = np.full_like(self.ne, float(t_iso))
        m = dict(self.meta, temperature=f"isothermal ({t_iso:.3e} K)", input_case="density_only")
        return MHDCube(self.r, self.theta, self.phi, self.ne, T, m)

    def summary(self) -> str:
        th = np.degrees([self.theta.min(), self.theta.max()])
        ph = np.degrees([self.phi.min(), self.phi.max()])
        lines = [
            f"MHDCube  case={self.meta.get('input_case','?')}  fmt={self.meta.get('fmt','?')}",
            f"  r     : {self.r.min():.3f} – {self.r.max():.3f} R_sun  (n={self.r.size})",
            f"  theta : {th[0]:.1f} – {th[1]:.1f} deg  (n={self.theta.size})",
            f"  phi   : {ph[0]:.1f} – {ph[1]:.1f} deg  (n={self.phi.size})",
            f"  ne    : {self.ne.min():.2e} – {self.ne.max():.2e} cm^-3   shape={self.ne.shape}",
        ]
        if self.T is not None:
            lines.append(f"  T     : {self.T.min():.2e} – {self.T.max():.2e} K   "
                         f"[{self.meta.get('temperature','?')}]")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level format readers → (r, theta, phi, cube_in_rtp_order)
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(path: str) -> str:
    """Return ``'mas'`` or ``'arms'`` by sniffing the file's top-level layout."""
    import h5py
    with h5py.File(os.path.expanduser(path), "r") as f:
        if "Data" in f and "dim1" in f:
            return "mas"
        if "coords" in f or any(k.startswith("vars") for k in f):
            return "arms"
    # default: assume ARMS-style
    return "arms"


def _detect_nghost(f, override) -> int:
    if override != "auto":
        return int(override)
    flag = f.attrs.get("ghost_cells_included", None)
    if flag is not None and not bool(np.asarray(flag).ravel()[0]):
        return 0
    for a in ("NGHOST", "nghost", "num_ghost", "Nghost", "ghost_zones", "num_ghosts"):
        if a in f.attrs:
            return int(np.asarray(f.attrs[a]).ravel()[0])
    return 2 if (flag is not None and bool(np.asarray(flag).ravel()[0])) else 0


def _permute_to_rtp(cube, order_attr, nr, nth, nph, meta):
    """Return ``cube`` transposed to (r, θ, φ) using the axis_order attr, else a
    size match, else the (2,1,0) ARMS default (with a recorded warning)."""
    axes = None
    if order_attr:
        toks = [a.strip() for a in str(order_attr).strip("() ").split(",")]
        toks = [a for a in toks if a in ("r", "theta", "phi")]
        if len(toks) == 3:
            axes = [toks.index("r"), toks.index("theta"), toks.index("phi")]
            meta["axis_source"] = "axis_order attr"
    if axes is None:
        shp = cube.shape
        if len({nr, nth, nph}) == 3 and sorted(shp) == sorted((nr, nth, nph)):
            axes = [shp.index(nr), shp.index(nth), shp.index(nph)]
            meta["axis_source"] = "size match"
        else:
            axes = [2, 1, 0]
            meta["axis_source"] = "(2,1,0) fallback"
            meta.setdefault("warnings", []).append(
                "axis order ambiguous (equal dims, no axis_order attr) — assumed (phi,theta,r)")
    meta["axis_perm"] = tuple(axes)
    return np.transpose(cube, axes)


def _read_arms(path, var, nghost, datasets, meta):
    import h5py
    keys = dict(_ARMS_KEYS)
    if datasets:
        keys.update(datasets)
    rho_key = datasets.get("rho", f"vars/{var}") if datasets else f"vars/{var}"
    with h5py.File(path, "r") as f:
        ng = _detect_nghost(f, nghost)
        sl = slice(ng, -ng) if ng > 0 else slice(None)
        r = np.asarray(f[keys["r"]][sl], dtype=np.float64)
        th = np.asarray(f[keys["theta"]][sl], dtype=np.float64)
        ph = np.asarray(f[keys["phi"]][sl], dtype=np.float64)
        dset = f[rho_key]
        # Prefer an explicit axis_order attr; if absent, leave None so the
        # permutation falls through to size-matching, then the (2,1,0) default.
        order = dset.attrs.get("axis_order", f.attrs.get("axis_order", None))
        cube = np.asarray(dset[...], dtype=np.float64)
        units = str(dset.attrs.get("units", ""))
        # angle-unit hints from the coord attrs (used later)
        meta["theta_units_attr"] = str(f[keys["theta"]].attrs.get("units", ""))
        meta["phi_units_attr"] = str(f[keys["phi"]].attrs.get("units", ""))
    if ng > 0:
        cube = cube[sl, sl, sl]
    meta["nghost"] = ng
    meta["density_units_attr"] = units
    cube = _permute_to_rtp(cube, order, r.size, th.size, ph.size, meta)
    return r, th, ph, np.ascontiguousarray(cube), units


def _read_mas(path, meta):
    """MAS/PSI: a single ``Data`` cube + dim1/dim2/dim3 scales, no attributes.

    Axes are identified from coordinate ranges: r reaches furthest from the Sun;
    of the remaining two, φ has the wider span than θ.
    """
    import h5py
    with h5py.File(path, "r") as f:
        dims = [np.asarray(f[f"dim{i}"][...], dtype=np.float64) for i in (1, 2, 3)]
        data = np.asarray(f["Data"][...], dtype=np.float64)
    i_r = int(np.argmax([d.max() for d in dims]))
    rest = [i for i in (0, 1, 2) if i != i_r]
    spans = [float(d.max() - d.min()) for d in dims]
    i_ph, i_th = (rest[0], rest[1]) if spans[rest[0]] > spans[rest[1]] else (rest[1], rest[0])
    r, th, ph = dims[i_r], dims[i_th], dims[i_ph]
    cube = np.transpose(data, [i_r, i_th, i_ph])
    meta["nghost"] = 0
    meta["density_units_attr"] = ""
    meta["axis_perm"] = (i_r, i_th, i_ph)
    meta["axis_source"] = "MAS span heuristic"
    return r, th, ph, np.ascontiguousarray(cube), ""


def _read_field(path, fmt, var, nghost, datasets, meta):
    fmt = detect_format(path) if fmt == "auto" else fmt
    meta["fmt"] = fmt
    if fmt == "mas":
        return _read_mas(path, meta)
    return _read_arms(path, var, nghost, datasets, meta)


# ─────────────────────────────────────────────────────────────────────────────
# Unit / axis normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _angles_to_rad(theta, phi, override, meta):
    if override == "deg":
        return np.deg2rad(theta), np.deg2rad(phi)
    if override == "rad":
        return theta, phi
    # auto
    if np.nanmax(theta) > np.pi + 0.05:
        theta = np.deg2rad(theta); meta.setdefault("converted", []).append("theta deg→rad")
    if np.nanmax(np.abs(phi)) > _TWO_PI + 0.05:
        phi = np.deg2rad(phi); meta.setdefault("converted", []).append("phi deg→rad")
    return theta, phi


def _to_number_density(rho, units, override, mu, m_p, meta):
    posmed = np.median(rho[rho > 0]) if np.any(rho > 0) else 0.0
    is_mass = (
        override == "mass_cgs"
        or (override == "auto" and "g" in units.lower() and "cm" in units.lower())
        or (override == "auto" and 0.0 < posmed < 1e-5)
    )
    if override == "number_cgs":
        is_mass = False
    meta["density"] = "mass→number" if is_mass else "number"
    return (rho / (mu * m_p)) if is_mass else rho


def _to_kelvin(vals, units, override, meta):
    finite = vals[np.isfinite(vals) & (vals > 0)]
    med = np.median(finite) if finite.size else 0.0
    mode = str(override).lower()
    if mode != "auto":
        if mode not in _T_MODES:
            raise ValueError(
                f"temperature_units={override!r} not recognised; choose from "
                f"auto, K, MK, log10")
        mode = _T_MODES[mode]
    if mode == "auto":
        u = units.lower()
        if "log" in u:
            mode = "log10"
        elif u in ("mk", "megakelvin"):
            mode = "megakelvin"
        elif "k" in u:
            mode = "kelvin"
        elif 3.5 <= med <= 9.0:
            mode = "log10"
        elif 0.05 <= med <= 50.0:
            mode = "megakelvin"
        else:
            mode = "kelvin"
    meta["temperature"] = {"log10": "log10(T/K)", "megakelvin": "MK", "kelvin": "K"}[mode]
    if mode == "log10":
        return 10.0 ** vals
    if mode == "megakelvin":
        return vals * 1.0e6
    return vals


def _ascending(r, theta, phi, *cubes):
    """Force every coordinate ascending, flipping the matching data axis."""
    coords = [r, theta, phi]
    out_cubes = list(cubes)
    for ax, c in enumerate(coords):
        if c.size > 1 and c[0] > c[-1]:
            coords[ax] = c[::-1]
            out_cubes = [np.flip(g, axis=ax) for g in out_cubes]
    coords = [np.ascontiguousarray(c) for c in coords]
    out_cubes = [np.ascontiguousarray(g) for g in out_cubes]
    return (*coords, *out_cubes)


def _clip_radial(r, cubes, r_range):
    if r_range is None:
        return r, cubes
    lo, hi = r_range
    m = np.ones(r.shape, bool)
    if lo is not None:
        m &= r >= lo
    if hi is not None:
        m &= r <= hi
    return np.ascontiguousarray(r[m]), [np.ascontiguousarray(g[m]) for g in cubes]


# ─────────────────────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────────────────────

def _split_arg(arg):
    """Accept ``path`` or ``(path, dataset_key)``."""
    if isinstance(arg, (tuple, list)):
        return arg[0], (arg[1] if len(arg) > 1 else None)
    return arg, None


def load_mhd(
    density,
    temperature=None,
    *,
    input_case="auto",
    fmt="auto",
    nghost="auto",
    density_units="auto",
    temperature_units="auto",
    angle_units="auto",
    t_iso=1.0e6,
    ne_scale=1.0,
    t_scale=1.0,
    mu=0.6,
    m_p=1.6726e-24,
    r_range=None,
    datasets=None,
) -> MHDCube:
    """Load a spherical MHD cube. See the module docstring for the full contract.

    Parameters
    ----------
    density : str | (str, str)
        Path to the density file, or ``(path, dataset_key)``.
    temperature : str | None
        A **path** to a separate temperature file (→ ``separate`` case), a
        **dataset key** inside the density file (→ ``single`` case), or ``None``
        (auto-probe the density file; fall back to isothermal ``t_iso``).
    input_case, fmt, nghost, density_units, temperature_units, angle_units
        Override the corresponding auto-detection (``"auto"`` = detect).
    t_iso : float
        Isothermal fallback temperature [K] for the density-only case.
    ne_scale, t_scale : float
        Multiplicative code-unit → physical scale factors (MAS).
    r_range : (float, float) | None
        Optional inclusive radial clip [R_sun].
    datasets : dict | None
        Override ARMS dataset keys (``r/theta/phi/rho``).

    Returns
    -------
    MHDCube
    """
    meta: dict = {}
    # Coerce numerics up front: YAML 1.1 parses "1.0e6" as a *string* (the
    # exponent needs a sign to be a float), so configs can deliver strings here.
    t_iso = float(t_iso); ne_scale = float(ne_scale); t_scale = float(t_scale)
    mu = float(mu); m_p = float(m_p)
    dens_path, dens_key = _split_arg(density)
    dens_path = os.path.expanduser(dens_path)          # allow ~ in config paths
    if temperature is not None and isinstance(temperature, str):
        _exp = os.path.expanduser(temperature)
        if os.path.exists(_exp):                       # a path (not a dataset key)
            temperature = _exp
    var = dens_key if dens_key else "rho"
    # strip a leading "vars/" if the caller passed a full key as the var
    if dens_key and "/" in str(dens_key):
        datasets = dict(datasets or {}, rho=dens_key)
        var = dens_key

    r, th, ph, ne_raw, dunits = _read_field(dens_path, fmt, var, nghost, datasets, meta)
    meta["density_source"] = dens_path

    # units / angles
    th, ph = _angles_to_rad(th, ph, angle_units, meta)
    ne = _to_number_density(ne_raw, dunits, density_units, mu, m_p, meta) * ne_scale

    # ── temperature / input case ─────────────────────────────────────────────
    T = None
    case = input_case
    temp_is_path = temperature is not None and os.path.exists(str(temperature))

    if case == "auto":
        if temp_is_path:
            case = "separate"
        elif temperature is not None:
            case = "single"
        else:
            case = "probe"  # decide after looking in the density file

    if case in ("separate",) or (input_case == "separate"):
        tmeta = {}
        _, _, _, T_raw, tunits = _read_field(str(temperature), fmt, "T", nghost, None, tmeta)
        T = _to_kelvin(T_raw, tunits, temperature_units, meta) * t_scale
        meta["temperature_source"] = str(temperature)
        case = "separate"
    elif case in ("single", "probe") or (input_case == "single"):
        key = temperature if (temperature and not temp_is_path) else None
        T_raw, tunits, found_key = _probe_temperature(dens_path, meta.get("fmt", "arms"),
                                                       key, nghost, datasets, meta)
        if T_raw is not None:
            # bring T into (r,θ,φ) the same way density was permuted. T_raw is in
            # the file's native order, identical to how density was stored, so the
            # same permutation applies (do it unconditionally — a shape compare is
            # unsafe when two dims are equal).
            perm = meta.get("axis_perm", (2, 1, 0))
            if meta.get("fmt") == "arms" and T_raw.ndim == 3:
                T_raw = np.transpose(T_raw, perm)
            T = _to_kelvin(np.ascontiguousarray(T_raw), tunits, temperature_units, meta) * t_scale
            meta["temperature_source"] = f"{dens_path}:{found_key}"
            case = "single"
        else:
            case = "density_only"

    if T is None or case == "density_only":
        case = "density_only"
        T = np.full_like(ne, float(t_iso))
        meta["temperature"] = f"isothermal ({t_iso:.3e} K)"
        meta["temperature_source"] = None

    meta["input_case"] = case

    # ── ascending + optional radial clip ─────────────────────────────────────
    r, th, ph, ne, T = _ascending(r, th, ph, ne, T)
    r, (ne, T) = _clip_radial(r, [ne, T], r_range)

    return MHDCube(r=r, theta=th, phi=ph,
                   ne=np.ascontiguousarray(ne, dtype=np.float64),
                   T=np.ascontiguousarray(T, dtype=np.float64),
                   meta=meta)


def _probe_temperature(path, fmt, explicit_key, nghost, datasets, meta):
    """Look for a temperature dataset inside a single file.

    Returns ``(T_raw_or_None, units, key)``. ``T_raw`` is returned in the file's
    native axis order (the caller permutes it to match density).
    """
    import h5py
    with h5py.File(os.path.expanduser(path), "r") as f:
        ng = _detect_nghost(f, nghost)
        sl = slice(ng, -ng) if ng > 0 else slice(None)
        candidates = [explicit_key] if explicit_key else list(_T_CANDIDATES)
        for key in candidates:
            if key and key in f:
                arr = np.asarray(f[key][...], dtype=np.float64)
                if ng > 0 and arr.ndim == 3:
                    arr = arr[sl, sl, sl]
                units = str(f[key].attrs.get("units", ""))
                return arr, units, key
    return None, "", None
