"""
rrt.config
──────────
Config parsing for the ``rrt-euv`` / ``rrt-wl`` / ``rrt-los`` console scripts.

A config is a YAML or JSON document. Only ``data.density`` is strictly required;
everything else has a sensible default. Example (YAML)::

    data:
      density: data/mhd_data_0220.h5   # ARMS wedge, density-only
      temperature: null                # path | dataset-key | null (auto)
      input_case: auto
      t_iso: 1.0e6
      nghost: auto
    response: data/aia_temp_response_chiantifix.npz
    wavelength: 171
    image: {Npix: 512, Rmax: 1.5}
    views:
      - {label: Side view, phi_obs_deg: -70, B0_deg: 0}
      - {label: Nose view, phi_obs_deg: -70, B0_deg: 89}
      # For Carrington-framed data (e.g. MAS) you may give an observation time
      # instead of angles; the sub-Earth L0 (Carrington longitude) and B0 are
      # computed via sunpy. Optional phi_offset_deg absorbs a φ zero-point shift.
      # - {label: sub-Earth, obs_time: '2024-05-08T14:09:00'}
    white_light: {Rocc: 1.0, U_LIMB: 0.63, SIGMA_T: 7.95e-26, nrgf: false}
    los: {probes: [[0.0, 0.0], [1.3, 0.0]], coverage_n: 28}
    output_dir: out/euv_wedge
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    data: dict = field(default_factory=dict)
    response: str | None = None
    wavelength: int = 171
    wavelengths: list = field(default_factory=list)
    Npix: int = 256
    Rmax: float = 1.5
    views: list = field(default_factory=lambda: [
        {"label": "Side view", "phi_obs_deg": -70.0, "B0_deg": 0.0}])
    output_dir: str = "out"
    vlim: list | None = None          # [lo, hi] colorbar limits; None → percentiles
    scale: str = "log"                # "log" | "linear" intensity scale
    white_light: dict = field(default_factory=dict)
    los: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    # ── white-light knobs (with defaults matching rrt.wl) ────────────────────
    @property
    def Rocc(self) -> float:
        return float(self.white_light.get("Rocc", 1.0))

    @property
    def U_LIMB(self) -> float:
        return float(self.white_light.get("U_LIMB", 0.63))

    @property
    def SIGMA_T(self) -> float:
        # default = classical electron radius squared r_e² (van de Hulst/Billings)
        return float(self.white_light.get("SIGMA_T", 7.95e-26))

    @property
    def use_nrgf(self) -> bool:
        return bool(self.white_light.get("nrgf", False))


def _read(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text) or {}
    if path.endswith(".json"):
        return json.loads(text)
    # try YAML first (superset), then JSON
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def load_config(path: str) -> Config:
    """Parse a YAML/JSON config file into a :class:`Config`."""
    d: dict[str, Any] = _read(path)
    if "data" not in d or "density" not in d.get("data", {}):
        raise ValueError(f"{path}: config must define data.density")
    image = d.get("image", {}) or {}
    # "wavelengths: [94, 131, ...]" renders every channel into one grid figure;
    # "wavelength: 171" (singular) stays a one-channel run.
    wls = d.get("wavelengths")
    wls = [int(w) for w in wls] if wls else [int(d.get("wavelength", 171))]
    cfg = Config(
        data=d["data"],
        response=d.get("response"),
        wavelength=int(d.get("wavelength", wls[0])),
        wavelengths=wls,
        Npix=int(image.get("Npix", 256)),
        Rmax=float(image.get("Rmax", 1.5)),
        views=d.get("views") or [{"label": "Side view", "phi_obs_deg": -70.0, "B0_deg": 0.0}],
        output_dir=d.get("output_dir", "out"),
        vlim=([float(x) for x in d["vlim"]] if d.get("vlim") else None),
        scale=str(d.get("scale", "log")).lower(),
        white_light=d.get("white_light", {}) or {},
        los=d.get("los", {}) or {},
        raw=d,
    )
    # "vlim_log: [lo, hi]" is the same limits written as log10 exponents
    if cfg.vlim is None and d.get("vlim_log"):
        cfg.vlim = [10.0 ** float(x) for x in d["vlim_log"]]
    if cfg.scale not in ("log", "linear"):
        raise ValueError(f"{path}: scale must be 'log' or 'linear', got {cfg.scale!r}")
    return cfg


def load_cube_from_config(cfg: Config):
    """Build an MHDCube from a Config's ``data`` block via rrt.io.load_mhd."""
    from rrt.io import load_mhd

    data = dict(cfg.data)
    density = data.pop("density")
    temperature = data.pop("temperature", None)
    # pass the remaining recognised keys straight through as overrides
    allowed = {"input_case", "fmt", "nghost", "density_units", "temperature_units",
               "angle_units", "t_iso", "ne_scale", "t_scale", "mu", "m_p",
               "r_range", "datasets"}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return load_mhd(density, temperature=temperature, **kwargs)
