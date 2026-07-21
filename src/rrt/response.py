"""
rrt.response
────────────
SDO/AIA temperature-response tables and the log-log R(T) interpolation used by
the EUV integrator.

The response ``.npz`` (canonical: ``aia_temp_response_chiantifix.npz``) stores a
``logte`` grid (log10 of temperature in K) and one ``resp{λ}`` curve per channel,
e.g. ``resp94, resp131, resp171, resp193, resp211, resp304, resp335``
(DN cm⁵ s⁻¹ pixel⁻¹).

``response_at`` interpolates **log10(R) vs log10(T)** — the same convention the
``rrt.euv`` kernel uses internally — so an isothermal scalar built here matches a
full-field render to round-off.
"""

from __future__ import annotations

import os

import numpy as np

# Filenames we accept when a caller passes a directory or an alias.
_ALIASES = ("aia_temp_response_chiantifix.npz", "aia_temp_response_ssw.npz")


def _resolve(path: str) -> str:
    """Accept a file, or a directory containing a known response ``.npz``."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        for name in _ALIASES:
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            f"no AIA response .npz ({' / '.join(_ALIASES)}) found in {path!r}")
    return path


def available_channels(path: str) -> list[int]:
    """List the AIA channels present in a response file, ascending."""
    with np.load(_resolve(path)) as npz:
        chans = sorted(int(k[4:]) for k in npz.files if k.startswith("resp"))
    return chans


def load_response(path: str, wavelength: int = 171):
    """Return ``(T_resp, R_resp)`` for one AIA channel.

    ``T_resp`` is Kelvin (``10**logte``); ``R_resp`` is the channel response.
    Raises ``KeyError`` (listing available channels) if the channel is absent.
    """
    path = _resolve(path)
    with np.load(path) as npz:
        if "logte" not in npz.files:
            raise KeyError(f"{path!r} has no 'logte' grid; keys={list(npz.files)}")
        T_resp = 10.0 ** np.asarray(npz["logte"], dtype=np.float64)
        key = f"resp{wavelength}"
        if key not in npz.files:
            avail = [int(k[4:]) for k in npz.files if k.startswith("resp")]
            raise KeyError(
                f"channel {wavelength} not in {os.path.basename(path)}; "
                f"available: {sorted(avail)}")
        R_resp = np.asarray(npz[key], dtype=np.float64)
    return T_resp, R_resp


def response_at(T, T_resp, R_resp):
    """Log-log interpolate R(T). Scalar or array ``T`` (Kelvin) → same shape.

    Clamped to the table ends (flat extrapolation), matching the ``rrt.euv``
    kernel. Zero/negative response entries are floored to 1e-40 before the log.
    """
    T = np.asarray(T, dtype=np.float64)
    logT = np.log10(np.clip(T, 1e-30, None))
    logTr = np.log10(np.asarray(T_resp, dtype=np.float64))
    logRr = np.log10(np.clip(np.asarray(R_resp, dtype=np.float64), 1e-40, None))
    logR = np.interp(logT, logTr, logRr, left=logRr[0], right=logRr[-1])
    return 10.0 ** logR


def isothermal_response(t_iso, T_resp, R_resp) -> float:
    """Scalar R(T_iso) for an isothermal corona (log-log interpolation)."""
    return float(response_at(t_iso, T_resp, R_resp))


# Flat unit response → run_siddon returns pure column EM = ∫ n_e² dℓ.
FLAT_T = np.array([1.0e3, 1.0e9])
FLAT_R = np.array([1.0, 1.0])
