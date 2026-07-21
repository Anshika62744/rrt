"""
rrt — Rectilinear Ray Tracing of spherical MHD cubes into synthetic observables.

Given electron density (and optionally temperature) on a spherical ``(r, θ, φ)``
grid, ``rrt`` integrates along each image-plane pixel's line of sight to produce:

* synthetic **EUV / SDO-AIA** emission maps (``rrt.euv``),
* synthetic **white-light polarised-brightness (pB)** maps (``rrt.wl``), and
* single **line-of-sight diagnostics** that verify the ray tracing (``rrt.los``).

All three share one geometry core (``rrt.geometry``): analytic ray/shell,
ray/cone and ray/half-plane intersections, cell-edge construction, and the
domain detector ``prepare_domain`` (wedge / polar cap / full shell / full disk).

The public API is re-exported here for convenience; see ``docs/API.md``.
"""

__version__ = "0.1.0"

# Geometry core (always available — pure numpy/numba, no heavy optional deps).
from rrt.geometry import make_edges, prepare_domain  # noqa: E402,F401

__all__ = [
    "__version__",
    "make_edges",
    "prepare_domain",
]

# Higher-level modules are imported lazily/opportunistically so that a partially
# installed environment (e.g. missing sunpy for colormaps) still exposes the
# geometry core. Each block is additive and independent.
try:
    from rrt.euv import run_siddon, siddon_integrate  # noqa: F401
    __all__ += ["run_siddon", "siddon_integrate"]
except Exception:  # pragma: no cover - defensive import
    pass

try:
    from rrt.wl import run_siddon_pB, siddon_integrate_pB  # noqa: F401
    __all__ += ["run_siddon_pB", "siddon_integrate_pB"]
except Exception:  # pragma: no cover
    pass

try:
    from rrt.los import (  # noqa: F401
        sample_along_los,
        plot_los_profile,
        plot_los_voxel_proof,
        analyze_los,
    )
    __all__ += [
        "sample_along_los",
        "plot_los_profile",
        "plot_los_voxel_proof",
        "analyze_los",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from rrt.io import load_mhd, load_response, MHDCube  # noqa: F401
    __all__ += ["load_mhd", "load_response", "MHDCube"]
except Exception:  # pragma: no cover
    pass

try:
    from rrt.observer import make_observer, image_grid  # noqa: F401
    __all__ += ["make_observer", "image_grid"]
except Exception:  # pragma: no cover
    pass
