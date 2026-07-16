"""Price assignment mechanisms.

Phase 1: fixed price and randomized price cells (ZipRecruiter-style RCT).
Phase 3 stubs: surge-with-rounding (Uber RD) and switchback schedules.
Phase 4 stub: endogenous rule (price responds to demand shocks) — see docs/05.
"""
from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def assign_prices(cfg: SimulationConfig, n: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (quoted_price, cell_label) for n sessions."""
    if cfg.price_mechanism == "fixed":
        prices = np.full(n, cfg.fixed_price)
        cells = np.full(n, f"fixed_{cfg.fixed_price:g}", dtype=object)
        return prices, cells

    if cfg.price_mechanism == "random_cell":
        cells_arr = np.array(cfg.price_cells)
        w = (np.array(cfg.cell_weights) if cfg.cell_weights is not None
             else np.ones(len(cells_arr)) / len(cells_arr))
        idx = rng.choice(len(cells_arr), size=n, p=w / w.sum())
        prices = cells_arr[idx]
        cells = np.array([f"cell_{p:g}" for p in prices], dtype=object)
        return prices, cells

    raise NotImplementedError(
        f"mechanism '{cfg.price_mechanism}' arrives in a later phase; see docs/00_spec.md")


# --------------------------------------------------------------- Phase 3 stubs
def surge_price(base_price: float, latent_multiplier: np.ndarray,
                step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Uber-style rounding: continuous latent multiplier -> discrete surge levels.

    Returns (quoted_price, latent_minus_threshold) — the second array is the
    RD running variable (oracle view), distance to the nearest rounding cutpoint.
    """
    rounded = np.round(latent_multiplier / step) * step
    cutpoints = (np.floor(latent_multiplier / step) + 0.5) * step
    running = latent_multiplier - cutpoints
    return base_price * rounded, running
