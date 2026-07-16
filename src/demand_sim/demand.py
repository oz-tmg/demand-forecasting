"""Ground-truth demand: choice probabilities, demand curves, elasticities, surplus.

Everything in this module is ORACLE math — closed-form (or numerical) truth that
estimators are graded against. No sampling noise here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .config import SegmentConfig, SimulationConfig


# ----------------------------------------------------------------- choice probs
def purchase_prob(seg: SegmentConfig, price: np.ndarray | float,
                  model: str, promo: np.ndarray | float = 0.0) -> np.ndarray:
    """P(buy | segment, price, promo) under the chosen ground-truth model."""
    p = np.asarray(price, dtype=float)
    if model == "logit":
        util = seg.base_util + seg.price_coef * p + seg.promo_uplift * np.asarray(promo)
        return 1.0 / (1.0 + np.exp(-util))
    if model == "wtp_threshold":
        # buy iff WTP >= price; WTP ~ LogNormal(mu, sigma)
        # (promo modeled as a multiplicative WTP boost of exp(promo_uplift * promo))
        boost = np.exp(seg.promo_uplift * np.asarray(promo))
        return lognorm.sf(p / boost, s=seg.wtp_sigma, scale=np.exp(seg.wtp_mu))
    raise ValueError(f"unknown choice model: {model}")


# ------------------------------------------------------------------ true curves
def demand_curve(cfg: SimulationConfig, price_grid: np.ndarray,
                 promo: float = 0.0) -> pd.DataFrame:
    """Aggregate + per-segment expected purchase probability across a price grid.

    D(p) per session = sum_s share_s * P_s(p). Multiply by session volume to get units.
    """
    rows = {"price": price_grid}
    agg = np.zeros_like(price_grid, dtype=float)
    for seg in cfg.segments:
        ps = purchase_prob(seg, price_grid, cfg.choice_model, promo)
        rows[f"P_{seg.name}"] = ps
        agg += seg.share * ps
    rows["P_aggregate"] = agg
    return pd.DataFrame(rows)


def true_elasticity(cfg: SimulationConfig, price_grid: np.ndarray,
                    promo: float = 0.0) -> pd.DataFrame:
    """Point elasticity dlnD/dlnp on the grid.

    Logit segments have the closed form  e_s(p) = price_coef * p * (1 - P_s(p));
    the aggregate is the demand-share-weighted mixture. We compute numerically so
    the same code covers both ground-truth models (and any future ones).
    """
    eps = 1e-4
    out = {"price": price_grid}
    for label in ["aggregate"] + [s.name for s in cfg.segments]:
        col = f"P_{label}"
        up = demand_curve(cfg, price_grid * (1 + eps), promo)[col].to_numpy()
        dn = demand_curve(cfg, price_grid * (1 - eps), promo)[col].to_numpy()
        mid = demand_curve(cfg, price_grid, promo)[col].to_numpy()
        out[f"elasticity_{label}"] = (up - dn) / (2 * eps) / np.clip(mid, 1e-12, None)
    return pd.DataFrame(out)


def consumer_surplus(cfg: SimulationConfig, price: float,
                     upper: float = 500.0, n_grid: int = 2_000) -> float:
    """CS per session at a posted price: integral of P(p) from price to upper.

    The Uber exercise: with the whole curve known, surplus is just an integral.
    """
    grid = np.linspace(price, upper, n_grid)
    probs = demand_curve(cfg, grid)["P_aggregate"].to_numpy()
    return float(np.trapezoid(probs, grid))
