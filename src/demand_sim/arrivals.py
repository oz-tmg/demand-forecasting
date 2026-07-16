"""Phase 2: non-homogeneous Poisson arrival process (docs/00_spec.md §3.2).

Builds the daily NHPP intensity lambda_t for one product-store series by
composing multiplicative components:

    lambda_t = base_rate * trend * weekly * annual * price_effect * promo_lift

The price effect is the Phase 1 demand curve carried over as a
constant-elasticity multiplier; the remaining terms are the time-series
structure the forecasting literature treats as covariates (M5, Chronos-2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_multiplier(price: np.ndarray, base_price: float,
                     elasticity: float) -> np.ndarray:
    """Constant-elasticity demand-curve multiplier: (p / p0) ** elasticity.

    At p = p0 the multiplier is 1. `elasticity` must be negative (law of
    demand). This is the Phase 1 carry-over — swap in
    `demand.demand_curve` evaluated on a price grid for the full
    random-utility population model.
    """
    if elasticity >= 0:
        raise ValueError("elasticity must be < 0")
    return (price / base_price) ** elasticity


def build_intensity(dates: pd.DatetimeIndex,
                    base_rate: float,
                    trend_pct_yr: float,
                    weekly_amp: float,
                    annual_amp: float,
                    annual_phase: float,
                    price: np.ndarray,
                    base_price: float,
                    elasticity: float,
                    on_promo: np.ndarray,
                    promo_lift: float) -> np.ndarray:
    """Daily NHPP rate lambda_t for one product-store series.

    Components:
      trend   — (1 + trend_pct_yr) ** (t / 365.25), multiplicative growth/decay
      weekly  — Fri/Sat/Sun lift of +weekly_amp, Mon/Tue dip of -weekly_amp/2
      annual  — sinusoid of amplitude annual_amp with phase annual_phase
      price   — Phase 1 constant-elasticity demand curve
      promo   — multiplicative promo_lift on promo days
    """
    t = np.arange(len(dates))

    trend = (1.0 + trend_pct_yr) ** (t / 365.25)

    dow = dates.dayofweek.values  # 0=Mon .. 6=Sun
    weekly = 1.0 + weekly_amp * np.isin(dow, [4, 5, 6]).astype(float) \
                 - 0.5 * weekly_amp * np.isin(dow, [0, 1]).astype(float)

    doy = dates.dayofyear.values
    annual = 1.0 + annual_amp * np.sin(2 * np.pi * doy / 365.25 + annual_phase)

    p_mult = price_multiplier(price, base_price, elasticity)
    promo_mult = np.where(on_promo, promo_lift, 1.0)

    lam = base_rate * trend * weekly * annual * p_mult * promo_mult
    return np.clip(lam, 0.05, None)


def sample_daily_demand(lam: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw true (uncensored) daily demand: Poisson(lambda_t)."""
    return rng.poisson(lam)
