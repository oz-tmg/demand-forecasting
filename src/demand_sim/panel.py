"""Phase 2: forecasting dataset generator — fact_sales_daily (docs/00_spec.md §2.2).

Composes the arrival process (arrivals.py) with inventory censoring
(inventory.py) across a portfolio of product-store series and emits the
daily panel in observable and oracle views:

  observable — what a real firm logs: units_sold, price, promo, inventory
  oracle     — adds units_demanded, lost_sales, stockout, lambda_true

Estimators and forecast models must only ever see the observable view;
the oracle view is the answer key (design principles 2 and 4).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .arrivals import build_intensity, sample_daily_demand
from .config import InventoryConfig
from .inventory import simulate_inventory

ORACLE_COLS = ["units_demanded", "lost_sales", "stockout", "lambda_true"]

CATEGORIES = ("beverages", "snacks", "household", "personal_care")


@dataclass(frozen=True)
class SeriesParams:
    """Static parameters for one product-store series (dim_product_store)."""

    product_id: str
    store_id: str
    category: str
    base_price: float
    elasticity: float        # negative; constant-elasticity price response
    base_rate: float         # NHPP lambda at reference price, no promo, day 0
    trend_pct_yr: float      # annual multiplicative trend
    weekly_amp: float        # weekend lift amplitude
    annual_amp: float        # annual seasonality amplitude
    annual_phase: float      # radians
    promo_lift: float        # multiplicative demand lift on promo days
    promo_discount: float    # fractional price cut on promo days
    s_reorder: int           # (s, S) reorder point
    S_upto: int              # (s, S) order-up-to level
    lead_time_days: int


@dataclass(frozen=True)
class PanelConfig:
    """Portfolio-level configuration for the Phase 2 dataset."""

    n_products: int = 20
    n_stores: int = 5
    start_date: str = "2024-07-01"
    n_days: int = 731            # 2 years incl. leap day
    lead_time_days: int = 3
    promos_per_year: float = 6.0
    promo_len_days: tuple[int, int] = (5, 11)   # half-open randint bounds
    seed: int = 42


def sample_series_params(cfg: PanelConfig,
                         rng: np.random.Generator) -> list[SeriesParams]:
    """Draw heterogeneous parameters for every product-store series.

    (s, S) is sized off *peak* demand (trend + seasonality) so stockouts
    occur — concentrated around promos — but stay in a realistic 5-12%%
    of days range.
    """
    out = []
    store_scale = {f"S{k+1:02d}": rng.uniform(0.6, 1.5)
                   for k in range(cfg.n_stores)}
    for i in range(cfg.n_products):
        product = f"SKU{i+1:03d}"
        category = CATEGORIES[i % len(CATEGORIES)]
        base_price = float(np.round(rng.uniform(2.0, 25.0), 2))
        elasticity = float(rng.uniform(-2.5, -0.8))
        sku_rate = rng.uniform(2.0, 18.0)
        trend = float(rng.uniform(-0.10, 0.20))
        weekly_amp = float(rng.uniform(0.05, 0.35))
        annual_amp = float(rng.uniform(0.05, 0.40))
        phase = float(rng.uniform(0, 2 * np.pi))
        promo_lift = float(rng.uniform(1.5, 3.5))
        promo_disc = float(rng.uniform(0.10, 0.30))
        for store, scale in store_scale.items():
            base_rate = sku_rate * scale
            peak_rate = base_rate * (1 + max(trend, 0)) * (1 + weekly_amp) \
                        * (1 + annual_amp)
            cover = rng.uniform(7.0, 14.0)
            S_upto = max(5, int(peak_rate * cover))
            s_reorder = max(2, int(peak_rate * (cfg.lead_time_days + 1)
                                   * rng.uniform(0.9, 1.3)))
            s_reorder = min(s_reorder, S_upto - 1)
            out.append(SeriesParams(
                product_id=product, store_id=store, category=category,
                base_price=base_price, elasticity=elasticity,
                base_rate=base_rate, trend_pct_yr=trend,
                weekly_amp=weekly_amp, annual_amp=annual_amp,
                annual_phase=phase, promo_lift=promo_lift,
                promo_discount=promo_disc, s_reorder=s_reorder,
                S_upto=S_upto, lead_time_days=cfg.lead_time_days))
    return out


def sample_promo_mask(cfg: PanelConfig,
                      rng: np.random.Generator) -> np.ndarray:
    """Boolean promo mask: ~promos_per_year windows of 5-10 days each."""
    mask = np.zeros(cfg.n_days, dtype=bool)
    lo, hi = cfg.promo_len_days
    n_windows = rng.poisson(cfg.promos_per_year * cfg.n_days / 365.25)
    for _ in range(n_windows):
        start = int(rng.integers(0, cfg.n_days - (hi - 1)))
        length = int(rng.integers(lo, hi))
        mask[start:start + length] = True
    return mask


def observable_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop oracle columns — the view a forecast model is allowed to see."""
    return panel.drop(columns=[c for c in ORACLE_COLS if c in panel.columns])


def generate_panel(cfg: PanelConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate the Phase 2 forecasting dataset.

    Returns dict with:
      panel_oracle    — fact_sales_daily, full view incl. ground truth
      panel           — observable view (estimator/forecaster-safe)
      dim_product_store — static params per series
      promo_calendar  — promo windows (known-future covariate table)
    """
    cfg = cfg or PanelConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    series = sample_series_params(cfg, rng)

    rows, promo_rows = [], []

    for p in series:
        on_promo = sample_promo_mask(cfg, rng)
        price = np.where(on_promo,
                         np.round(p.base_price * (1 - p.promo_discount), 2),
                         p.base_price)

        lam = build_intensity(
            dates, p.base_rate, p.trend_pct_yr, p.weekly_amp, p.annual_amp,
            p.annual_phase, price, p.base_price, p.elasticity,
            on_promo, p.promo_lift)
        demanded = sample_daily_demand(lam, rng)

        inv_cfg = InventoryConfig(
            enabled=True, initial_stock=p.S_upto,
            reorder_point=p.s_reorder, order_up_to=p.S_upto,
            lead_time_days=p.lead_time_days)
        inv = simulate_inventory(demanded, inv_cfg, rng)

        rows.append(pd.DataFrame({
            "date": dates,
            "product_id": p.product_id,
            "store_id": p.store_id,
            "dow": dates.dayofweek,
            "week_of_year": dates.isocalendar().week.to_numpy(),
            "avg_price": price,
            "promo_flag": on_promo,
            "promo_depth": np.where(on_promo, p.promo_discount, 0.0).round(3),
            "inventory_start": inv.inventory_start,
            "replenishment": inv.replenishment,
            "units_sold": inv.units_sold,
            "sold_out": inv.sold_out,
            # --- oracle ---
            "units_demanded": demanded,
            "lost_sales": demanded - inv.units_sold,
            "stockout": inv.stockout,
            "lambda_true": np.round(lam, 4),
        }))

        d = np.diff(on_promo.astype(int), prepend=0, append=0)
        for st, en in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            promo_rows.append({
                "product_id": p.product_id, "store_id": p.store_id,
                "start_date": dates[st].date(),
                "end_date": dates[en - 1].date(),
                "discount": round(p.promo_discount, 3),
            })

    panel_oracle = pd.concat(rows, ignore_index=True)
    dim = pd.DataFrame([asdict(p) for p in series]).round(3)
    promo_calendar = pd.DataFrame(promo_rows)

    return {
        "panel_oracle": panel_oracle,
        "panel": observable_view(panel_oracle),
        "dim_product_store": dim,
        "promo_calendar": promo_calendar,
    }
