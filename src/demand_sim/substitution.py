"""Phase 4: substitution — cross-price effects and stockout spill (spec §3.3, §2.1).

Two coupled product-store series (the `substitutes` link in dim_product):

  1. Cross-price demand: product i's intensity scales with its substitute's
     price, (p_j / p0_j) ** cross_elasticity, cross_elasticity > 0 — when
     the rival gets expensive, my demand rises.
  2. Stockout spill: when the substitute is out of stock, a fraction
     `spill_rate` of its unmet demand arrives at my shelf the same day.

Both mechanisms bias single-product elasticity estimates and inflate the
observed demand of whatever happens to be in stock — the reason censoring
plus substitution is harder than censoring alone (docs/02).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .arrivals import build_intensity
from .config import InventoryConfig
from .panel import SeriesParams


@dataclass(frozen=True)
class SubstitutionConfig:
    cross_elasticity: float = 0.8   # >0: rival price up => my demand up
    spill_rate: float = 0.5         # share of rival's unmet demand that spills
    seed: int = 0


def simulate_substitute_pair(a: SeriesParams, b: SeriesParams,
                             dates: pd.DatetimeIndex,
                             price_a: np.ndarray, price_b: np.ndarray,
                             promo_a: np.ndarray, promo_b: np.ndarray,
                             sub: SubstitutionConfig) -> pd.DataFrame:
    """Coupled day loop for a substitute pair with (s, S) inventories.

    Demand realizes independently per product from its own NHPP intensity
    (scaled by the rival's price), then unmet demand spills to the rival
    within the same day. Returns a long panel with product_id in {a, b}
    and the usual observable/oracle columns plus `spill_in` (oracle).
    """
    rng = np.random.default_rng(sub.seed)
    n = len(dates)

    lam = {}
    for me, rival, p_me, p_rival, promo_me in (
            (a, b, price_a, price_b, promo_a),
            (b, a, price_b, price_a, promo_b)):
        base = build_intensity(dates, me.base_rate, me.trend_pct_yr,
                               me.weekly_amp, me.annual_amp, me.annual_phase,
                               p_me, me.base_price, me.elasticity,
                               promo_me, me.promo_lift)
        cross = (p_rival / rival.base_price) ** sub.cross_elasticity
        lam[me.product_id] = base * cross

    own = {p.product_id: rng.poisson(lam[p.product_id]) for p in (a, b)}

    # sequential (s,S) state per product
    state = {}
    for p in (a, b):
        state[p.product_id] = {
            "on_hand": p.S_upto, "pipeline": {}, "params": p,
            "sold": np.zeros(n, int), "inv_start": np.zeros(n, int),
            "receipts": np.zeros(n, int), "stockout": np.zeros(n, bool),
            "spill_in": np.zeros(n, int), "demand_total": np.zeros(n, int),
        }

    ids = [a.product_id, b.product_id]
    for day in range(n):
        # receipts
        for pid in ids:
            st = state[pid]
            qty = st["pipeline"].pop(day, 0)
            st["on_hand"] += qty
            st["receipts"][day] = qty
            st["inv_start"][day] = st["on_hand"]

        # first pass: own demand, record unmet
        unmet = {}
        for pid in ids:
            st = state[pid]
            want = int(own[pid][day])
            take = min(want, st["on_hand"])
            st["sold"][day] = take
            st["on_hand"] -= take
            unmet[pid] = want - take
            st["demand_total"][day] = want

        # second pass: spill unmet demand to the rival (same day)
        for pid, rival_id in ((ids[0], ids[1]), (ids[1], ids[0])):
            spill = int(np.floor(sub.spill_rate * unmet[rival_id]))
            if spill > 0:
                st = state[pid]
                extra = min(spill, st["on_hand"])
                st["sold"][day] += extra
                st["on_hand"] -= extra
                st["spill_in"][day] = spill
                st["demand_total"][day] += spill

        # reorder on position
        for pid in ids:
            st = state[pid]
            p = st["params"]
            position = st["on_hand"] + sum(st["pipeline"].values())
            if position <= p.s_reorder:
                arrival = day + p.lead_time_days + int(rng.integers(0, 2))
                st["pipeline"][arrival] = st["pipeline"].get(arrival, 0) \
                    + (p.S_upto - position)
            st["stockout"][day] = st["sold"][day] < st["demand_total"][day]

    frames = []
    prices = {a.product_id: price_a, b.product_id: price_b}
    promos = {a.product_id: promo_a, b.product_id: promo_b}
    for pid in ids:
        st = state[pid]
        frames.append(pd.DataFrame({
            "date": dates, "product_id": pid,
            "store_id": st["params"].store_id,
            "avg_price": prices[pid], "promo_flag": promos[pid],
            "inventory_start": st["inv_start"],
            "replenishment": st["receipts"],
            "units_sold": st["sold"],
            # oracle
            "units_demanded": st["demand_total"],
            "own_demand": own[pid],
            "spill_in": st["spill_in"],
            "stockout": st["stockout"],
        }))
    return pd.concat(frames, ignore_index=True)
