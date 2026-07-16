"""Phase 2: (s, S) replenishment and inventory censoring (docs/02).

Censoring is first-class (docs/00_spec.md design principle 4): the simulator
always knows true demand; the observable view only shows sales.

    units_sold = min(units_demanded, on_hand)

Stockout behavior is "lost" (unmet demand vanishes). "substitute" and
"backorder" are Phase 4 (docs/00_spec.md §3.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import InventoryConfig


@dataclass(frozen=True)
class InventoryResult:
    """Per-day inventory trajectories for one product-store series."""

    units_sold: np.ndarray       # observable — censored at on-hand stock
    inventory_start: np.ndarray  # observable — on-hand after receipts
    replenishment: np.ndarray    # observable — units received that day
    stockout: np.ndarray         # oracle — units_sold < units_demanded
    sold_out: np.ndarray         # observable proxy — ended the day at zero stock


def simulate_inventory(units_demanded: np.ndarray,
                       cfg: InventoryConfig,
                       rng: np.random.Generator) -> InventoryResult:
    """Sequential (s, S) policy with fixed lead time (+0/1 day jitter).

    Reorders on inventory *position* (on hand + on order) so pipeline stock
    is not double-ordered. Starts fully stocked at S.
    """
    if cfg.stockout_behavior != "lost":
        raise NotImplementedError(
            f"stockout_behavior={cfg.stockout_behavior!r} is Phase 4")

    s, S, lead = cfg.reorder_point, cfg.order_up_to, cfg.lead_time_days
    n = len(units_demanded)
    on_hand = S
    pipeline: dict[int, int] = {}  # arrival_day -> qty

    sold = np.zeros(n, dtype=int)
    inv_start = np.zeros(n, dtype=int)
    receipts = np.zeros(n, dtype=int)
    stockout = np.zeros(n, dtype=bool)
    sold_out = np.zeros(n, dtype=bool)

    for day in range(n):
        qty = pipeline.pop(day, 0)
        on_hand += qty
        receipts[day] = qty
        inv_start[day] = on_hand

        take = min(int(units_demanded[day]), on_hand)
        sold[day] = take
        on_hand -= take
        stockout[day] = take < units_demanded[day]
        sold_out[day] = on_hand == 0

        position = on_hand + sum(pipeline.values())
        if position <= s:
            arrival = day + lead + int(rng.integers(0, 2))
            pipeline[arrival] = pipeline.get(arrival, 0) + (S - position)

    return InventoryResult(sold, inv_start, receipts, stockout, sold_out)
