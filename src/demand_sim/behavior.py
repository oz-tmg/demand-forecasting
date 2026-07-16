"""Phase 4: behavioral demand — reference prices and demand shifting.

Reference prices (spec §3.1): consumers anchor on a reference price and
punish increases above it more than they reward equal cuts below it (loss
aversion). Activates the `SegmentConfig.ref_price_sensitivity` hook that has
ridden along since Phase 1.

Demand shifting (spec §3.2): under per-session price randomization, consumers
who see a high price can DEFER and return — so cheap-cell purchases become
enriched with patient bargain-hunters who previously saw high prices. This is
the Uber "surge waiting" limitation and the contamination mechanism docs/04
§1.2 warns about; `guardrails.strategic_waiting_check` detects its signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SegmentConfig, SimulationConfig
from .population import build_population


# ------------------------------------------------------------ reference prices
def purchase_prob_ref(seg: SegmentConfig, price: np.ndarray | float,
                      ref_price: float, model: str = "logit") -> np.ndarray:
    """P(buy) with loss aversion around a reference price (logit model).

    Utility gains an extra penalty only ABOVE the anchor:

        U = base_util + price_coef*p − ref_price_sensitivity*max(0, p − p_ref)

    With sensitivity 0 this reduces exactly to the Phase 1 logit. The result
    is a kinked demand curve: elasticity is larger just above p_ref than just
    below it — the classic reason firms defend price points.
    """
    if model != "logit":
        raise NotImplementedError("reference prices implemented for logit only")
    p = np.asarray(price, dtype=float)
    loss = np.maximum(0.0, p - ref_price)
    util = seg.base_util + seg.price_coef * p - seg.ref_price_sensitivity * loss
    return 1.0 / (1.0 + np.exp(-util))


def demand_curve_ref(cfg: SimulationConfig, price_grid: np.ndarray,
                     ref_price: float) -> pd.DataFrame:
    """Aggregate demand curve under reference-price loss aversion."""
    agg = np.zeros_like(price_grid, dtype=float)
    for seg in cfg.segments:
        agg += seg.share * purchase_prob_ref(seg, price_grid, ref_price)
    return pd.DataFrame({"price": price_grid, "P_aggregate": agg})


# ------------------------------------------------------------- demand shifting
@dataclass(frozen=True)
class WaitingConfig:
    """Strategic-waiting knob (docs/04 §3.4, Phase 4)."""

    wait_prob: float = 0.5          # P(defer | would have bought, price > anchor)
    anchor_price: float = 29.0      # prices above this trigger waiting
    organic_revisit_prob: float = 0.15  # price-independent return traffic (null world)
    max_revisits: int = 3
    n_consumers: int = 20_000
    price_cells: tuple[float, ...] = (19.0, 29.0, 49.0)
    seed: int = 0


def simulate_with_waiting(cfg: SimulationConfig,
                          wcfg: WaitingConfig) -> pd.DataFrame:
    """Per-session randomization with strategic waiting.

    Each consumer draws a price cell per visit. Would-be buyers who see a
    price above the anchor defer with probability wait_prob and revisit; on
    top of that, any non-buyer may return organically (price-independent),
    so the wait_prob=0 world still has revisit traffic — the honest null.

    The refresh-arbitrage signature: with wait_prob>0, revisit sessions'
    PREVIOUS prices skew above the anchor (people come back because it was
    expensive), which `guardrails.strategic_waiting_check` tests. Emits
    `prev_price` (NaN on first visits) and `prior_sessions`.
    """
    rng = np.random.default_rng(wcfg.seed)
    pop = build_population(cfg, wcfg.n_consumers, rng)
    seg_idx = pop["segment_idx"].to_numpy()
    cells = np.asarray(wcfg.price_cells, dtype=float)

    from .demand import purchase_prob
    frames = []
    active = np.arange(wcfg.n_consumers)          # consumers still shopping
    prev_price = np.full(wcfg.n_consumers, np.nan)
    for visit in range(wcfg.max_revisits + 1):
        n = len(active)
        if n == 0:
            break
        price = cells[rng.integers(0, len(cells), size=n)]
        prob = np.empty(n)
        seg_of_active = seg_idx[active]
        for i, seg in enumerate(cfg.segments):
            mask = seg_of_active == i
            if mask.any():
                prob[mask] = purchase_prob(seg, price[mask], cfg.choice_model)
        wants = rng.random(n) < prob
        can_return = visit < wcfg.max_revisits
        defers = (wants & (price > wcfg.anchor_price) & can_return
                  & (rng.random(n) < wcfg.wait_prob))
        purchased = wants & ~defers
        # organic returns are PRICE- and PURCHASE-independent (repeat
        # browsing); conditioning them on not buying would smuggle price
        # selection into the null world and break the detector's calibration
        organic = (~defers & can_return
                   & (rng.random(n) < wcfg.organic_revisit_prob))
        frames.append(pd.DataFrame({
            "consumer_id": active,
            "quoted_price": price,
            "prev_price": prev_price[active],
            "prior_sessions": visit,
            "purchased": purchased,
        }))
        returning = defers | organic
        prev_price[active[returning]] = price[returning]
        active = active[returning]

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "session_id", np.arange(len(out)))
    return out
