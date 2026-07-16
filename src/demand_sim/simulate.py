"""Session generator: the Phase 1 simulation loop.

Produces fact_session with an observable view (what estimators may see) and an
oracle view (ground truth for scoring). See docs/00_spec.md §2.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demand import purchase_prob
from .population import build_population

ORACLE_COLS = ["segment_idx", "segment_id", "wtp_true", "purchase_prob_true"]


def run_simulation(cfg: SimulationConfig) -> dict[str, pd.DataFrame]:
    """Simulate sessions under cfg.

    Returns dict with:
      sessions          — observable view (estimator-safe)
      sessions_oracle   — full view including ground truth columns
      population        — consumer pool (oracle)
    """
    rng = np.random.default_rng(cfg.seed)

    # Consumers: enough that expected sessions/consumer matches config.
    n_consumers = max(1, int(round(cfg.n_sessions / cfg.sessions_per_consumer_mean)))
    pop = build_population(cfg, n_consumers, rng)

    # Sessions: sample consumers with replacement (mean sessions/consumer emerges).
    consumer_rows = rng.integers(0, n_consumers, size=cfg.n_sessions)
    seg_idx = pop["segment_idx"].to_numpy()[consumer_rows]

    from .pricing import assign_prices
    prices, cells = assign_prices(cfg, cfg.n_sessions, rng)
    promo = (rng.random(cfg.n_sessions) < cfg.promo_prob).astype(float)

    # Ground-truth purchase probability per session, then Bernoulli outcome.
    prob = np.empty(cfg.n_sessions)
    for i, seg in enumerate(cfg.segments):
        mask = seg_idx == i
        if mask.any():
            prob[mask] = purchase_prob(seg, prices[mask], cfg.choice_model, promo[mask])

    if cfg.choice_model == "wtp_threshold":
        # Persistent-WTP variant: outcome is deterministic given the consumer's draw
        # (correlated repeat sessions — matters for clustering / per-user designs).
        wtp = pop["wtp_true"].to_numpy()[consumer_rows]
        boost = np.exp(np.array([cfg.segments[i].promo_uplift for i in seg_idx]) * promo)
        purchased = (wtp * boost >= prices)
    else:
        purchased = rng.random(cfg.n_sessions) < prob

    oracle = pd.DataFrame({
        "session_id": np.arange(cfg.n_sessions),
        "ts": pd.RangeIndex(cfg.n_sessions),          # Phase 2: real NHPP timestamps
        "consumer_id": pop["consumer_id"].to_numpy()[consumer_rows],
        "segment_idx": seg_idx,
        "segment_id": pop["segment_id"].to_numpy()[consumer_rows],
        "product_id": "sku_001",                       # Phase 2: multi-product
        "quoted_price": prices,
        "price_mechanism": cfg.price_mechanism,
        "cell": cells,
        "promo_flag": promo.astype(bool),
        "in_stock": True,                              # Phase 2: inventory censoring
        "wtp_true": pop["wtp_true"].to_numpy()[consumer_rows],
        "purchase_prob_true": prob,
        "purchased": purchased,
    })

    sessions = oracle.drop(columns=ORACLE_COLS)
    return {"sessions": sessions, "sessions_oracle": oracle, "population": pop}
