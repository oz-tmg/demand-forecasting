"""Consumer population: segment assignment and per-consumer draws."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig


def build_population(cfg: SimulationConfig, n_consumers: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Create the consumer pool.

    Returns a DataFrame with one row per consumer:
      consumer_id, segment_id (oracle), and — for the wtp_threshold model —
      a persistent WTP draw (oracle). Persistent draws mean a consumer's
      repeated sessions are correlated, which is what makes per-user vs.
      per-session randomization genuinely different designs.
    """
    shares = np.array([s.share for s in cfg.segments])
    seg_idx = rng.choice(len(cfg.segments), size=n_consumers, p=shares)

    wtp = np.empty(n_consumers)
    for i, seg in enumerate(cfg.segments):
        mask = seg_idx == i
        wtp[mask] = rng.lognormal(mean=seg.wtp_mu, sigma=seg.wtp_sigma, size=mask.sum())

    return pd.DataFrame({
        "consumer_id": np.arange(n_consumers),
        "segment_idx": seg_idx,
        "segment_id": np.array([cfg.segments[i].name for i in seg_idx]),
        "wtp_true": wtp,
    })
