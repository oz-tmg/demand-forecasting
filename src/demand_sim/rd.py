"""Phase 3: surge-pricing regression discontinuity (Uber, Cohen et al. 2016).

An algorithmic pricing rule computes a continuous latent surge multiplier and
*rounds* it to discrete levels (pricing.surge_price). Consumers just left and
right of a rounding cutpoint are statistically identical, but face different
prices — a regression discontinuity that identifies demand locally without a
randomized experiment.

The firm knows its own algorithm, so the latent multiplier (the running
variable) is observable; the oracle here is the true conversion jump implied
by the Phase 1 demand model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demand import demand_curve, purchase_prob
from .population import build_population
from .pricing import surge_price


def run_surge_sessions(cfg: SimulationConfig, n_sessions: int,
                       base_price: float, step: float = 0.1,
                       mult_mean: float = 1.25, mult_sd: float = 0.20,
                       seed: int = 0) -> pd.DataFrame:
    """Sessions priced by a surge-with-rounding rule (price_mechanism='surge').

    Returns observable columns: surge_latent (running variable),
    surge_multiplier (rounded), quoted_price, purchased.
    """
    rng = np.random.default_rng(seed)
    n_consumers = max(1, int(round(n_sessions / cfg.sessions_per_consumer_mean)))
    pop = build_population(cfg, n_consumers, rng)
    rows = rng.integers(0, n_consumers, size=n_sessions)
    seg_idx = pop["segment_idx"].to_numpy()[rows]

    latent = np.clip(rng.normal(mult_mean, mult_sd, size=n_sessions), 1.0, None)
    quoted, _running = surge_price(base_price, latent, step)

    prob = np.empty(n_sessions)
    for i, seg in enumerate(cfg.segments):
        mask = seg_idx == i
        if mask.any():
            prob[mask] = purchase_prob(seg, quoted[mask], cfg.choice_model)
    purchased = rng.random(n_sessions) < prob

    return pd.DataFrame({
        "session_id": np.arange(n_sessions),
        "consumer_id": pop["consumer_id"].to_numpy()[rows],
        "surge_latent": latent,
        "surge_multiplier": np.round(quoted / base_price, 10),
        "quoted_price": quoted,
        "price_mechanism": "surge",
        "purchased": purchased,
    })


def rd_estimate(sessions: pd.DataFrame, cut: float,
                bandwidth: float = 0.04) -> pd.DataFrame:
    """Local-linear RD at rounding cutpoint `cut` on the latent multiplier.

    Fits purchased ~ above + (latent-cut) + above*(latent-cut) within the
    bandwidth; the coefficient on `above` is the conversion jump at the
    price discontinuity. HC1 standard errors.
    """
    import statsmodels.api as sm
    x = sessions["surge_latent"].to_numpy() - cut
    keep = np.abs(x) <= bandwidth
    x, y = x[keep], sessions["purchased"].to_numpy()[keep].astype(float)
    above = (x >= 0).astype(float)
    X = sm.add_constant(np.column_stack([above, x, above * x]))
    fit = sm.OLS(y, X).fit(cov_type="HC1")
    jump, se = float(fit.params[1]), float(fit.bse[1])
    return pd.DataFrame([{
        "cut": cut, "bandwidth": bandwidth, "n_in_window": int(keep.sum()),
        "jump": jump, "se": se,
        "ci_lo": jump - 1.96 * se, "ci_hi": jump + 1.96 * se,
    }])


def true_rd_jump(cfg: SimulationConfig, base_price: float, cut: float,
                 step: float = 0.1) -> float:
    """Oracle conversion jump at cutpoint `cut`: rounding sends latent just
    below to level floor(cut/step)*step and just above to the next level."""
    lo = np.floor(cut / step) * step
    hi = lo + step
    grid = np.array([base_price * lo, base_price * hi])
    p = demand_curve(cfg, grid)["P_aggregate"].to_numpy()
    return float(p[1] - p[0])
