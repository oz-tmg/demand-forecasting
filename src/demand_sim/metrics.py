"""Estimator layer + scoring against ground truth.

Phase 1 estimators:
  * cell conversion rates (the raw demand curve, ZipRecruiter Figure-1 style)
  * arc elasticity between adjacent cells
  * pooled binary logit MLE
Each is scored against oracle truth from demand.true_elasticity / demand_curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demand import demand_curve, true_elasticity


# ------------------------------------------------------------------ estimators
def cell_conversion(sessions: pd.DataFrame) -> pd.DataFrame:
    """Conversion rate and CI per price cell — the empirical demand curve."""
    g = (sessions.groupby("quoted_price", as_index=False)
                 .agg(n=("purchased", "size"), conv=("purchased", "mean")))
    se = np.sqrt(g["conv"] * (1 - g["conv"]) / g["n"])
    g["ci_lo"], g["ci_hi"] = g["conv"] - 1.96 * se, g["conv"] + 1.96 * se
    return g.sort_values("quoted_price").reset_index(drop=True)


def arc_elasticity(cells: pd.DataFrame) -> pd.DataFrame:
    """Midpoint arc elasticity between adjacent price cells."""
    p, q = cells["quoted_price"].to_numpy(), cells["conv"].to_numpy()
    dq = np.diff(q) / ((q[1:] + q[:-1]) / 2)
    dp = np.diff(p) / ((p[1:] + p[:-1]) / 2)
    return pd.DataFrame({
        "price_mid": (p[1:] + p[:-1]) / 2,
        "arc_elasticity": dq / dp,
    })


def fit_pooled_logit(sessions: pd.DataFrame):
    """Pooled purchase logit: P(buy) = sigma(a + b * price). Returns statsmodels fit."""
    import statsmodels.api as sm
    X = sm.add_constant(sessions["quoted_price"].to_numpy())
    return sm.Logit(sessions["purchased"].astype(int), X).fit(disp=0)


# --------------------------------------------------------------------- scoring
def score_run(cfg: SimulationConfig, sessions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Grade Phase 1 estimators against oracle truth."""
    cells = cell_conversion(sessions)
    grid = cells["quoted_price"].to_numpy(dtype=float)

    # 1) Demand-curve recovery: empirical conversion vs true P(p), CI coverage.
    truth_curve = demand_curve(cfg, grid).rename(columns={"price": "quoted_price"})
    curve = cells.merge(truth_curve[["quoted_price", "P_aggregate"]], on="quoted_price")
    curve["abs_error"] = (curve["conv"] - curve["P_aggregate"]).abs()
    curve["covered"] = (curve["ci_lo"] <= curve["P_aggregate"]) & \
                       (curve["P_aggregate"] <= curve["ci_hi"])

    # 2) Elasticity recovery at arc midpoints.
    arcs = arc_elasticity(cells)
    truth_e = true_elasticity(cfg, arcs["price_mid"].to_numpy())
    arcs["true_elasticity"] = truth_e["elasticity_aggregate"].to_numpy()
    arcs["bias"] = arcs["arc_elasticity"] - arcs["true_elasticity"]

    # 3) Pooled logit price coefficient vs. share-weighted truth (aggregation bias
    #    is EXPECTED under mixtures — the report surfaces it rather than hiding it).
    fit = fit_pooled_logit(sessions)
    b_hat = float(np.asarray(fit.params)[1])
    b_mix = float(np.average([s.price_coef for s in cfg.segments],
                             weights=[s.share for s in cfg.segments]))
    logit_report = pd.DataFrame([{
        "b_hat": b_hat,
        "b_share_weighted_truth": b_mix,
        "pct_gap_vs_weighted": 100 * (b_hat - b_mix) / abs(b_mix),
        "n_sessions": len(sessions),
    }])

    return {"demand_curve": curve, "elasticity": arcs, "pooled_logit": logit_report}


# ------------------------------------------------------------------ power tool
def power_two_proportions(p1: float, p2: float, alpha: float = 0.05,
                          power: float = 0.8) -> float:
    """n per arm to detect conversion p1 vs p2 (docs/04 §2)."""
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    es = proportion_effectsize(p1, p2)
    return float(NormalIndPower().solve_power(es, alpha=alpha, power=power, ratio=1))
