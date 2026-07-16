"""Phase 3: experiment module — assignment engines and design-aware estimation.

Implements docs/04 §1 (randomization units) and the fact_experiment table
(docs/00_spec.md §2.2):

  user               — each consumer assigned once (ZipRecruiter style);
                       analyze on user-level means (clustered by design)
  session            — each offer draws an arm independently (Uber-adjacent);
                       maximum power, vulnerable to dual exposure
  switchback_window  — whole market toggles arms on a randomized schedule
                       (Lyft/DoorDash); the unit is the window
  market             — geo/cluster randomization; the unit is the market

The estimand is the conversion contrast between two price arms — that
difference IS the demand-curve slope. `estimate_contrast` aggregates to the
randomization unit before computing the difference, which is what makes the
designs genuinely different: same sessions, very different effective n.

Interference between arms (shared capacity) is Phase 4; here switchback and
geo designs pay their variance cost without yet earning their bias advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demand import demand_curve, purchase_prob
from .population import build_population

RandUnit = Literal["user", "session", "switchback_window", "market"]

ORACLE_COLS = ["segment_idx", "segment_id", "wtp_true"]


@dataclass(frozen=True)
class ExperimentConfig:
    """One two-or-more-arm pricing experiment (docs/04)."""

    experiment_id: str = "exp_001"
    randomization_unit: RandUnit = "user"
    arm_prices: tuple[float, ...] = (29.0, 34.8)   # arm 0 = control/anchor
    allocation: tuple[float, ...] | None = None    # None = uniform
    horizon_days: int = 28
    sessions_per_day: float = 4_000.0
    window_hours: int = 2                          # switchback schedule
    burn_in_hours: float = 0.0                     # trim after each toggle
    n_markets: int = 20                            # geo design
    seed: int = 0

    def weights(self) -> np.ndarray:
        w = (np.ones(len(self.arm_prices)) if self.allocation is None
             else np.asarray(self.allocation, dtype=float))
        return w / w.sum()


def _arm_labels(exp: ExperimentConfig) -> np.ndarray:
    return np.array([f"arm_{i}" for i in range(len(exp.arm_prices))])


def run_experiment(cfg: SimulationConfig,
                   exp: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Simulate one experiment.

    Returns:
      sessions         — observable view (estimator-safe)
      sessions_oracle  — adds segment / wtp ground truth
      fact_experiment  — one row per randomized unit (spec §2.2)
    """
    rng = np.random.default_rng(exp.seed)
    n_sessions = int(exp.horizon_days * exp.sessions_per_day)
    labels = _arm_labels(exp)
    w = exp.weights()

    # --- population & sessions ------------------------------------------
    n_consumers = max(1, int(round(n_sessions / cfg.sessions_per_consumer_mean)))
    pop = build_population(cfg, n_consumers, rng)
    # observable segment proxy (spec §3.5, Phase 3): noisy log-WTP signal
    proxy = np.log(pop["wtp_true"].to_numpy()) + rng.normal(0, 0.8, n_consumers)
    market_id = rng.integers(0, exp.n_markets, size=n_consumers)

    consumer_rows = rng.integers(0, n_consumers, size=n_sessions)
    ts_hours = np.sort(rng.uniform(0, exp.horizon_days * 24, size=n_sessions))

    # --- assignment ------------------------------------------------------
    unit = exp.randomization_unit
    n_windows = int(np.ceil(exp.horizon_days * 24 / exp.window_hours))

    if unit == "user":
        arm_of_consumer = rng.choice(len(labels), size=n_consumers, p=w)
        arm_idx = arm_of_consumer[consumer_rows]
        unit_ids = pop["consumer_id"].to_numpy()
        unit_arms = arm_of_consumer
    elif unit == "session":
        arm_idx = rng.choice(len(labels), size=n_sessions, p=w)
        unit_ids = np.arange(n_sessions)
        unit_arms = arm_idx
    elif unit == "switchback_window":
        arm_of_window = rng.choice(len(labels), size=n_windows, p=w)
        window_idx = np.minimum((ts_hours / exp.window_hours).astype(int),
                                n_windows - 1)
        arm_idx = arm_of_window[window_idx]
        unit_ids = np.arange(n_windows)
        unit_arms = arm_of_window
    elif unit == "market":
        arm_of_market = rng.choice(len(labels), size=exp.n_markets, p=w)
        arm_idx = arm_of_market[market_id[consumer_rows]]
        unit_ids = np.arange(exp.n_markets)
        unit_arms = arm_of_market
    else:  # pragma: no cover
        raise ValueError(f"unknown randomization unit: {unit}")

    prices = np.asarray(exp.arm_prices, dtype=float)[arm_idx]

    # --- outcomes via the Phase 1 ground-truth demand model ---------------
    seg_idx = pop["segment_idx"].to_numpy()[consumer_rows]
    prob = np.empty(n_sessions)
    for i, seg in enumerate(cfg.segments):
        mask = seg_idx == i
        if mask.any():
            prob[mask] = purchase_prob(seg, prices[mask], cfg.choice_model)
    purchased = rng.random(n_sessions) < prob

    oracle = pd.DataFrame({
        "session_id": np.arange(n_sessions),
        "ts_hours": ts_hours,
        "consumer_id": pop["consumer_id"].to_numpy()[consumer_rows],
        "market_id": market_id[consumer_rows],
        "segment_proxy": proxy[consumer_rows],
        "quoted_price": prices,
        "arm": labels[arm_idx],
        "experiment_id": exp.experiment_id,
        "randomization_unit": unit,
        "purchased": purchased,
        # oracle
        "segment_idx": seg_idx,
        "segment_id": pop["segment_id"].to_numpy()[consumer_rows],
        "wtp_true": pop["wtp_true"].to_numpy()[consumer_rows],
    })

    # switchback burn-in: drop sessions within burn_in_hours of a toggle
    if unit == "switchback_window" and exp.burn_in_hours > 0:
        into_window = ts_hours % exp.window_hours
        oracle = oracle[into_window >= exp.burn_in_hours].reset_index(drop=True)

    fact = pd.DataFrame({
        "experiment_id": exp.experiment_id,
        "randomization_unit": unit,
        "unit_id": unit_ids,
        "arm": labels[unit_arms],
        "price_cell": np.asarray(exp.arm_prices, dtype=float)[unit_arms],
        "assigned_at": 0.0,
    })
    if unit == "switchback_window":
        fact["window_start"] = fact["unit_id"] * exp.window_hours
        fact["window_end"] = (fact["unit_id"] + 1) * exp.window_hours

    sessions = oracle.drop(columns=ORACLE_COLS)
    return {"sessions": sessions, "sessions_oracle": oracle,
            "fact_experiment": fact}


# --------------------------------------------------------------- estimation
_UNIT_KEY = {"user": "consumer_id", "session": "session_id",
             "switchback_window": None, "market": "market_id"}


def estimate_contrast(sessions: pd.DataFrame, exp: ExperimentConfig,
                      arm_a: str = "arm_0",
                      arm_b: str = "arm_1") -> pd.DataFrame:
    """Conversion contrast arm_b − arm_a with design-appropriate inference.

    Aggregates purchases to the randomization unit, then applies a
    two-sample normal contrast on unit-level means (docs/04 §1: 'the unit
    of the experiment is the unit of analysis'). For session-split this
    reduces to the classic two-proportion test.
    """
    df = sessions[sessions["arm"].isin([arm_a, arm_b])].copy()
    unit = exp.randomization_unit
    if unit == "switchback_window":
        df["_unit"] = np.minimum((df["ts_hours"] / exp.window_hours).astype(int),
                                 int(np.ceil(exp.horizon_days * 24
                                             / exp.window_hours)) - 1)
    else:
        df["_unit"] = df[_UNIT_KEY[unit]]

    g = df.groupby(["arm", "_unit"])["purchased"].mean()
    stats = {}
    for arm in (arm_a, arm_b):
        u = g.loc[arm]
        stats[arm] = (u.mean(), u.var(ddof=1) / len(u), len(u))

    (ma, va, na), (mb, vb, nb) = stats[arm_a], stats[arm_b]
    diff = mb - ma
    se = float(np.sqrt(va + vb))
    return pd.DataFrame([{
        "arm_a": arm_a, "arm_b": arm_b,
        "conv_a": ma, "conv_b": mb,
        "diff": diff, "se": se,
        "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se,
        "z": diff / se if se > 0 else np.nan,
        "n_units_a": na, "n_units_b": nb,
        "n_sessions": len(df),
    }])


def true_contrast(cfg: SimulationConfig, exp: ExperimentConfig,
                  arm_a: int = 0, arm_b: int = 1) -> float:
    """Oracle conversion difference between two arms' prices."""
    grid = np.array([exp.arm_prices[arm_a], exp.arm_prices[arm_b]], dtype=float)
    p = demand_curve(cfg, grid)["P_aggregate"].to_numpy()
    return float(p[1] - p[0])
