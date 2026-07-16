"""Phase 4: interference — shared-capacity coupling between arms (spec §3.4).

The marketplace SUTVA problem: arms share a finite resource (driver supply,
inventory, ad slots). A cheap arm's extra purchases deplete the pool that
expensive-arm sessions draw from, so the control group's outcome depends on
the treatment group's assignment — and user/session-split contrasts are
biased. Switchback and geo designs keep each capacity pool in one arm at a
time, which is the entire reason they exist (docs/04 §1.3-1.4).

Mechanics: sessions want to purchase per the Phase 1 demand model; each
(market, capacity-window) pool can fulfil at most `capacity` purchases.
Excess desired purchases are rationed at random within the pool. The
`capacity_slack` dial sets pool size relative to expected desired purchases
at the anchor price — slack >= ~1.5 approximates no interference; slack < 1
is strongly coupled.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .demand import demand_curve
from .experiments import ExperimentConfig, estimate_contrast, run_experiment


@dataclass(frozen=True)
class InterferenceConfig:
    capacity_slack: float = 0.8     # pool capacity / expected desired purchases
    capacity_window_hours: int = 2  # how often supply replenishes
    seed: int = 0


def pool_capacity(cfg: SimulationConfig, exp: ExperimentConfig,
                  icfg: InterferenceConfig) -> int:
    """Fixed physical capacity per (market, window) pool.

    Sized as slack x expected desired purchases per pool at the ANCHOR
    price. This is a property of the market, so it must be computed once
    from the reference experiment and held fixed across counterfactual
    worlds — recomputing it per world would let the resource move with the
    treatment, which is exactly what interference forbids.
    """
    anchor_conv = float(demand_curve(
        cfg, np.array([exp.arm_prices[0]]))["P_aggregate"].iloc[0])
    n_sessions = int(exp.horizon_days * exp.sessions_per_day)
    n_pools = exp.n_markets * int(np.ceil(exp.horizon_days * 24
                                          / icfg.capacity_window_hours))
    return max(1, int(round(icfg.capacity_slack * anchor_conv
                            * n_sessions / n_pools)))


def apply_capacity(sessions: pd.DataFrame, cfg: SimulationConfig,
                   exp: ExperimentConfig, icfg: InterferenceConfig,
                   capacity: int | None = None) -> pd.DataFrame:
    """Ration desired purchases within (market, window) capacity pools.

    Returns sessions with `purchased` replaced by the post-rationing outcome
    and oracle columns `desired` and `rationed` added. Pass `capacity`
    explicitly when comparing counterfactual worlds that must share the
    same physical resource.
    """
    rng = np.random.default_rng(icfg.seed + 77)
    df = sessions.copy()
    df["desired"] = df["purchased"]
    df["_pool_window"] = (df["ts_hours"] // icfg.capacity_window_hours).astype(int)

    if capacity is None:
        capacity = pool_capacity(cfg, exp, icfg)

    purchased = df["desired"].to_numpy().copy()
    rationed = np.zeros(len(df), dtype=bool)
    for _, idx in df[df["desired"]].groupby(
            ["market_id", "_pool_window"]).groups.items():
        idx = np.asarray(idx)
        if len(idx) > capacity:
            drop = rng.choice(idx, size=len(idx) - capacity, replace=False)
            purchased[drop] = False
            rationed[drop] = True

    df["purchased"] = purchased
    df["rationed"] = rationed
    return df.drop(columns="_pool_window")


def run_interference_experiment(cfg: SimulationConfig, exp: ExperimentConfig,
                                icfg: InterferenceConfig,
                                capacity: int | None = None) -> dict:
    """run_experiment + shared-capacity rationing."""
    out = run_experiment(cfg, exp)
    sessions = apply_capacity(out["sessions_oracle"], cfg, exp, icfg, capacity)
    return {**out, "sessions": sessions}


def interference_design_bias(cfg: SimulationConfig, base: ExperimentConfig,
                             icfg: InterferenceConfig,
                             designs: tuple[str, ...] = ("session", "user",
                                                         "switchback_window",
                                                         "market"),
                             n_reps: int = 30) -> pd.DataFrame:
    """The Phase 4 payoff table: same coupled market, four designs.

    The benchmark ("truth under capacity") is the GLOBAL counterfactual
    contrast: everyone at arm price 0 vs everyone at arm price 1, each with
    capacity applied — what a switchback measures in expectation. Mixed-arm
    designs (session/user) shift capacity between arms within pools and
    should show bias; switchback and geo should not.
    """
    # one physical capacity, shared by every world and every design
    capacity = pool_capacity(cfg, base, icfg)

    # global-rollout benchmark under the same capacity mechanics
    def global_mean(arm: int, rep: int) -> float:
        e = replace(base, randomization_unit="session",
                    arm_prices=(base.arm_prices[arm], base.arm_prices[arm]),
                    seed=base.seed + 9000 + rep)
        s = run_interference_experiment(cfg, e, replace(icfg, seed=rep),
                                        capacity=capacity)
        return float(s["sessions"]["purchased"].mean())

    truth = float(np.mean([global_mean(1, r) - global_mean(0, r)
                           for r in range(max(10, n_reps // 3))]))

    rows = []
    for unit in designs:
        diffs = []
        for r in range(n_reps):
            e = replace(base, randomization_unit=unit,  # type: ignore[arg-type]
                        seed=base.seed + 3000 + r)
            out = run_interference_experiment(cfg, e, replace(icfg, seed=r),
                                              capacity=capacity)
            est = estimate_contrast(out["sessions"], e)
            diffs.append(float(est["diff"].iloc[0]))
        d = np.array(diffs)
        rows.append({
            "randomization_unit": unit,
            "true_diff_under_capacity": truth,
            "mean_estimate": float(d.mean()),
            "bias": float(d.mean() - truth),
            "rmse": float(np.sqrt(((d - truth) ** 2).mean())),
            "n_reps": n_reps,
        })
    return pd.DataFrame(rows)
