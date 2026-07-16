"""Phase 4 demo: every way observational pricing data lies — with the fix.

1. Endogeneity bias curves: OLS degrades with gamma, IV/CF stay flat (docs/05 §8)
2. Interference: session/user-split overstate the rollout effect ~2x;
   switchback/geo recover it (docs/04 §1.3-1.4)
3. Strategic waiting: the refresh-arbitrage detector fires only when the
   behavior exists (docs/04 §3.4)
4. Reference prices: the demand curve kinks at the anchor

Run:  python examples/phase4_endogeneity.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from demand_sim import (EndogenousConfig, ExperimentConfig, InterferenceConfig,
                        WaitingConfig, default_config, demand_curve,
                        demand_curve_ref, endogeneity_scorecard,
                        interference_design_bias, simulate_with_waiting,
                        strategic_waiting_check)
from dataclasses import replace

cfg = default_config()

print("=== 1. Endogeneity bias curves (theta_true = -1.5) ===")
card = endogeneity_scorecard(EndogenousConfig(n_periods=8_000), n_reps=10)
pivot = card.pivot(index="gamma", columns="estimator", values="bias").round(3)
print(pivot.to_string())
f = card.loc[card.estimator == "2sls", "mean_first_stage_F"].mean()
print(f"(mean first-stage F: {f:,.0f} — instruments healthy)\n")

print("=== 2. Interference: same coupled market, four designs ===")
base = ExperimentConfig(arm_prices=(29.0, 39.0), horizon_days=7,
                        sessions_per_day=3_000, window_hours=2,
                        n_markets=10, seed=5)
tab = interference_design_bias(cfg, base,
                               InterferenceConfig(capacity_slack=0.85),
                               n_reps=12)
print(tab.round(4).to_string(index=False))
print("Session/user-split promise more than a rollout would deliver; "
      "switchback and geo\ndesigns finally earn their Phase 3 variance cost.\n")

print("=== 3. Strategic-waiting detector ===")
for wp in (0.5, 0.0):
    w = simulate_with_waiting(cfg, WaitingConfig(wait_prob=wp,
                                                 n_consumers=10_000, seed=2))
    c = strategic_waiting_check(w)
    print(f"wait_prob={wp}: alarm={c['waiting_alarm']} "
          f"(revisits' prev price {c['mean_prev_price_revisits']:.2f} "
          f"vs {c['mean_price_served']:.2f} served)")

print("\n=== 4. Reference-price kink at p_ref = $29 ===")
grid = np.array([25.0, 27.0, 29.0, 31.0, 33.0])
cfg_ref = replace(cfg, segments=tuple(
    replace(s, ref_price_sensitivity=0.08) for s in cfg.segments))
plain = demand_curve(cfg, grid)["P_aggregate"].to_numpy()
kinked = demand_curve_ref(cfg_ref, grid, ref_price=29.0)["P_aggregate"].to_numpy()
for p, a, b in zip(grid, plain, kinked):
    mark = "  <- loss aversion bites" if p > 29 else ""
    print(f"  p={p:5.2f}  no-anchor={a:.4f}  anchored={b:.4f}{mark}")
