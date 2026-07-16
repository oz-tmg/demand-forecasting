"""Quickstart: simulate a randomized pricing experiment, recover the demand curve,
and grade the estimators against ground truth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from demand_sim import (default_config, run_simulation, score_run,
                        true_elasticity, consumer_surplus, power_two_proportions)

cfg = default_config()
out = run_simulation(cfg)
reports = score_run(cfg, out["sessions"])

print("=== Empirical demand curve vs truth ===")
print(reports["demand_curve"].round(4).to_string(index=False))
print("\n=== Arc elasticity vs truth ===")
print(reports["elasticity"].round(3).to_string(index=False))
print("\n=== Pooled logit recovery ===")
print(reports["pooled_logit"].round(4).to_string(index=False))

print("\n=== True aggregate elasticity along the curve ===")
grid = np.array(cfg.price_cells, dtype=float)
print(true_elasticity(cfg, grid)[["price", "elasticity_aggregate"]]
      .round(3).to_string(index=False))

print(f"\nConsumer surplus per session at p=$29: ${consumer_surplus(cfg, 29.0):.2f}")
print(f"n/arm to detect 10.0% vs 8.5% conversion: {power_two_proportions(0.10, 0.085):,.0f}")
