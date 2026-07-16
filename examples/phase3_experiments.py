"""Phase 3 demo: same market, same arms, four randomization designs.

Prints the experiment scorecard (docs/04 §5), guardrail outcomes, an A/A
battery, and a surge-RD run — everything graded against oracle truth.

Run:  python examples/phase3_experiments.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from demand_sim import (ExperimentConfig, aa_battery, default_config,
                        experiment_scorecard, rd_estimate, run_surge_sessions,
                        true_rd_jump)

cfg = default_config()
base = ExperimentConfig(arm_prices=(29.0, 34.8), horizon_days=14,
                        sessions_per_day=3_000, window_hours=2, n_markets=20)

print("=== Experiment scorecard: design x (bias, RMSE, power, coverage) ===")
card = experiment_scorecard(cfg, base, n_reps=30)
print(card.round(4).to_string(index=False))
print("\nNote the mechanics: session-split maximizes units but is ~25% "
      "dual-exposed;\nswitchback has only ~170 windows; geo has 20 markets. "
      "Interference (the\nreason to accept those costs) arrives in Phase 4.")

print("\n=== A/A battery (identical arms; rejection rate should be ~5%) ===")
print(aa_battery(cfg, base, n_reps=40))

print("\n=== Surge-RD at the 1.25 rounding cutpoint (Uber-style) ===")
sessions = run_surge_sessions(cfg, n_sessions=400_000, base_price=25.0, seed=3)
est = rd_estimate(sessions, cut=1.25).iloc[0]
truth = true_rd_jump(cfg, base_price=25.0, cut=1.25)
print(f"jump estimate: {est['jump']:+.4f}  (95% CI {est['ci_lo']:+.4f} "
      f"to {est['ci_hi']:+.4f})")
print(f"true jump:     {truth:+.4f}")
