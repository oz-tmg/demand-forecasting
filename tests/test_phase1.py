"""Phase 1 acceptance tests (docs/00_spec.md section 5)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from demand_sim import (default_config, run_simulation, score_run,
                        demand_curve, true_elasticity)
from demand_sim.config import SegmentConfig, SimulationConfig


def test_demand_downward_sloping():
    cfg = default_config()
    grid = np.linspace(5, 150, 60)
    d = demand_curve(cfg, grid)["P_aggregate"].to_numpy()
    assert np.all(np.diff(d) < 0)


def test_elasticity_negative():
    cfg = default_config()
    e = true_elasticity(cfg, np.linspace(5, 150, 30))["elasticity_aggregate"]
    assert (e < 0).all()


def test_curve_recovery_coverage():
    cfg = default_config()
    reports = score_run(cfg, run_simulation(cfg)["sessions"])
    assert reports["demand_curve"]["covered"].mean() >= 0.85  # ~95% nominal


def test_single_segment_logit_recovery():
    cfg = SimulationConfig(
        segments=(SegmentConfig("only", share=1.0, base_util=1.0, price_coef=-0.05),),
        n_sessions=250_000, seed=7,
    )
    rep = score_run(cfg, run_simulation(cfg)["sessions"])["pooled_logit"]
    assert abs(rep["pct_gap_vs_weighted"].iloc[0]) < 5.0


def test_deterministic_under_seed():
    cfg = default_config()
    a = run_simulation(cfg)["sessions"]["purchased"].sum()
    b = run_simulation(cfg)["sessions"]["purchased"].sum()
    assert a == b
