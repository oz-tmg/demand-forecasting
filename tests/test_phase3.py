"""Phase 3 acceptance tests: designs, guardrails, power, RD (docs/04)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace

import numpy as np
import pytest

from demand_sim import (ExperimentConfig, aa_battery, covariate_balance,
                        default_config, dual_exposure, estimate_contrast,
                        monte_carlo_power, power_two_proportions, rd_estimate,
                        run_experiment, run_surge_sessions, srm_check,
                        true_contrast, true_rd_jump)

CFG = default_config()
BASE = ExperimentConfig(arm_prices=(29.0, 34.8), horizon_days=14,
                        sessions_per_day=2_000, window_hours=2,
                        n_markets=20, seed=5)

DESIGNS = ["user", "session", "switchback_window", "market"]


@pytest.mark.parametrize("unit", DESIGNS)
def test_design_recovers_truth(unit):
    exp = replace(BASE, randomization_unit=unit)
    out = run_experiment(CFG, exp)
    est = estimate_contrast(out["sessions"], exp).iloc[0]
    truth = true_contrast(CFG, exp)
    assert abs(est["diff"] - truth) < 4 * est["se"]


def test_fact_experiment_schema():
    for unit in DESIGNS:
        exp = replace(BASE, randomization_unit=unit)
        fact = run_experiment(CFG, exp)["fact_experiment"]
        assert {"experiment_id", "randomization_unit", "unit_id", "arm",
                "price_cell"} <= set(fact.columns)
        assert (fact["randomization_unit"] == unit).all()
        # one row per unit, arms drawn from the configured cells
        assert fact["unit_id"].is_unique
        assert set(fact["price_cell"]) <= set(exp.arm_prices)


def test_unit_counts_match_design():
    out_sb = run_experiment(CFG, replace(BASE, randomization_unit="switchback_window"))
    n_windows = int(np.ceil(BASE.horizon_days * 24 / BASE.window_hours))
    assert len(out_sb["fact_experiment"]) == n_windows
    out_geo = run_experiment(CFG, replace(BASE, randomization_unit="market"))
    assert len(out_geo["fact_experiment"]) == BASE.n_markets


def test_dual_exposure_by_design():
    user = run_experiment(CFG, replace(BASE, randomization_unit="user"))
    sess = run_experiment(CFG, replace(BASE, randomization_unit="session"))
    assert dual_exposure(user["sessions"])["n_dual_exposed"] == 0
    assert dual_exposure(sess["sessions"])["share_dual_exposed"] > 0.05


def test_srm_clean_and_broken():
    out = run_experiment(CFG, BASE)
    clean = srm_check(out["fact_experiment"], BASE.weights())
    assert not clean["srm_alarm"]
    # a 70/30 realized split audited against a 50/50 plan must alarm
    skewed = run_experiment(CFG, replace(BASE, allocation=(0.7, 0.3)))
    broken = srm_check(skewed["fact_experiment"], np.array([0.5, 0.5]))
    assert broken["srm_alarm"]


def test_covariate_balance_clean():
    out = run_experiment(CFG, BASE)
    bal = covariate_balance(out["sessions"])
    assert not bal["flag"].any()


def test_aa_battery_nominal():
    res = aa_battery(CFG, replace(BASE, sessions_per_day=1_000), n_reps=40)
    assert res["pass"]


def test_mc_power_matches_analytic_for_session_split():
    exp = replace(BASE, randomization_unit="session",
                  horizon_days=2, sessions_per_day=1_500)  # small on purpose
    from demand_sim import demand_curve
    p = demand_curve(CFG, np.array(exp.arm_prices))["P_aggregate"].to_numpy()
    n_per_arm = exp.horizon_days * exp.sessions_per_day / 2
    # analytic power at this n (invert the n-for-power helper by search)
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    analytic = NormalIndPower().power(proportion_effectsize(p[0], p[1]),
                                      nobs1=n_per_arm, alpha=0.05, ratio=1)
    mc = monte_carlo_power(CFG, exp, n_reps=60)
    assert abs(mc["empirical_power"] - analytic) < 0.2
    assert mc["ci_coverage"] > 0.85


def test_rd_recovers_true_jump():
    s = run_surge_sessions(CFG, 400_000, base_price=25.0, seed=3)
    est = rd_estimate(s, cut=1.25).iloc[0]
    truth = true_rd_jump(CFG, 25.0, 1.25)
    assert abs(est["jump"] - truth) < 3 * est["se"]


def test_deterministic_under_seed():
    a = run_experiment(CFG, BASE)["sessions"]["purchased"].sum()
    b = run_experiment(CFG, BASE)["sessions"]["purchased"].sum()
    assert a == b
