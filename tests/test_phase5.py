"""Phase 5 acceptance tests: the forecasting benchmark.

Encodes the CLAUDE.md invariants: observable/oracle split (no leakage into
any forecaster), grading against true demand, the censoring-bias money
result, the oracle ceiling, and seed determinism.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import warnings

import numpy as np
import pandas as pd
import pytest

from demand_sim import (ForecastBenchmarkConfig, PanelConfig, generate_panel,
                        run_forecast_benchmark)
from demand_sim.forecast import (FEATURE_COLS, OBSERVABLE_MODELS, _MODELS,
                                 _censored_poisson_mean)
from demand_sim.panel import ORACLE_COLS

warnings.filterwarnings("ignore")

CFG = ForecastBenchmarkConfig(n_origins=2)


@pytest.fixture(scope="module")
def bench():
    panel = generate_panel(PanelConfig(n_products=8, n_stores=2,
                                       n_days=500, seed=7))["panel_oracle"]
    return run_forecast_benchmark(panel, CFG)


# ---------------------------------------------------------- oracle hygiene
def test_no_oracle_leakage_into_forecasters(bench):
    """Every non-oracle model must produce identical forecasts when the
    oracle columns are removed entirely — the operational definition of
    'estimators only see the observable view'."""
    panel = generate_panel(PanelConfig(n_products=3, n_stores=1,
                                       n_days=400, seed=3))["panel_oracle"]
    cfg = ForecastBenchmarkConfig(n_origins=1, models=OBSERVABLE_MODELS)
    with_oracle = run_forecast_benchmark(panel, cfg)["predictions"]

    stripped = panel.copy()
    stripped[list(ORACLE_COLS)] = np.nan     # poison every oracle field
    without = run_forecast_benchmark(stripped.assign(
        # scoring columns must exist; fill with sales so metrics run
        units_demanded=stripped["units_sold"],
        lost_sales=0,
        lambda_true=0.0), cfg)["predictions"]

    np.testing.assert_allclose(with_oracle["pred"].to_numpy(),
                               without["pred"].to_numpy())


def test_feature_whitelist_excludes_oracle():
    assert not set(FEATURE_COLS) & set(ORACLE_COLS)
    assert "units_demanded" not in FEATURE_COLS
    assert "lambda_true" not in FEATURE_COLS


# ------------------------------------------------------- the money results
def _get(sc, model, slice_, target="units_demanded"):
    row = sc[(sc.model == model) & (sc.series_slice == slice_)
             & (sc.eval_target == target)]
    return row.iloc[0]


def test_censoring_blind_underforecasts_true_demand(bench):
    sc = bench["scorecard"]
    blind = _get(sc, "poisson_glm_blind", "high_censoring")
    assert blind["bias"] < -5          # severe under-forecast where it matters


def test_blind_model_looks_fine_vs_sales_the_trap(bench):
    sc = bench["scorecard"]
    vs_sales = _get(sc, "poisson_glm_blind", "high_censoring", "units_sold")
    vs_demand = _get(sc, "poisson_glm_blind", "high_censoring")
    assert abs(vs_sales["bias"]) < 2          # calibrated against sales...
    assert vs_demand["bias"] < vs_sales["bias"] - 5   # ...lying about demand


def test_stockout_aware_models_remove_the_bias(bench):
    sc = bench["scorecard"]
    blind = _get(sc, "poisson_glm_blind", "high_censoring")
    for fixed in ("poisson_glm_drop_stockouts", "poisson_glm_unconstrained"):
        row = _get(sc, fixed, "high_censoring")
        assert abs(row["bias"]) < 0.3 * abs(blind["bias"])
        assert row["rmse"] < 0.5 * blind["rmse"]


def test_oracle_lambda_is_the_ceiling(bench):
    sc = bench["scorecard"]
    sub = sc[(sc.eval_target == "units_demanded")
             & (sc.series_slice == "all_series")].set_index("model")
    floor = sub.loc["oracle_lambda", "rmse"]
    assert (sub.drop(index="oracle_lambda")["rmse"] >= floor).all()


# ----------------------------------------------------------------- mechanics
def test_censored_poisson_mean():
    # E[D | D >= 0] = mu; truncation raises the mean; large c dominates
    mu = np.array([5.0, 5.0, 5.0])
    c = np.array([0.0, 5.0, 12.0])
    e = _censored_poisson_mean(mu, c)
    assert e[0] == pytest.approx(5.0)
    assert e[1] > 5.0
    assert e[2] >= 12.0


def test_scorecard_covers_all_models_and_slices(bench):
    sc = bench["scorecard"]
    assert set(sc["model"]) == set(_MODELS)
    assert set(sc["series_slice"]) == {"all_series", "high_censoring"}
    assert set(sc["eval_target"]) == {"units_demanded", "units_sold"}


def test_deterministic(bench):
    panel = generate_panel(PanelConfig(n_products=8, n_stores=2,
                                       n_days=500, seed=7))["panel_oracle"]
    again = run_forecast_benchmark(panel, CFG)["predictions"]
    np.testing.assert_allclose(bench["predictions"]["pred"].to_numpy(),
                               again["pred"].to_numpy())