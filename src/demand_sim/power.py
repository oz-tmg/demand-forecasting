"""Phase 3: Monte Carlo power and the experiment scorecard (docs/04 §2, §5).

`metrics.power_two_proportions` is the analytic answer for iid session-split
designs. Monte Carlo power is the ground truth for every design: run the whole
pipeline R times, count rejections. The scorecard grades each design on bias,
RMSE, CI coverage, empirical power, and guardrail outcomes — against oracle
truth from the Phase 1 demand engine.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .experiments import (ExperimentConfig, estimate_contrast, run_experiment,
                          true_contrast)
from .guardrails import dual_exposure, srm_check


def monte_carlo_power(cfg: SimulationConfig, exp: ExperimentConfig,
                      n_reps: int = 100) -> dict:
    """Empirical power for the arm_1 vs arm_0 contrast under `exp`."""
    truth = true_contrast(cfg, exp)
    rejects, estimates, covered = 0, [], 0
    for r in range(n_reps):
        out = run_experiment(cfg, replace(exp, seed=exp.seed + 5000 + r))
        est = estimate_contrast(out["sessions"], exp).iloc[0]
        estimates.append(est["diff"])
        if abs(est["z"]) > 1.96:
            rejects += 1
        if est["ci_lo"] <= truth <= est["ci_hi"]:
            covered += 1
    estimates = np.array(estimates)
    return {
        "true_diff": truth,
        "mean_estimate": float(estimates.mean()),
        "bias": float(estimates.mean() - truth),
        "rmse": float(np.sqrt(((estimates - truth) ** 2).mean())),
        "empirical_power": rejects / n_reps,
        "ci_coverage": covered / n_reps,
        "n_reps": n_reps,
    }


def experiment_scorecard(cfg: SimulationConfig,
                         base: ExperimentConfig,
                         designs: tuple[str, ...] = ("user", "session",
                                                     "switchback_window",
                                                     "market"),
                         n_reps: int = 50) -> pd.DataFrame:
    """Design x estimator scorecard (docs/04 §5).

    Same market, same arms, same session budget — only the randomization
    unit changes. Read across rows to see the power cost of clustering,
    switchbacks, and geo designs.
    """
    rows = []
    for unit in designs:
        exp = replace(base, randomization_unit=unit)  # type: ignore[arg-type]
        mc = monte_carlo_power(cfg, exp, n_reps=n_reps)
        out = run_experiment(cfg, exp)
        srm = srm_check(out["fact_experiment"], exp.weights())
        dual = dual_exposure(out["sessions"])
        est = estimate_contrast(out["sessions"], exp).iloc[0]
        rows.append({
            "randomization_unit": unit,
            "n_units": int(est["n_units_a"] + est["n_units_b"]),
            "true_diff": mc["true_diff"],
            "bias": mc["bias"],
            "rmse": mc["rmse"],
            "empirical_power": mc["empirical_power"],
            "ci_coverage": mc["ci_coverage"],
            "srm_alarm": srm["srm_alarm"],
            "share_dual_exposed": dual["share_dual_exposed"],
        })
    return pd.DataFrame(rows)
