"""Phase 3: analytical guardrails (docs/04 §3).

Automatic post-run checks that must pass before any estimate is believed:

  srm_check          — sample-ratio mismatch (assignment/logging broken)
  covariate_balance  — standardized mean differences on pre-assignment covariates
  dual_exposure      — users observed in more than one arm
  aa_battery         — run identical-arm experiments; rejection rate must be ~alpha
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .experiments import ExperimentConfig, estimate_contrast, run_experiment

SRM_ALPHA = 1e-3      # docs/04 §3.1: p < 0.001 => stop the experiment
SMD_THRESHOLD = 0.1   # docs/04 §3.3


def srm_check(fact_experiment: pd.DataFrame,
              expected_weights: np.ndarray) -> dict:
    """Chi-square test of realized unit counts vs planned allocation."""
    from scipy.stats import chisquare
    counts = fact_experiment["arm"].value_counts().sort_index().to_numpy()
    expected = expected_weights * counts.sum()
    stat, p = chisquare(f_obs=counts, f_exp=expected)
    return {"counts": counts.tolist(), "chi2": float(stat), "p_value": float(p),
            "srm_alarm": bool(p < SRM_ALPHA)}


def covariate_balance(sessions: pd.DataFrame,
                      covariates: tuple[str, ...] = ("segment_proxy",),
                      arm_a: str = "arm_0", arm_b: str = "arm_1") -> pd.DataFrame:
    """Standardized mean differences across arms; |SMD| > 0.1 flags imbalance."""
    a = sessions[sessions["arm"] == arm_a]
    b = sessions[sessions["arm"] == arm_b]
    rows = []
    for c in covariates:
        pooled_sd = np.sqrt((a[c].var(ddof=1) + b[c].var(ddof=1)) / 2)
        smd = (b[c].mean() - a[c].mean()) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"covariate": c, "smd": float(smd),
                     "flag": bool(abs(smd) > SMD_THRESHOLD)})
    return pd.DataFrame(rows)


def dual_exposure(sessions: pd.DataFrame) -> dict:
    """Users observed in >1 arm (docs/04 §3.4 multi-arm exposure audit).

    Structural under session/switchback randomization; must be ZERO under
    user-split (anything else is an identity bug).
    """
    arms_per_user = sessions.groupby("consumer_id")["arm"].nunique()
    multi = int((arms_per_user > 1).sum())
    return {"n_users": int(len(arms_per_user)),
            "n_dual_exposed": multi,
            "share_dual_exposed": float(multi / len(arms_per_user))}


def aa_battery(cfg: SimulationConfig, exp: ExperimentConfig,
               n_reps: int = 50, alpha: float = 0.05) -> dict:
    """A/A test battery (docs/04 §3.2): identical arms, R replications.

    The rejection rate should be ~alpha. Materially higher means the
    variance estimator is wrong for the design (usually missing clustering).
    """
    anchor = exp.arm_prices[0]
    aa = replace(exp, arm_prices=(anchor, anchor))
    rejections = 0
    for r in range(n_reps):
        out = run_experiment(cfg, replace(aa, seed=exp.seed + 1000 + r))
        est = estimate_contrast(out["sessions"], aa)
        if abs(float(est["z"].iloc[0])) > 1.96:
            rejections += 1
    return {"n_reps": n_reps, "rejection_rate": rejections / n_reps,
            "nominal_alpha": alpha,
            "pass": bool(rejections / n_reps <= alpha + 2 * np.sqrt(
                alpha * (1 - alpha) / n_reps))}


def strategic_waiting_check(sessions: pd.DataFrame) -> dict:
    """Refresh-arbitrage signature (docs/04 §3.4).

    Under per-session randomization, strategic waiting means consumers come
    back BECAUSE the last price was high — so revisit sessions' previous
    prices skew above the price distribution served at random. Organic
    return traffic (price-independent) has uniform previous prices, which is
    the null this test is calibrated against.

    Requires `prev_price` (NaN on first visits; real logs derive it from
    consumer_id + timestamps). One-sided z-test of mean prev_price of
    revisit sessions vs the unconditional mean served price; alarm at
    p < 0.01.
    """
    revisits = sessions.loc[sessions["prev_price"].notna(), "prev_price"]
    baseline = sessions["quoted_price"]
    if len(revisits) < 30:
        return {"n_revisits": int(len(revisits)), "waiting_alarm": False,
                "note": "too few revisit sessions to test"}
    diff = float(revisits.mean() - baseline.mean())
    se = float(np.sqrt(revisits.var(ddof=1) / len(revisits)
                       + baseline.var(ddof=1) / len(baseline)))
    z = diff / se if se > 0 else 0.0
    from scipy.stats import norm
    p = float(norm.sf(z))  # one-sided: waiting pushes prev prices UP
    return {"n_revisits": int(len(revisits)),
            "mean_prev_price_revisits": float(revisits.mean()),
            "mean_price_served": float(baseline.mean()),
            "diff": diff, "z": float(z), "p_value": p,
            "waiting_alarm": bool(p < 0.01)}


def interference_probe(sessions: pd.DataFrame,
                       pure_control_sessions: pd.DataFrame,
                       arm_control: str = "arm_0") -> dict:
    """Pure-control probe (docs/04 §3.4): compare within-experiment control
    conversion against held-out markets untouched by the experiment. A gap
    beyond noise means treatment is leaking through shared capacity.
    """
    a = sessions.loc[sessions["arm"] == arm_control, "purchased"]
    b = pure_control_sessions["purchased"]
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    z = diff / se if se > 0 else 0.0
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z)))
    return {"conv_in_experiment_control": float(a.mean()),
            "conv_pure_control": float(b.mean()),
            "diff": diff, "z": float(z), "p_value": p,
            "interference_alarm": bool(p < 0.01)}


def run_guardrails(cfg: SimulationConfig, exp: ExperimentConfig,
                   out: dict) -> dict:
    """All post-run checks for one experiment output (docs/04 §5)."""
    return {
        "srm": srm_check(out["fact_experiment"], exp.weights()),
        "balance": covariate_balance(out["sessions"]),
        "dual_exposure": dual_exposure(out["sessions"]),
    }
