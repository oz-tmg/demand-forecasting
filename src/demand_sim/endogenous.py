"""Phase 4 flagship: endogenous (demand-responsive) pricing (docs/05).

The pricer partially observes the demand shock u and prices into it — the
mechanism that makes observational elasticity estimates lie. The DGP is
docs/05 §2 exactly:

    log_p = beta_xp*x + pi*z + gamma*u + noise      (price responds to u!)
    log_q = alpha + theta*log_p + beta_x*x + u

gamma is the endogeneity dial, pi the instrument-strength dial (cost shifter
z is a valid instrument: moves price, not demand). Estimators:

    fit_ols               — biased: plim = theta + gamma-driven Cov(P,u)/Var(P)
    fit_2sls              — IV fix, with first-stage F (weak-IV guard)
    fit_control_function  — two-step residual-inclusion fix (Petrin & Train)

`endogeneity_scorecard` sweeps gamma and draws the bias curves docs/05 §8
asks for: OLS bias grows with gamma, IV/CF stay flat while instruments are
strong. DML variants live in the optional [estimators] extra and are out of
scope here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

ORACLE_COLS = ["u"]


@dataclass(frozen=True)
class EndogenousConfig:
    """docs/05 §2 demonstration DGP with dials."""

    n_periods: int = 20_000
    theta: float = -1.5              # TRUE price elasticity (oracle)
    gamma: float = 0.5               # endogeneity strength: pricing into shocks
    pi: float = 0.4                  # instrument strength (first-stage coef on z)
    alpha: float = 2.0
    beta_x: float = 0.8              # demand effect of observed confounder
    beta_xp: float = 0.3             # price effect of observed confounder
    sd_u: float = 0.5
    sd_price_noise: float = 0.3
    seed: int = 7


def simulate_endogenous_market(cfg: EndogenousConfig) -> pd.DataFrame:
    """Observable columns: log_q, log_p, x, z. Oracle: u (the demand shock)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_periods
    u = rng.normal(0, cfg.sd_u, n)
    z = rng.normal(0, 1.0, n)
    x = rng.normal(0, 1.0, n)
    log_p = (cfg.beta_xp * x + cfg.pi * z + cfg.gamma * u
             + rng.normal(0, cfg.sd_price_noise, n))
    log_q = cfg.alpha + cfg.theta * log_p + cfg.beta_x * x + u
    return pd.DataFrame({"log_q": log_q, "log_p": log_p, "x": x, "z": z, "u": u})


# ------------------------------------------------------------------ estimators
def fit_ols(df: pd.DataFrame) -> dict:
    """Naive OLS of log_q on log_p and x — biased whenever gamma != 0."""
    import statsmodels.api as sm
    X = sm.add_constant(df[["log_p", "x"]].to_numpy())
    fit = sm.OLS(df["log_q"].to_numpy(), X).fit(cov_type="HC1")
    return {"estimator": "ols", "theta_hat": float(fit.params[1]),
            "se": float(fit.bse[1])}


def fit_2sls(df: pd.DataFrame) -> dict:
    """2SLS with z instrumenting log_p; reports first-stage F (must be > 10)."""
    import statsmodels.api as sm
    n = len(df)
    # first stage: log_p ~ z + x
    Z = sm.add_constant(df[["z", "x"]].to_numpy())
    first = sm.OLS(df["log_p"].to_numpy(), Z).fit()
    f_stat = float(first.tvalues[1] ** 2)          # F of the excluded instrument
    p_hat = first.fittedvalues
    # second stage: log_q ~ p_hat + x, with 2SLS-correct residual variance
    X2 = sm.add_constant(np.column_stack([p_hat, df["x"].to_numpy()]))
    second = sm.OLS(df["log_q"].to_numpy(), X2).fit()
    theta_hat = float(second.params[1])
    # correct sigma^2 uses residuals at the ACTUAL log_p, not p_hat
    X_actual = sm.add_constant(np.column_stack([df["log_p"].to_numpy(),
                                                df["x"].to_numpy()]))
    resid = df["log_q"].to_numpy() - X_actual @ second.params
    sigma2 = float(resid @ resid) / (n - X2.shape[1])
    XtX_inv = np.linalg.inv(X2.T @ X2)
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    return {"estimator": "2sls", "theta_hat": theta_hat, "se": se,
            "first_stage_F": f_stat, "weak_instrument": bool(f_stat < 10)}


def fit_control_function(df: pd.DataFrame) -> dict:
    """Residual-inclusion CF (docs/05 §4); t on v_hat is the endogeneity test."""
    import statsmodels.api as sm
    Z = sm.add_constant(df[["z", "x"]].to_numpy())
    v_hat = sm.OLS(df["log_p"].to_numpy(), Z).fit().resid
    X = sm.add_constant(np.column_stack([df["log_p"].to_numpy(),
                                         df["x"].to_numpy(), v_hat]))
    fit = sm.OLS(df["log_q"].to_numpy(), X).fit()
    return {"estimator": "control_function",
            "theta_hat": float(fit.params[1]), "se": float(fit.bse[1]),
            "endogeneity_t": float(fit.tvalues[3])}


# ------------------------------------------------------------------- scorecard
def endogeneity_scorecard(base: EndogenousConfig | None = None,
                          gammas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
                          n_reps: int = 20) -> pd.DataFrame:
    """Bias curves: {OLS, 2SLS, CF} x endogeneity strength (docs/05 §8).

    Expected picture: OLS bias grows ~linearly in gamma; IV and CF stay flat
    at zero while the first-stage F is healthy.
    """
    base = base or EndogenousConfig()
    rows = []
    for g in gammas:
        ests: dict[str, list[float]] = {"ols": [], "2sls": [], "control_function": []}
        f_stats = []
        for r in range(n_reps):
            df = simulate_endogenous_market(replace(base, gamma=g,
                                                    seed=base.seed + 100 * r))
            ests["ols"].append(fit_ols(df)["theta_hat"])
            iv = fit_2sls(df)
            ests["2sls"].append(iv["theta_hat"])
            f_stats.append(iv["first_stage_F"])
            ests["control_function"].append(fit_control_function(df)["theta_hat"])
        for name, vals in ests.items():
            v = np.array(vals)
            rows.append({
                "gamma": g, "estimator": name,
                "theta_true": base.theta,
                "mean_theta_hat": float(v.mean()),
                "bias": float(v.mean() - base.theta),
                "rmse": float(np.sqrt(((v - base.theta) ** 2).mean())),
                "mean_first_stage_F": float(np.mean(f_stats)) if name == "2sls" else np.nan,
            })
    return pd.DataFrame(rows)
