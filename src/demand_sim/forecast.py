"""Phase 5: the forecasting benchmark — the payoff module.

Everything before this generates the dataset; this consumes it. Forecasters
train on the OBSERVABLE view only (units_sold + known-future covariates:
price, promo, calendar) and are graded against ORACLE true demand
(units_demanded) — the evaluation no real dataset can run. The oracle enters
exactly twice, both scoring-only: as the evaluation target, and as the
labeled `oracle_lambda` accuracy ceiling.

The economics: units_sold = min(units_demanded, on_hand), and censoring
concentrates on promo days (docs/06 §2) — precisely the high-demand days a
forecaster must get right. So training on sales as if they were demand
under-forecasts the fast movers. Three censoring treatments make the point:

  blind          — train on units_sold as-is (the common practice)
  drop_stockouts — exclude sold-out days from training (observable flag;
                   loses exactly the informative high-demand days)
  unconstrained  — EM: treat sold-out days as right-censored, impute
                   E[D | D >= sold] under the working Poisson model, refit

Models are deliberately core-stack (statsmodels only): seasonal naive, ETS,
Poisson GLM on known-future covariates. GluonTS/Chronos remain pending
hooks per PROJECT_CONTEXT.md — this benchmark is about censoring handling,
not architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Observable, known-future covariates a real forecaster may use.
# Everything else in fact_sales_daily is either target, inventory telemetry,
# or oracle — see the whitelist test in tests/test_phase5.py.
FEATURE_COLS = ("date", "avg_price", "promo_flag", "dow")
ORACLE_TARGET = "units_demanded"
OBS_TARGET = "units_sold"


@dataclass(frozen=True)
class ForecastBenchmarkConfig:
    horizon_days: int = 28
    n_origins: int = 2            # rolling-origin evaluation folds
    em_iterations: int = 4
    models: tuple[str, ...] = ("seasonal_naive", "ets",
                               "poisson_glm_blind",
                               "poisson_glm_drop_stockouts",
                               "poisson_glm_unconstrained",
                               "oracle_lambda")
    seed: int = 42


# ---------------------------------------------------------------- features
def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Known-future covariates only: log price, promo, dow, trend, annual.

    Identification note: within a series, price varies only through promo
    discounts, so the GLM's price/promo terms absorb the promo-driven demand
    spikes — which is what makes promo-day censoring correction possible.
    """
    d = pd.to_datetime(df["date"])
    t = (d - pd.Timestamp("2024-01-01")).dt.days.to_numpy() / 365.25
    doy = d.dt.dayofyear.to_numpy() / 365.25
    dow = pd.get_dummies(d.dt.dayofweek, drop_first=True)
    dow = dow.reindex(columns=range(1, 7), fill_value=0).to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(df)),
        np.log(df["avg_price"].to_numpy(dtype=float)),
        df["promo_flag"].to_numpy(dtype=float),
        t,
        np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy),
        dow,
    ])


def _censored_poisson_mean(mu: np.ndarray, c: np.ndarray) -> np.ndarray:
    """E[D | D >= c] for D ~ Poisson(mu).

    Closed form: sum_{k>=c} k pmf(k) = mu * P(D >= c-1), so
    E[D | D >= c] = mu * sf(c-2) / sf(c-1) with scipy's sf(x) = P(D > x).
    """
    from scipy.stats import poisson
    c = np.asarray(c, dtype=float)
    num = poisson.sf(c - 2, mu)
    den = poisson.sf(c - 1, mu)
    out = np.where(den > 0, mu * num / np.maximum(den, 1e-300), c)
    return np.where(c <= 0, mu, out)


# ---------------------------------------------------------------- forecasters
def _fit_predict_glm(train: pd.DataFrame, future: pd.DataFrame,
                     y: np.ndarray, weights: np.ndarray | None = None
                     ) -> np.ndarray:
    import statsmodels.api as sm
    X, Xf = _design_matrix(train), _design_matrix(future)
    model = sm.GLM(y, X, family=sm.families.Poisson(),
                   freq_weights=weights)
    fit = model.fit()
    return np.asarray(fit.predict(Xf))


def seasonal_naive(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Same weekday, most recent observed week."""
    last = train.tail(7).set_index(train.tail(7)["dow"].to_numpy())[OBS_TARGET]
    return last.reindex(future["dow"].to_numpy()).to_numpy(dtype=float)


def ets(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Additive ETS with weekly seasonality on raw sales (censoring-blind)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = train[OBS_TARGET].to_numpy(dtype=float)
    try:
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=7, initialization_method="estimated").fit()
        return np.clip(np.asarray(fit.forecast(len(future))), 0, None)
    except Exception:                        # rare non-convergence
        return seasonal_naive(train, future)


def poisson_glm_blind(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Sales treated as demand — the practice this benchmark indicts."""
    return _fit_predict_glm(train, future, train[OBS_TARGET].to_numpy(float))


def poisson_glm_drop_stockouts(train: pd.DataFrame,
                               future: pd.DataFrame) -> np.ndarray:
    """Exclude sold-out days (observable flag). Removes contaminated labels
    but discards exactly the high-demand days — bias down, variance up, and
    a selection effect when stockouts correlate with promos (they do)."""
    keep = ~train["sold_out"].to_numpy(bool)
    return _fit_predict_glm(train[keep], future,
                            train.loc[keep, OBS_TARGET].to_numpy(float))


def poisson_glm_unconstrained(train: pd.DataFrame, future: pd.DataFrame,
                              n_iter: int = 4) -> np.ndarray:
    """EM unconstraining: sold-out days are right-censored observations.

    E-step imputes E[D | D >= units_sold] under the working model; M-step
    refits the GLM on imputed demand. Uses only observable columns
    (sold_out, units_sold) — the oracle stays out of the training loop.
    """
    censored = train["sold_out"].to_numpy(bool)
    y = train[OBS_TARGET].to_numpy(dtype=float).copy()
    for _ in range(n_iter):
        import statsmodels.api as sm
        X = _design_matrix(train)
        fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = np.asarray(fit.predict(X))
        y_new = train[OBS_TARGET].to_numpy(dtype=float).copy()
        y_new[censored] = _censored_poisson_mean(
            mu[censored], train.loc[censored, OBS_TARGET].to_numpy(float))
        if np.allclose(y_new, y, atol=1e-3):
            y = y_new
            break
        y = y_new
    return _fit_predict_glm(train, future, y)


def oracle_lambda(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """ORACLE ceiling: the true conditional mean lambda_t. Scoring benchmark
    only — no real forecaster can use this. Its error is pure Poisson noise,
    the irreducible floor every model row should be read against."""
    return future["lambda_true"].to_numpy(dtype=float)


_MODELS = {
    "seasonal_naive": seasonal_naive,
    "ets": ets,
    "poisson_glm_blind": poisson_glm_blind,
    "poisson_glm_drop_stockouts": poisson_glm_drop_stockouts,
    "poisson_glm_unconstrained": poisson_glm_unconstrained,
    "oracle_lambda": oracle_lambda,
}
OBSERVABLE_MODELS = tuple(m for m in _MODELS if m != "oracle_lambda")


# ---------------------------------------------------------------- benchmark
def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = pred - actual
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()),
            "wape": float(np.abs(err).sum() / max(actual.sum(), 1e-9))}


def run_forecast_benchmark(panel_oracle: pd.DataFrame,
                           cfg: ForecastBenchmarkConfig | None = None
                           ) -> dict[str, pd.DataFrame]:
    """Rolling-origin benchmark over every product-store series.

    Returns:
      scorecard      — model x eval_target x censoring-slice metrics
      predictions    — per (series, origin, model, day) point forecasts
    Both grade against ORACLE units_demanded (primary) and, for the trap
    exhibit, against observable units_sold: a censoring-blind model looks
    fine vs sales while under-forecasting true demand.
    """
    cfg = cfg or ForecastBenchmarkConfig()
    df = panel_oracle.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "store_id", "date"])
    if "dow" not in df.columns:
        df["dow"] = df["date"].dt.dayofweek

    # censoring-severity slice (oracle-derived, used for REPORTING only)
    sev = (df.groupby(["product_id", "store_id"])
             .apply(lambda g: g["lost_sales"].sum()
                    / max(g[ORACLE_TARGET].sum(), 1), include_groups=False)
             .rename("censoring_share"))
    hi_cut = sev.quantile(0.75)

    n_days = df["date"].nunique()
    all_dates = np.sort(df["date"].unique())
    origins = [n_days - cfg.horizon_days * k for k in range(cfg.n_origins, 0, -1)]

    pred_rows = []
    for (pid, sid), g in df.groupby(["product_id", "store_id"]):
        g = g.reset_index(drop=True)
        for origin in origins:
            train_full = g.iloc[:origin]
            future_full = g.iloc[origin:origin + cfg.horizon_days]
            # observable slices for the fitters; oracle kept aside for scoring
            obs_cols = list(FEATURE_COLS) + [OBS_TARGET, "sold_out",
                                             "promo_depth"]
            train = train_full[obs_cols]
            future_obs = future_full[list(FEATURE_COLS) + ["promo_depth"]]
            for name in cfg.models:
                fn = _MODELS[name]
                fut = future_full if name == "oracle_lambda" else future_obs
                pred = np.clip(np.asarray(fn(train, fut), dtype=float), 0, None)
                pred_rows.append(pd.DataFrame({
                    "product_id": pid, "store_id": sid,
                    "origin": all_dates[origin],
                    "date": future_full["date"].to_numpy(),
                    "model": name,
                    "pred": pred,
                    "actual_demand": future_full[ORACLE_TARGET].to_numpy(),
                    "actual_sales": future_full[OBS_TARGET].to_numpy(),
                    "high_censoring": bool(sev.loc[(pid, sid)] >= hi_cut),
                }))

    preds = pd.concat(pred_rows, ignore_index=True)

    rows = []
    slices = {"all_series": preds,
              "high_censoring": preds[preds["high_censoring"]]}
    for slice_name, sl in slices.items():
        for name, gm in sl.groupby("model"):
            for target, col in (("units_demanded", "actual_demand"),
                                ("units_sold", "actual_sales")):
                m = _metrics(gm["pred"].to_numpy(), gm[col].to_numpy())
                rows.append({"model": name, "eval_target": target,
                             "series_slice": slice_name, **m})
    order = {m: i for i, m in enumerate(_MODELS)}
    scorecard = (pd.DataFrame(rows)
                 .sort_values(["series_slice", "eval_target", "model"],
                              key=lambda s: s.map(order).fillna(s))
                 .reset_index(drop=True))
    return {"scorecard": scorecard, "predictions": preds}
