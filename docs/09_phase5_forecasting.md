# Phase 5 — The Forecasting Benchmark

**Status:** implemented (v0.5.0)
**Module:** `forecast.py` · **Demo:** `examples/phase5_forecasting.py` · **Tests:** `tests/test_phase5.py`
**Grounding:** docs/02 (censored demand), docs/06 (the dataset), CLAUDE.md invariants

Phases 1–4 generate the dataset; Phase 5 consumes it. This is the payoff the
project exists for: train forecasters on the observable view, grade them
against oracle true demand — the evaluation no real dataset can run.

## 1. Design

Rolling-origin evaluation (default 28-day horizon) over every product-store
series in `fact_sales_daily`. Forecasters receive only observable,
known-future covariates — `date`, `avg_price`, `promo_flag`, calendar terms —
plus the training target `units_sold` and the observable `sold_out` flag.
The oracle enters exactly twice, both scoring-only: as the evaluation target
(`units_demanded`) and as the labeled `oracle_lambda` ceiling. A poisoning
test enforces this: every non-oracle model must produce byte-identical
forecasts when the oracle columns are destroyed.

## 2. Models × censoring treatments

| model | censoring handling |
|---|---|
| `seasonal_naive`, `ets` | blind — time-series baselines on raw sales |
| `poisson_glm_blind` | blind — sales treated as demand (the indicted practice) |
| `poisson_glm_drop_stockouts` | drop sold-out days from training (observable flag) |
| `poisson_glm_unconstrained` | EM: sold-out days are right-censored; impute E[D \| D ≥ sold] under the working Poisson model, refit |
| `oracle_lambda` | ORACLE — predicts the true conditional mean; residual is pure Poisson noise |

The EM E-step uses the closed form E[D | D ≥ c] = μ·sf(c−2)/sf(c−1) for
D ~ Poisson(μ). Identification note: within a series, price varies only
through promo discounts, so the GLM's price/promo terms are what let both
corrections reconstruct promo-day demand — the days censoring hits hardest.

## 3. The money table (8 products × 2 stores, 500 days, heavier censoring)

```
vs TRUE DEMAND [high_censoring slice]      mae    rmse    bias
seasonal_naive                           31.80   47.28   -6.37
ets                                      23.87   38.53   -9.29
poisson_glm_blind                        12.91   23.29  -10.69   <- systematic shortfall
poisson_glm_drop_stockouts                4.24    5.49   -0.17
poisson_glm_unconstrained                 4.63    6.19   -1.23
oracle_lambda                             4.15    5.45   -0.02   <- irreducible floor
```

**The trap:** the same blind GLM graded against *sales* shows bias −0.4 — it
looks calibrated on every metric a real forecaster can compute, while
under-forecasting true demand by ~11 units/day on the series that matter.
Censoring bias is invisible without ground truth; that is the argument for
experiment-backed unconstraining in production systems.

**Reading the fixes:** dropping stockout days works remarkably well *here*
because promo/price covariates fully explain the high-demand days and enough
uncensored promo days remain — a best case. EM unconstraining is the more
general tool (it keeps the sample and handles censoring within covariate
cells); it lands within noise of the ceiling on the committed dataset. Both
degrade gracefully to blind when `sold_out` never fires.

## 4. Scorecard schema

`run_forecast_benchmark(panel_oracle, cfg)` returns `scorecard` (model ×
eval_target × series-slice with mae/rmse/bias/wape) and `predictions`
(per-day point forecasts). Slices: `all_series` and `high_censoring` (top
quartile of oracle censoring share — a reporting label, never a feature).

## 5. Known gaps

- Point forecasts only; quantile loss (P50/P90, the Amazon metric) needs
  probabilistic forecasters — natural next step alongside the GluonTS/Chronos
  export hooks (docs/06 §6)
- ETS/seasonal-naive get no stockout-aware variants (state-space censoring
  handling is out of core-stack scope)
- Holiday covariates absent until the Phase 2 intensity gains them
