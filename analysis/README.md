# analysis/ — R Markdown estimator walk-throughs

One `.Rmd` per estimator from `docs/03_models_python_r.md` (the two Python-only
pillars — BLP and deep/probabilistic forecasting — are excluded). Each file is
deliberately **raw**: minimal helper functions, section-by-section, so the code
and its logic can be reviewed inline. Every file follows the same arc:

data → train → validate against the oracle → plot the demand curve → forecast demand (Phase 5)

| File | Estimator | Data | Headline |
|---|---|---|---|
| `01_log_log_ols.Rmd` | log-log OLS / PPML with FE (`fixest`) | daily panel | censoring + aggregation flatten the naive elasticity |
| `02_binary_logit_sessions.Rmd` | binary logit (`glm`) + power (`pwr`) | sessions | cell means beat functional form; pooled b is an aggregation compromise |
| `03_mixed_logit.Rmd` | mixed logit (`mlogit`) | sessions | normal mixing vs discrete segments — controlled misspecification |
| `04_dml.Rmd` | DML PLR vs PLIV (`DoubleML`) | endogenous market | flexible ML is not an identification strategy |
| `05_regression_discontinuity.Rmd` | RD (`rdrobust`, `rddensity`) | surge sessions | one certified step of the curve; LATE, not ATE |
| `06_tobit_censored.Rmd` | per-row Tobit (`survival::survreg`) | daily panel | the R money table — sales-calibrated, demand-wrong |
| `07_iv_2sls_control_function.Rmd` | 2SLS (`AER::ivreg`) + control function | endogenous market | prediction accuracy is not causal validity |

## Data

All inputs live in `../data/` and are seed-reproducible:

```bash
python scripts/generate_phase2_dataset.py      # fact_sales_daily + dims
python scripts/generate_analysis_datasets.py   # sessions, surge, endogenous, truth files
```

Observable files (estimator inputs): `fact_sales_daily.csv`,
`fact_session.csv.gz`, `surge_sessions.csv.gz`, `endogenous_market.csv`.
Oracle files (used ONLY in validation sections and truth overlays):
`ground_truth_curves.csv`, `dim_segment_truth.csv`, `truth_params.csv`, and the
oracle columns of the panel. Keep that boundary intact if you extend these.

## Setup (VS Code)

1. Install R (≥ 4.2), then the packages:

```r
install.packages(c(
  "rmarkdown", "knitr", "languageserver",          # tooling
  "ggplot2", "fixest", "sandwich", "lmtest",       # 01, 02
  "pwr", "mlogit",                                 # 02, 03
  "DoubleML", "mlr3", "mlr3learners", "ranger",    # 04
  "rdrobust", "rddensity",                         # 05
  "survival", "AER"                                # 06, 07
))
```

2. Install **pandoc** — `rmarkdown::render` requires it, and it does NOT ship
   with R (it ships with RStudio, which we're not using):

```bash
brew install pandoc        # macOS; verify with: pandoc -v
```

3. VS Code: install the **R extension** (REditorSupport). `languageserver`
   powers completion/linting; [radian](https://github.com/randy3k/radian) is a
   nicer console but optional.
4. Knit from the repo's `analysis/` directory (relative paths assume it):
   open a file and run **R: Knit Rmd**, or from a terminal:

```bash
cd analysis
Rscript -e 'rmarkdown::render("01_log_log_ols.Rmd")'
```

Chunks can also be run interactively (Ctrl/Cmd+Enter) — each file is written
to be stepped through top to bottom.

## Runtime notes

- `03_mixed_logit.Rmd` (simulated ML, 300 Halton draws) and `04_dml.Rmd`
  (cross-fitted forests) are the slow ones — a few minutes each; both
  subsample/cache accordingly.
- Every file is deterministic: `set.seed` where the estimator itself draws.
- These are reference implementations knit locally, not CI-executed; the
  Python package (`src/demand_sim/`, `tests/`) remains the load-bearing path.
