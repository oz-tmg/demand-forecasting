# Model Playbook — The Estimators, in Python and R

R has historically owned applied econometrics, but every model this project needs now has a
first-class Python implementation — in one case (BLP) Python is now *ahead* of R. Since the
simulator and the Amazon-style stack are Python, **Python is the primary language; R snippets
are included as the reference implementations** many papers used.

Package parity at a glance:

| Model | R | Python | Parity notes |
|---|---|---|---|
| Log-log OLS / FE panels | `fixest` | `pyfixest`, `statsmodels`, `linearmodels` | `pyfixest` is a deliberate port of `fixest`; near-identical API |
| Binary logit / GLM | `glm` | `statsmodels` | full parity |
| Mixed logit / conjoint | `mlogit`, `logitr` | `xlogit`, `pylogit` | R slightly richer; `xlogit` is fast (GPU) |
| DML | `DoubleML` | `DoubleML`, `EconML` | same team ships both; identical algorithms |
| IV / 2SLS | `AER::ivreg`, `fixest` | `linearmodels.IV2SLS` | full parity |
| Regression discontinuity | `rdrobust` | `rdrobust` (pip) | same authors, same package name |
| BLP | (no maintained pkg) | `PyBLP` | **Python only** — the field standard |
| Tobit / censored | `AER::tobit` | `statsmodels`, custom MLE | R more convenient; Py workable |
| Probabilistic forecasting | `fable`, `forecast` | GluonTS (DeepAR), Chronos, `statsforecast` | **Python only** for the deep/foundation models |
| Power analysis | `pwr` | `statsmodels.stats.power` | full parity |

Verdict: nothing in this project requires R; two pillars (BLP, deep forecasting) require Python.

---

## 1. Log-log OLS (the baseline everyone starts with)

Constant-elasticity model: `log Q = α + θ log P + Xγ + u`, θ = elasticity.

**Python**
```python
import pyfixest as pf
# daily panel: log units on log price with product & week fixed effects
fit = pf.feols("np.log(units_sold) ~ np.log(avg_price) | product_id + week", data=panel)
fit.summary()          # θ = coefficient on log(avg_price)
```

**R**
```r
library(fixest)
fit <- feols(log(units_sold) ~ log(avg_price) | product_id + week, data = panel)
summary(fit)
```

*Use as:* the naive benchmark. On simulator data with endogenous pricing (Phase 4) this is the
estimator we show being wrong (docs/05). Note the zero-sales problem: `log(0)` drops no-sale
days — itself a selection bias; prefer PPML (`pf.fepois` / `fepois`) when zeros matter.

## 2. Binary logit on sessions (the simulator's native estimand)

With session data, demand is a purchase probability: `P(buy) = σ(a + b·price)`.
Elasticity at p: `b · p · (1 − P(p))`.

**Python**
```python
import statsmodels.formula.api as smf
fit = smf.logit("purchased ~ quoted_price", data=sessions).fit()
b = fit.params["quoted_price"]
```

**R**
```r
fit <- glm(purchased ~ quoted_price, data = sessions, family = binomial())
```

*Simulator tie-in:* `metrics.py` runs exactly this and scores `b` against the ground-truth
`price_coef`. Under segment mixtures, pooled logit is misspecified — the recovered b is a
weighted compromise; the scoring report quantifies the aggregation bias.

## 3. Mixed logit (heterogeneity / conjoint workhorse)

Random coefficients over individuals; recovers the WTP *distribution*, not just the mean.

**Python**
```python
from xlogit import MixedLogit
m = MixedLogit()
m.fit(X=df[["price", "promo"]], y=df["chose"], ids=df["session_id"],
      panels=df["consumer_id"], varnames=["price", "promo"],
      randvars={"price": "n"}, n_draws=600)   # price coef ~ Normal
m.summary()
```

**R**
```r
library(mlogit)
m <- mlogit(chose ~ price + promo, data = dfml,
            rpar = c(price = "n"), panel = TRUE, R = 600, halton = NA)
```

*Simulator tie-in:* the ground truth IS a finite-mixture logit, so mixed logit with a normal
mixing distribution is a controlled misspecification test — how well does a continuous mixture
approximate discrete segments?

## 4. Double Machine Learning

Partially linear: `log Q = θ log P + g(X) + u`. See docs/05 for when this works.

**Python**
```python
from doubleml import DoubleMLData, DoubleMLPLR
from lightgbm import LGBMRegressor
dml_data = DoubleMLData(panel, y_col="log_q", d_cols="log_p", x_cols=confounders)
mod = DoubleMLPLR(dml_data,
                  ml_l=LGBMRegressor(), ml_m=LGBMRegressor(),
                  n_folds=5)                 # cross-fitting is non-optional
print(mod.fit().summary)                     # θ with valid SEs
```

Heterogeneous elasticity (CATE):
```python
from econml.dml import LinearDML
est = LinearDML(model_y=LGBMRegressor(), model_t=LGBMRegressor(), cv=5)
est.fit(Y=log_q, T=log_p, X=segment_proxies, W=confounders)
theta_x = est.effect(X_grid)                 # elasticity by segment features
```

**R**
```r
library(DoubleML); library(mlr3); library(mlr3learners)
obj <- DoubleMLData$new(panel, y_col = "log_q", d_cols = "log_p", x_cols = confounders)
mod <- DoubleMLPLR$new(obj,
        ml_l = lrn("regr.ranger"), ml_m = lrn("regr.ranger"), n_folds = 5)
mod$fit(); mod$summary()
```

## 5. Regression discontinuity (Uber replication)

**Python** (`pip install rdrobust`)
```python
from rdrobust import rdrobust, rdplot
# running var: latent surge minus threshold; outcome: purchased
res = rdrobust(y=df["purchased"], x=df["latent_minus_threshold"], c=0)
print(res)      # local jump in purchase prob at the price discontinuity
```

**R**
```r
library(rdrobust)
res <- rdrobust(y = df$purchased, x = df$latent_minus_threshold, c = 0)
summary(res)
```

Convert the jump to a local elasticity: `ε ≈ (ΔP/P) / (Δprice/price)` at the threshold.
Run the McCrary density test (`rddensity`, both languages) as the manipulation check.

## 6. BLP / random-coefficients demand (structural)

**Python only — `PyBLP`** (Conlon & Gortmaker). R has no maintained equivalent; this alone
settles the language question for structural work.

```python
import pyblp
product_formulations = (
    pyblp.Formulation("1 + prices"),          # linear part
    pyblp.Formulation("1 + prices"),          # random coefficients
)
problem = pyblp.Problem(product_formulations, product_data)   # shares, prices, instruments
results = problem.solve(sigma=np.eye(2))
elasticities = results.compute_elasticities()                  # full own/cross matrix
```

*Simulator tie-in (Phase 4):* aggregate `fact_session` to market shares, add cost-shifter
instruments to the DGP, and test whether BLP recovers the mixture.

## 7. Censored-demand corrections (pairs with docs/02)

**Exposure-offset GLM — Python**
```python
import statsmodels.api as sm
# demand rate per in-stock hour; offset = log exposure
glm = sm.GLM(panel.units_sold,
             sm.add_constant(panel[["log_price", "promo"]]),
             family=sm.families.NegativeBinomial(),
             offset=np.log(panel.in_stock_hours.clip(lower=0.1)))
res = glm.fit()
```

**Tobit — R** (reference implementation)
```r
library(AER)
fit <- tobit(units_sold ~ log_price + promo, right = Inf,
             left = -Inf, data = panel)   # configure censoring point per row via survreg
```
(Python: censored likelihood is ~30 lines of custom MLE with `scipy.optimize`, or Bayesian via
`pymc` — workable, just less turnkey. The simulator will ship the custom MLE in Phase 2.)

## 8. Probabilistic forecasting (the Amazon side)

**Python only.**
```python
# DeepAR via GluonTS
from gluonts.torch import DeepAREstimator
est = DeepAREstimator(freq="D", prediction_length=28,
                      num_feat_dynamic_real=3)      # price, promo, holiday
predictor = est.train(training_dataset)

# Chronos (zero-shot foundation model)
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base")
quantiles, mean = pipe.predict_quantiles(context, prediction_length=28,
                                         quantile_levels=[0.1, 0.5, 0.9])
```

*Simulator tie-in (Phase 2):* export `fact_sales_daily` in GluonTS format; evaluate with
weighted quantile loss against **true demand** (oracle) vs. censored sales to quantify §2.3 of
docs/02.

## 9. Power analysis

**Python**
```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
es = proportion_effectsize(0.10, 0.085)          # conv 10% vs 8.5% (a real elasticity signal)
n = NormalIndPower().solve_power(es, alpha=0.05, power=0.8, ratio=1)
```

**R**
```r
library(pwr)
pwr.2p.test(h = ES.h(0.10, 0.085), sig.level = 0.05, power = 0.8)
```

See docs/04 for how these numbers translate into experiment design.
