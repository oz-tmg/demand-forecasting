# Endogenous Pricing — Why Naive Estimates Lie, and How to Recover the Truth

Phase 4's flagship feature: a pricing rule that *responds to demand*, which is how real prices
are set and the reason observational elasticity estimates are usually wrong. This doc breaks
down every mechanism, shows the bias arising in a simulated DGP, and implements each fix in
Python and R.

---

## 1. What "endogenous" means here

The regression `log Q = θ log P + u` is only causal if `P ⊥ u`. Three mechanisms break that
in practice:

1. **Simultaneity (demand-responsive pricing).** The pricer sees a positive demand shock
   (trend spike, viral moment, weather) and raises price. High-u periods are high-P periods
   ⇒ `Cov(P, u) > 0` ⇒ OLS elasticity is biased **upward** (toward zero, often flipping
   positive — the classic "we raised prices and sold more" illusion).
2. **Omitted variables.** Promotions co-move with price cuts *and* independently lift demand
   (email blasts, placement). If promo isn't controlled, the price cut absorbs the promo
   lift ⇒ elasticity biased **away from zero** (too negative). Same story for quality,
   seasonality, competitor moves.
3. **Algorithmic feedback.** A surge/repricing algorithm sets `P = f(recent demand)`. This is
   simultaneity implemented in code — and it's the norm at Amazon/Uber-scale. (Its one silver
   lining: discontinuities in f create RD identification — the Uber trick.)

The simulator implements each mechanism with a strength dial, so every estimator's bias can be
measured as a *function of endogeneity strength* — a plot no real dataset can produce.

## 2. The demonstration DGP

Log-linear demand with true elasticity θ = −1.5; the pricer partially observes the demand
shock and prices into it (γ controls endogeneity); a cost shifter Z moves price but not
demand (our instrument).

**Python**
```python
import numpy as np
rng = np.random.default_rng(7)
n, theta, gamma = 20_000, -1.5, 0.5

u  = rng.normal(0, 0.5, n)                  # demand shock (unobserved)
z  = rng.normal(0, 1.0, n)                  # cost shifter: valid instrument
x  = rng.normal(0, 1.0, n)                  # observed confounder (e.g., seasonality index)

log_p = 0.3*x + 0.4*z + gamma*u + rng.normal(0, 0.3, n)   # price responds to u!
log_q = 2.0 + theta*log_p + 0.8*x + u

# --- Naive OLS ---------------------------------------------------------
import statsmodels.api as sm
ols = sm.OLS(log_q, sm.add_constant(np.c_[log_p, x])).fit()
print(ols.params[1])        # ≈ -0.9, badly attenuated vs. -1.5
```

**R**
```r
set.seed(7); n <- 20000; theta <- -1.5; gamma <- 0.5
u <- rnorm(n, 0, .5); z <- rnorm(n); x <- rnorm(n)
log_p <- .3*x + .4*z + gamma*u + rnorm(n, 0, .3)
log_q <- 2 + theta*log_p + .8*x + u
coef(lm(log_q ~ log_p + x))["log_p"]    # ≈ -0.9
```

The bias formula makes the dial explicit: `plim θ̂_OLS = θ + Cov(P̃, u)/Var(P̃)` — every unit of
pricing-into-demand-shocks pushes the estimate up.

## 3. Fix #1 — Instrumental variables / 2SLS

Use variation in price driven by something that *doesn't* shift demand: cost shocks, exchange
rates, wholesale prices, Hausman instruments (same product's price in other markets), BLP
instruments (rival product characteristics).

**Python (`linearmodels`)**
```python
import pandas as pd
from linearmodels.iv import IV2SLS
df = pd.DataFrame(dict(log_q=log_q, log_p=log_p, x=x, z=z))
iv = IV2SLS.from_formula("log_q ~ 1 + x + [log_p ~ z]", df).fit(cov_type="robust")
print(iv.params["log_p"])       # ≈ -1.5, recovered
print(iv.first_stage)           # ALWAYS check: first-stage F > 10 (weak-IV guard)
```

**R (`AER` / `fixest`)**
```r
library(AER)
iv <- ivreg(log_q ~ log_p + x | z + x)      # z instruments log_p
summary(iv, diagnostics = TRUE)              # weak-instrument F, Wu-Hausman, Sargan
# or: fixest::feols(log_q ~ x | 0 | log_p ~ z)
```

**Assumptions & failure modes:** relevance (strong first stage — check F), exclusion (Z affects
Q only through P — untestable, argue it), monotonicity for LATE interpretations. Weak
instruments bias 2SLS toward OLS and wreck inference. The simulator lets you dial instrument
strength and watch exactly this happen.

## 4. Fix #2 — Control function

Two-step: regress price on instruments + controls, then include the *residual* in the demand
equation — the residual absorbs the endogenous part of price. Equivalent to 2SLS in linear
models but generalizes better to nonlinear demand (logit sessions!).

**Python**
```python
first = sm.OLS(log_p, sm.add_constant(np.c_[z, x])).fit()
v_hat = first.resid
cf = sm.OLS(log_q, sm.add_constant(np.c_[log_p, x, v_hat])).fit()
print(cf.params[1])             # ≈ -1.5; t-test on v_hat = endogeneity test
```

**R**
```r
v_hat <- resid(lm(log_p ~ z + x))
coef(lm(log_q ~ log_p + x + v_hat))["log_p"]
```

*Session-level version (Phase 4):* plug `v_hat` into the purchase logit
(`smf.logit("purchased ~ quoted_price + v_hat", ...)`) — the standard way to handle endogenous
prices in discrete choice (Petrin & Train 2010). Bootstrap the two steps for correct SEs.

## 5. Fix #3 — DML with instruments (partially linear IV)

When confounders are high-dimensional/nonlinear, combine ML nuisance estimation with IV
identification. Plain DML-PLR **does not** fix simultaneity — it only handles *observed*
confounding; you need the IV variant.

**Python**
```python
from doubleml import DoubleMLData, DoubleMLPLIV
from lightgbm import LGBMRegressor
data = DoubleMLData(df, y_col="log_q", d_cols="log_p", z_cols="z", x_cols=["x"])
pliv = DoubleMLPLIV(data, ml_l=LGBMRegressor(), ml_m=LGBMRegressor(),
                    ml_r=LGBMRegressor(), n_folds=5)
print(pliv.fit().summary)       # θ ≈ -1.5 with orthogonalized ML controls
```

**R**
```r
library(DoubleML); library(mlr3learners)
data <- DoubleMLData$new(df, y_col="log_q", d_cols="log_p", z_cols="z", x_cols="x")
pliv <- DoubleMLPLIV$new(data, ml_l=lrn("regr.ranger"),
                         ml_m=lrn("regr.ranger"), ml_r=lrn("regr.ranger"))
pliv$fit(); pliv$summary()
```

And the honest comparison the simulator automates: run `DoubleMLPLR` (no instrument) on the
same endogenous data and watch it return ≈ −0.9 — flexible ML does not substitute for an
identification strategy.

## 6. Fix #4 — Discontinuities in the pricing algorithm (RD)

If the endogenous rule has thresholds (surge rounding, repricing tiers), exploit them:
```python
from rdrobust import rdrobust
rdrobust(y=df["purchased"], x=df["latent_minus_threshold"], c=0)
```
```r
rdrobust::rdrobust(y = df$purchased, x = df$latent_minus_threshold, c = 0)
```
Local truth at the threshold, no instrument needed — but only local (see docs/01 §1).

## 7. Fix #5 — Just run the experiment

Randomized price cells sever `Cov(P, u)` by construction; every method above is a workaround
for not being able to randomize. The simulator's whole premise: quantify how much each
workaround costs in bias/variance relative to the experimental benchmark, under DGPs where we
control the truth.

## 8. Phase 4 implementation checklist

- [ ] `pricing.EndogenousRule(gamma, signal_noise)` — price = f(recent demand shock)
- [ ] `pricing.SurgeRule(thresholds, rounding)` — RD-ready algorithmic pricing
- [ ] Cost-shifter process emitted to the observable view (usable instrument)
- [ ] Instrument-strength dial (first-stage R²) and weak-IV stress tests
- [ ] Scorecard: {OLS, IV, CF, DML-PLR, DML-PLIV, RD, experiment} × endogeneity strength →
      bias curves. Expected picture: OLS bias grows linearly in γ; IV/CF/PLIV flat at 0 while
      instruments are strong; DML-PLR tracks OLS; RD flat but wide CIs.
