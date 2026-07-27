# 09 — The Log-Log Demand Model: Cobb-Douglas Structure, Assumptions, and Why PPML + FE

_Companion note for `01_log_log_ols.Rmd`. Part of the `demand-sim` docs set._

---

## 1. The model

$$
\log Q_{it} \;=\; \alpha_i \;+\; \theta \log P_{it} \;+\; x_{it}'\beta \;+\; \varepsilon_{it}
$$

where $i$ indexes a product-store series, $t$ a day, and $x_{it}$ carries trend,
day-of-week, annual seasonality, and promo terms.

$\theta$ is the price elasticity **by construction** — that is the entire appeal of
the specification. Everything in this document is the price paid for that
convenience.

---

## 2. Equivalence to a Cobb-Douglas function

Exponentiate and the log-linear form is a power law:

$$
Q_{it} \;=\; A_i \, P_{it}^{\theta} \prod_k Z_{kt}^{\beta_k} e^{\varepsilon_{it}},
\qquad A_i = e^{\alpha_i}
$$

which is Cobb-Douglas $Y = A L^{\alpha} K^{\beta}$ with price and covariates as the
inputs. Three properties carry over intact.

**Coefficients are elasticities, constant over the whole domain.**
The curve is a straight line in log space, forever. This is not a local
approximation near a reference price — it is a global functional restriction.

**Multiplicative separability.**
Covariates shift the *level* of demand and never the *slope*. Promo, trend, and
seasonality scale demand without touching price responsiveness. This is precisely
why $\theta$ in `arrivals.py` equals the elasticity of $E[Q \mid p]$: the seasonal
and promo terms factor out of the derivative. It stops holding the moment anything
interacts with price.

**The unit-specific $A_i$ is the TFP term.**
The `series` fixed effect *is* the Cobb-Douglas $A$ — a multiplicative scale factor
for that product-store, which becomes additive once logged.

### A distinction worth keeping straight

This is the same functional *family* as a Cobb-Douglas production function. It is
**not** a demand system derived from Cobb-Douglas *utility*. Cobb-Douglas
preferences imply constant expenditure shares, hence $\theta = -1$ exactly and zero
cross-price elasticities — a far stronger and generally false claim. The log-log
regression borrows the algebra, not the microfoundation.

### Two pathologies that matter for a pricing project

**No interior optimum unless $\theta < -1$.**
Revenue is $P \cdot Q \propto P^{1+\theta}$. With constant marginal cost $c$, the
monopoly price is

$$
p^{\ast} = c \cdot \frac{\theta}{1+\theta}
$$

which exists and is positive only when $\theta < -1$. Fit $\hat\theta = -0.8$ and
the estimated demand curve instructs you to **raise price without bound**. An
inelastic point estimate is not a small problem; it is a degenerate optimisation.

**Consumer surplus diverges unless $\theta < -1$.**
There is no choke price — $Q > 0$ at every finite $P$ — so

$$
CS(p^{\ast}) = \int_{p^{\ast}}^{\infty} A p^{\theta} \, dp
$$

is infinite for $\theta \ge -1$. Given the surplus shading in
`01_demand_curves.Rmd`, check the sign condition before integrating, or cap the
upper limit of integration and state the cap explicitly.

---

## 3. Assumptions

| # | Assumption | What breaks it in this simulator |
|---|---|---|
| 1 | Constant elasticity, globally | Segment mixture — aggregate demand bends in log space |
| 2 | Multiplicative separability | Any price × promo or price × season interaction |
| 3 | $Q > 0$ on every row | Stockout zeros, dropped non-randomly by `log()` |
| 4 | $E[\varepsilon \mid X] = 0$ **in logs** | Retransformation: $E[Q \mid X] = e^{x'\beta}\,E[e^{\varepsilon} \mid X]$ |
| 5 | Homoskedastic $\varepsilon$ | Count-like variance that scales with the mean |
| 6 | Strict exogeneity of price | Inventory-driven repricing; feedback from past demand shocks |
| 7 | $\alpha_i$ may correlate with $P$ — only under FE | RE assumes it does not (see §5) |

Assumptions 4 and 5 are the load-bearing pair.

If $\operatorname{Var}(\varepsilon \mid X)$ depends on $X$, then
$E[e^{\varepsilon} \mid X]$ depends on $X$ too. That dependence is absorbed into the
slope, and $\hat\theta$ is **inconsistent** — not merely inefficient. Duan smearing
repairs predicted *levels*; it does nothing for the *slope*.

Assumption 3 deserves separate emphasis in this project. `log(units_sold)` is
$-\infty$ at zero, so `lm` silently deletes those rows. Under the (s, S) policy the
zeros are not random: they cluster on stockouts, which cluster on promos, which are
the low-price end of the grid. The deletion is therefore correlated with the
regressor — a selection bias stacked on top of the censoring bias, and a distinct
mechanism from the one the Phase 2 export demonstrates.

---

## 4. Why Poisson (PPML) is better suited

`fepois` estimates the conditional mean directly,

$$
E[Q_{it} \mid X] = \exp\!\left(\alpha_i + \theta \log P_{it} + x_{it}'\beta\right),
$$

which preserves the entire Cobb-Douglas interpretation — coefficients are still
elasticities — while dropping the parts that fail.

**No retransformation.** The multiplicative mean is modelled directly, rather than a
log-scale mean that must be mapped back through an unknown $E[e^{\varepsilon}\mid X]$.

**Zeros are in the likelihood.** Stockout rows stay in the sample, closing the
selection channel from assumption 3.

**Consistency requires only the conditional mean.** Poisson is a linear-exponential
family, so quasi-ML is consistent under *any* true variance function
(Gourieroux, Monfort & Trognon 1984). The Poisson variance assumption buys
efficiency, not validity. Cluster the standard errors and overdispersion is handled.
Nothing requires the outcome to be an integer — PPML is perfectly well-defined on
continuous nonnegative outcomes such as revenue.

**Silva & Tenreyro (2006) is exactly assumption 5 failing.** Under
heteroskedasticity, log-log OLS estimates the wrong parameter and PPML does not.
This is the "log of gravity" result, and it transfers directly to demand panels.

**No incidental-parameters problem with fixed effects.** Poisson FE is consistent
with fixed $T$ as $N \to \infty$ (Wooldridge 1999), unlike FE logit or probit. The
`series` effect can therefore be absorbed freely.

### The honest cost

PPML weights observations by their fitted mean, so high-volume series dominate the
estimate. If elasticity is heterogeneous across series, PPML's pooled $\hat\theta$
is a volume-weighted average while OLS's is closer to unweighted. That is a
difference in **estimand**, not merely in estimator — and it is one the oracle can
adjudicate directly.

---

## 5. Why fixed effects, not random effects

Random effects requires

$$
E\!\left[\alpha_i \mid P_{i1}, \dots, P_{iT}\right] = 0
$$

— latent series-level demand uncorrelated with price. In a pricing panel this is
essentially never true: popular products are priced higher, and stores with strong
latent demand run shallower discounts. **That correlation is exactly the endogeneity
the `gamma` dial creates.** RE leaves it in the error term; the resulting bias is
the mechanism under study, not a nuisance to be assumed away.

**FE identifies from within-series variation only.** RE is a precision-weighted
blend of within and between variation, and the between component is precisely the
contaminated part. RE is more efficient at estimating something contaminated.

**RE's advantages do not apply here.** Within-variation is abundant by construction
(randomized cells, promo schedule), and the time-invariant series characteristics
that RE could identify are nuisance parameters, not targets. RE also imposes a
distributional assumption on $\alpha_i$ purchased for efficiency the design does not
need.

**The Hausman test can be skipped.** In a simulator, sweep $\gamma$ and plot the
RE–FE gap against it directly. That is a stronger artifact than a test statistic,
and it matches the project's house style of bias-as-a-function-of-mechanism-strength.

**A defensible middle ground exists.** Mundlak / Chamberlain correlated random
effects — adding series means of $\log P$ as regressors — reproduces the FE slope
while retaining time-invariant terms. Worth a scorecard row as the "RE done
defensibly" comparison.

### FE is not a cure-all

Fixed effects require **strict** exogeneity. If price at $t$ responds to demand
shocks at $t-1$ — which inventory-driven repricing does by design — the
within-transformation correlates the demeaned regressor with the demeaned error, and
FE is biased too. That residual gap is what IV, control function, and DML exist to
close. The Phase 4 scorecard should therefore carry a `fepois + FE` row that is
*still visibly biased*, demonstrating that fixed effects buy one identification
problem's worth of progress rather than all of it.

---

## 6. Summary table

| Concern | Log-log OLS | PPML |
|---|---|---|
| Coefficient is an elasticity | Yes | Yes |
| Handles $Q = 0$ | No — rows deleted, non-randomly | Yes |
| Needs homoskedastic errors for consistency | Yes | No |
| Retransformation bias in predicted levels | Yes (Duan smearing required) | None |
| Consistency conditions | Full log-scale mean **and** variance | Conditional mean only |
| FE with fixed $T$ | Consistent | Consistent |
| Implicit weighting | Roughly unweighted | Volume-weighted |
| Estimand under heterogeneous $\theta$ | Closer to unweighted mean | Volume-weighted mean |

---

## 7. Key references

- Silva, J.M.C. Santos & Tenreyro, S. (2006). *The Log of Gravity.* Review of
  Economics and Statistics, 88(4), 641–658.
- Gourieroux, C., Monfort, A. & Trognon, A. (1984). *Pseudo Maximum Likelihood
  Methods: Theory.* Econometrica, 52(3), 681–700.
- Wooldridge, J.M. (1999). *Distribution-Free Estimation of Some Nonlinear Panel
  Data Models.* Journal of Econometrics, 90(1), 77–97.
- Duan, N. (1983). *Smearing Estimate: A Nonparametric Retransformation Method.*
  JASA, 78(383), 605–610.
- Mundlak, Y. (1978). *On the Pooling of Time Series and Cross Section Data.*
  Econometrica, 46(1), 69–85.
- Chamberlain, G. (1982). *Multivariate Regression Models for Panel Data.*
  Journal of Econometrics, 18(1), 5–46.

---

_All elasticities discussed here are generated by a chosen DGP. No number in this
project describes any real market._
