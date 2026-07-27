# Why Simulate Demand at All?

_On the one measurement you cannot make with real data._

---

## The argument in one paragraph

Bias requires truth. To say an elasticity estimate is off by 40% you need the number
it is off from, and in any real market that number does not exist anywhere — not in
the transaction log, not in a holdout set, not in a better-specified model. Every
method for learning a demand curve therefore rests on assumptions that are argued
rather than verified: that assignment was as-good-as-random, that the instrument is
excluded, that arms don't interfere, that sales are demand. A simulator inverts the
problem. You set the truth, generate what a firm would actually log, hand the
estimator only the observable view, and then *measure* — rather than debate — how far
off it lands and under exactly which conditions. That measurement is the product.
Everything else here is scaffolding for it.

---

## Forecast accuracy is not the same as being right about price

The usual defense of a demand model is holdout error, and it is the wrong test. A
model can predict sales beautifully conditional on the prices actually charged and be
badly wrong about what happens at a price never charged. Conditional expectation and
causal response coincide only when the price variation in the training data is
exogenous — which is the assumption under examination in the first place.

Retail pricing data is close to the worst case for this. Prices move because someone
expected demand to move: a category manager marks down slow weeks, marks up ahead of a
holiday, matches a competitor who is reacting to the same unobserved shock. Regress
log quantity on log price across that history and you recover a blend of the demand
curve and the pricing policy, weighted by everything the manager knew and the analyst
didn't. The estimate can be attenuated, inflated, or sign-flipped depending on the sign
and strength of that correlation, and the residual diagnostics will look fine
throughout.

You can hold out a week. You cannot hold out a counterfactual.

---

## Bias is a function, not a number

"OLS is biased under endogenous pricing" is a sentence from a textbook and worth
approximately nothing operationally. The useful object is the *curve*: bias plotted
against the strength of the mechanism producing it.

How much correlation between price and the unobserved demand shock does it take before
a naive elasticity crosses from attenuated-but-decision-useful into wrong-sign? At what
first-stage F does weak-instrument bias exceed the OLS bias the instrument was brought
in to fix? Where does a control function beat 2SLS, and where does its functional-form
assumption cost more than it saves? How much cross-arm interference does a
geo-randomized test tolerate before the treatment effect it reports is mostly cannibalization?

Every one of those is a numeric question with a numeric answer, and every one requires
a dial — endogeneity strength, instrument strength, interference radius, censoring rate
— that can be turned while the truth stays fixed and known. Turn, re-estimate, plot.
That plot is the thing no real dataset can produce, and it is the reason to build the
generator rather than borrow a dataset.

---

## Sales are not demand, and only an oracle can prove it

Once inventory exists, `units_sold = min(units_demanded, on_hand)`, and the censored
observations are emphatically not missing at random. They cluster on the days demand
ran highest, which for a promoted item means they cluster on precisely the days that
carry the identifying variation for price. Fit on censored sales and the price
coefficient attenuates *because* the low-price cells were truncated hardest — the very
observations meant to trace out the responsive part of the curve are the ones the
warehouse ate.

Then it compounds. The forecast under-predicts the fast movers, the replenishment
policy orders short against that forecast, the stockouts recur, and next quarter's
training data is more censored than this quarter's. The mechanism is self-reinforcing
and entirely invisible in the metrics, because the thing you would need to see —
lost sales — is unobservable by construction. No field measurement of it exists.

Keeping `units_demanded` and `lost_sales` in the oracle converts "censoring probably
biases us somewhat" into: this share of days out of stock, this share of true demand
lost, this much attenuation in the estimated elasticity, this much of it recovered by
a censored-regression correction and this much not. A forecaster graded against
uncensored true demand rather than against realized sales is being asked the question
that actually matters, and it is a question a real evaluation harness structurally
cannot ask.

---

## Heterogeneity is the phenomenon, not the noise

A finite mixture of segments does not aggregate to a constant-elasticity curve even
when every constituent segment is constant-elasticity. Pooled logit fitted to that
mixture returns a coefficient that belongs to no segment, and coincides with the
aggregate elasticity only near whatever price the mixture weights happen to favor.
Move away from that price and the pooled estimate degrades in a direction that depends
on the segment composition — which, on real data, you do not know.

This is not an academic tidiness point. Personalized pricing, segment promotions, and
targeted markdowns are decisions taken directly off segment-level elasticities. When
pooled estimates get used because segment estimates are noisy, that is a bias-variance
trade, and someone should be able to state the size of the bias being purchased.
With segment membership as an oracle field, the aggregation error is computable at
every price point instead of asserted.

---

## Experiment design has a price; know it before paying it

Randomization units are not free. Cluster at the user and one user's forty sessions
stop being forty observations; the effective-n collapse *is* the cost of the design,
and it is routinely discovered after launch, in the form of a confidence interval too
wide to act on. Switchbacks trade contamination for effective sample size along a curve
that depends on carryover length. Geo tests buy interference protection at brutal cost
in units.

An underpowered pricing experiment does not fail cleanly. It produces a noisy estimate
that someone acts on anyway, because the alternative is admitting the quarter's
traffic was spent on nothing. Monte Carlo power against a known DGP is dramatically
cheaper than learning the same lesson from live traffic, and it lets the design
question — unit, window length, cell count, minimum detectable effect — be settled with
numbers before anyone commits.

The same logic covers guardrails. SRM checks, covariate balance thresholds,
dual-exposure detection, and A/A batteries are alarms, and an alarm can only be
calibrated on data where you know whether there was a fire. Here you can inject the
fire.

---

## The honest objection

A simulator only tells you about the simulator. That is true and worth stating plainly
rather than defending against.

What it establishes is internal validity. If an estimator fails on a DGP that satisfies
its own assumptions, that is disqualifying and the result travels. If it succeeds, what
you have learned is that it works in a world built to be hospitable to it — a necessary
condition, not a sufficient one. Nobody should read a simulated elasticity as a
statement about a real market, and this project makes no such claim anywhere.

Two things follow. First, the DGP choice *is* the argument, so it has to be stated,
sourced, and exposed as parameters rather than tuned quietly until the preferred method
wins. Second, the durable claims are comparative and conditional — "method A degrades
faster than method B as interference grows, and here is the crossover" survives
translation to a real problem in a way that "the elasticity is −1.8" never will.

The failure mode to actively guard against is method selection that overfits the
synthetic world. The defense is to break the simulator's own assumptions on purpose —
misspecify the functional form, mix the segments, censor the panel, contaminate the
arms — and to report where each estimator dies rather than where it wins.

---

## What this actually buys

- A scorecard that reports bias, RMSE, and CI coverage for every estimator against
  known truth, instead of a point estimate and a shrug.
- A library of bias-versus-mechanism curves: the artifact that turns identification
  from a stance into a measurement.
- A design tool that prices randomization choices in effective n and detectable effect
  before traffic is spent.
- A forecasting panel with known-future covariates, controllable censoring, and
  uncensored true demand as the target — an evaluation set that grades a forecaster on
  demand rather than on what the warehouse happened to permit.

The deeper case is simply this: the mistakes that matter in demand estimation are the
quiet ones — the attenuation you never notice, the aggregation error that looks like a
plausible number, the censoring loop that shows up as a forecasting win. The only way
to find out whether you would catch them is to commit them deliberately, in a world
where checking is possible.
