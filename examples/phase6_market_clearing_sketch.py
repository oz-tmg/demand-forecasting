"""Phase 6 (DESIGN SPIKE) — recovering a demand system *through a market that clears*.

This is a self-contained sketch, not production demand_sim code. It exists to size
the Phase 6 build and prove the identification claim end-to-end on a small example:

    price is an equilibrium OUTCOME, not an assignable treatment, and you still
    recover the truth — but only with a shifter (an instrument), never from
    (price, quantity) pairs alone.

What it demonstrates
--------------------
1. A nested-logit demand system over differentiated goods (cards), grouped into
   nests by (quality band x role). The nesting parameter *is* the substitution
   structure; the reported object is a DIVERSION MATRIX, not a scalar elasticity.
2. A reduced-form market: an upward-sloping supply curve per good, cleared against
   nested-logit demand by tatonnement (a stand-in for the full order book — see the
   Phase 6 note). No agent-level book yet; that is the larger build this sizes.
3. The content calendar as instruments. A pack promo is a SUPPLY shock (injects
   stock, shifts the supply curve, traces demand). An SBC requirement is a DEMAND
   shock (raises utility / consumes the card, shifts demand, traces supply).
   Randomizing *which cards* each event touches turns them into instruments.
4. The payoff: naive Berry inversion by OLS is biased (simultaneity — the
   unobserved demand shock xi moves price and quantity together); 2SLS with the
   pack-promo instrument recovers alpha (price sensitivity) and lambda (nest
   substitution) near truth. The recovered diversion matrix is then graded against
   the oracle the way metrics.py grades elasticities today.

Observable vs. oracle split (the project's sacred boundary) is preserved:
the estimator reads only {price, quantity, shares, promo/SBC assignment, attrs};
xi_true, alpha, lambda, and the true diversion matrix are scoring-only.

Run:  python examples/phase6_market_clearing_sketch.py
Deps: numpy, scipy (statsmodels not required — 2SLS is done by hand).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration + ground truth (the oracle: known because we generated it)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketConfig:
    n_nests: int = 3                 # e.g. gold / silver / bronze x role, collapsed
    goods_per_nest: int = 4          # cards per nest
    n_weeks: int = 400               # time blocks (each yields one equilibrium/good)
    population: float = 2.0e5        # M potential buyers per block
    alpha: float = 0.9               # TRUE price sensitivity (utils per unit log-price)
    lam: float = 0.55                # TRUE nesting param in (0,1]; smaller = tighter nest
    beta_x: float = 1.0              # TRUE taste coefficient on the one attribute x
    supply_elasticity: float = 1.3   # phi: sellers list more when price is high
    xi_sd: float = 0.35              # sd of the UNOBSERVED demand shock (endogeneity source)
    omega_sd: float = 0.15           # sd of supply noise
    promo_boost: float = 0.8         # pack promo: +log stock on treated cards (instrument)
    sbc_boost: float = 0.7           # SBC: +utility on treated cards (instrument)
    treat_frac: float = 0.35         # share of cards an event touches each week
    seed: int = 42

    @property
    def n_goods(self) -> int:
        return self.n_nests * self.goods_per_nest


@dataclass
class Truth:
    cfg: MarketConfig
    nest_id: np.ndarray = field(init=False)   # (J,) nest membership
    x: np.ndarray = field(init=False)         # (J,) exogenous attribute
    base_stock: np.ndarray = field(init=False)  # (J,) baseline log-supply intercept

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        self.nest_id = np.repeat(np.arange(self.cfg.n_nests), self.cfg.goods_per_nest)
        self.x = rng.normal(0.0, 1.0, self.cfg.n_goods)
        self.base_stock = rng.normal(0.0, 0.4, self.cfg.n_goods)


# --------------------------------------------------------------------------- #
# Nested-logit demand system
# --------------------------------------------------------------------------- #
def nested_logit_shares(delta: np.ndarray, nest_id: np.ndarray, lam: float):
    """Return (s_j, s0, s_j_given_g) for a nested logit with an outside good (u0=0)."""
    s_j = np.zeros_like(delta)
    s_jg = np.zeros_like(delta)
    nests = np.unique(nest_id)
    Dg_lam = np.zeros(len(nests))            # D_g ** lam per nest
    within = {}
    for i, g in enumerate(nests):
        idx = nest_id == g
        e = np.exp(delta[idx] / lam)
        Dg = e.sum()
        within[g] = (idx, e, Dg)
        Dg_lam[i] = Dg ** lam
    denom = 1.0 + Dg_lam.sum()               # +1 for the outside good
    s0 = 1.0 / denom
    for i, g in enumerate(nests):
        idx, e, Dg = within[g]
        s_given_g = e / Dg                   # P(j | g)
        s_g = Dg_lam[i] / denom              # P(g)
        s_j[idx] = s_given_g * s_g
        s_jg[idx] = s_given_g
    return s_j, s0, s_jg


def diversion_matrix(delta: np.ndarray, nest_id: np.ndarray, lam: float,
                     alpha: float, price: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Numeric diversion matrix D[j,k] = fraction of demand j loses that goes to k
    when good j's price rises. Computed by finite-differencing the share function,
    so it works identically for the oracle params and the recovered params."""
    J = len(delta)
    base, _, _ = nested_logit_shares(delta, nest_id, lam)
    D = np.zeros((J, J))
    for j in range(J):
        d2 = delta.copy()
        d2[j] += -alpha * eps            # a small price bump on j -> delta_j falls
        bumped, _, _ = nested_logit_shares(d2, nest_id, lam)
        dloss = base[j] - bumped[j]      # demand j loses (>0)
        gains = bumped - base            # where it went
        gains[j] = 0.0
        if dloss > 0:
            D[j] = gains / dloss
    return D


# --------------------------------------------------------------------------- #
# Reduced-form market clearing (tatonnement stand-in for the order book)
# --------------------------------------------------------------------------- #
def clear(mean_util, alpha, log_stock, phi, omega, nest_id, lam, M,
          iters: int = 600, eta: float = 0.25):
    """Find log-prices where nested-logit demand == upward-sloping supply per good.

        demand_j(p)  = M * s_j(p)                       (nested logit)
        supply_j(p)  = exp(log_stock_j + phi*logp_j + omega_j)

    Tatonnement: raise a good's price when it is in excess demand. Converges for
    the smooth nested-logit demand used here; the real Phase 6 book replaces this
    with a matching engine and carryover across blocks (see the Phase 6 note).
    """
    logp = np.full_like(mean_util, 0.5)
    for _ in range(iters):
        delta = mean_util - alpha * logp
        s_j, _, _ = nested_logit_shares(delta, nest_id, lam)
        log_demand = np.log(np.maximum(M * s_j, 1e-9))
        log_supply = log_stock + phi * logp + omega
        excess = log_demand - log_supply          # >0 => demand exceeds supply
        logp = logp + eta * excess                # raise price on excess demand
    delta = mean_util - alpha * logp
    s_j, s0, s_jg = nested_logit_shares(delta, nest_id, lam)
    q = M * s_j
    return logp, q, s_j, s0, s_jg


# --------------------------------------------------------------------------- #
# Simulate a panel of weekly equilibria with randomized promo/SBC instruments
# --------------------------------------------------------------------------- #
def simulate(cfg: MarketConfig, truth: Truth):
    rng = np.random.default_rng(cfg.seed + 1)
    J, T = cfg.n_goods, cfg.n_weeks
    rows = []
    for t in range(T):
        xi = rng.normal(0.0, cfg.xi_sd, J)                 # UNOBSERVED demand shock
        omega = rng.normal(0.0, cfg.omega_sd, J)           # supply noise
        promo = (rng.random(J) < cfg.treat_frac).astype(float)   # pack: SUPPLY instrument
        sbc = (rng.random(J) < cfg.treat_frac).astype(float)     # SBC:  DEMAND instrument

        mean_util = cfg.beta_x * truth.x + xi + cfg.sbc_boost * sbc
        log_stock = truth.base_stock + cfg.promo_boost * promo

        logp, q, s_j, s0, s_jg = clear(
            mean_util, cfg.alpha, log_stock, cfg.supply_elasticity,
            omega, truth.nest_id, cfg.lam, cfg.population,
        )
        for j in range(J):
            rows.append(dict(
                week=t, good=j, nest=int(truth.nest_id[j]),
                logp=logp[j], q=q[j], s_j=s_j[j], s0=s0, s_jg=s_jg[j],
                x=truth.x[j], promo=promo[j], sbc=sbc[j],
                xi_true=xi[j],                                # ORACLE-only
            ))
    import pandas as pd
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Estimators: Berry inversion, OLS (biased) vs 2SLS (identified)
# --------------------------------------------------------------------------- #
def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _tsls(y, X, Z):
    """2SLS: X endogenous columns instrumented by Z (Z includes exogenous cols)."""
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)      # projection onto instrument space
    Xhat = Pz @ X
    beta = np.linalg.solve(Xhat.T @ X, Xhat.T @ y)
    return beta


def estimate(df, truth: Truth, cfg: MarketConfig):
    """Berry nested-logit inversion:
        ln s_j - ln s0 = b0 + beta*x_j - alpha*logp_j + (1-lam)*ln s_{j|g} + xi_j
    Endogenous: logp_j and ln s_{j|g}. Instruments: promo (excluded supply shifter),
    a BLP-style rival-count/rival-SBC term, plus exogenous x and constant.
    """
    y = np.log(df["s_j"].values) - np.log(df["s0"].values)
    const = np.ones(len(df))
    x = df["x"].values
    logp = df["logp"].values
    ln_sjg = np.log(df["s_jg"].values)

    # BLP-flavoured instrument: how many rivals in the same nest got an SBC this week
    df = df.copy()
    df["nest_sbc"] = df.groupby(["week", "nest"])["sbc"].transform("sum") - df["sbc"]
    nest_sbc = df["nest_sbc"].values
    promo = df["promo"].values

    # Design: [const, x, logp, ln_sjg]; recover alpha=-coef3, lam=1-coef4
    X = np.column_stack([const, x, logp, ln_sjg])
    Z = np.column_stack([const, x, promo, nest_sbc])   # excluded: promo, nest_sbc

    b_ols = _ols(y, X)
    b_iv = _tsls(y, X, Z)

    out = {
        "alpha_true": cfg.alpha, "lam_true": cfg.lam,
        "alpha_ols": -b_ols[2], "lam_ols": 1 - b_ols[3],
        "alpha_iv": -b_iv[2], "lam_iv": 1 - b_iv[3],
    }
    return out


def score_diversion(est, truth: Truth, cfg: MarketConfig, df):
    """Grade the recovered diversion matrix against the oracle at mean prices."""
    # representative mean utility & price (a single reference market)
    mean_util = cfg.beta_x * truth.x
    # oracle diversion
    logp_ref, *_ = clear(mean_util, cfg.alpha, truth.base_stock,
                          cfg.supply_elasticity, np.zeros(cfg.n_goods),
                          truth.nest_id, cfg.lam, cfg.population)
    delta_true = mean_util - cfg.alpha * logp_ref
    D_true = diversion_matrix(delta_true, truth.nest_id, cfg.lam, cfg.alpha, logp_ref)
    # estimated diversion (recovered alpha/lam; attributes observed)
    delta_hat = mean_util - est["alpha_iv"] * logp_ref
    D_hat = diversion_matrix(delta_hat, truth.nest_id, est["lam_iv"], est["alpha_iv"], logp_ref)
    # summarise: mean absolute error, and within- vs cross-nest mass
    off = ~np.eye(cfg.n_goods, dtype=bool)
    mae = np.abs(D_true[off] - D_hat[off]).mean()
    same = truth.nest_id[:, None] == truth.nest_id[None, :]
    within_true = D_true[off & same].sum() / D_true[off].sum()
    within_hat = D_hat[off & same].sum() / D_hat[off].sum()
    return dict(diversion_mae=mae, within_nest_share_true=within_true,
                within_nest_share_hat=within_hat)


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = MarketConfig()
    truth = Truth(cfg)
    df = simulate(cfg, truth)
    est = estimate(df, truth, cfg)
    div = score_diversion(est, truth, cfg, df)

    print("Phase 6 design spike — recovery through a clearing market")
    print(f"  goods={cfg.n_goods}  nests={cfg.n_nests}  weeks={cfg.n_weeks}  "
          f"equilibria={len(df):,}")
    print()
    print("  price sensitivity alpha   nest substitution lambda")
    print(f"    truth   {cfg.alpha:7.3f}                 {cfg.lam:7.3f}")
    print(f"    OLS     {est['alpha_ols']:7.3f}  (biased)       {est['lam_ols']:7.3f}  (biased)")
    print(f"    2SLS    {est['alpha_iv']:7.3f}  (promo IV)     {est['lam_iv']:7.3f}  (promo IV)")
    print()
    print(f"  OLS  alpha bias: {est['alpha_ols'] - cfg.alpha:+.3f}   "
          f"2SLS alpha bias: {est['alpha_iv'] - cfg.alpha:+.3f}")
    print(f"  diversion-matrix MAE (off-diagonal): {div['diversion_mae']:.4f}")
    print(f"  within-nest diversion share  truth={div['within_nest_share_true']:.3f}  "
          f"recovered={div['within_nest_share_hat']:.3f}")


if __name__ == "__main__":
    main()
