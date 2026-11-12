"""Phase 6: recovering a demand system *through a market that clears* (docs/10).

Everything through Phase 5 recovers a parameter from a **posted price the studio
set**. Here price is an **equilibrium outcome** — sellers and buyers meet and a
price clears — and the estimand must be recovered through it. That is the harder,
rarely-demonstrated problem, and it is where the interesting failure modes live.

Structure
---------
- Demand is a **nested logit** over differentiated goods grouped into nests by
  (quality band x role). The nesting parameter `lam` *is* the substitution
  structure; the reported object is a **diversion matrix**, not a scalar.
- Price forms by **reduced-form clearing**: an upward-sloping supply curve per
  good, cleared against nested-logit demand by tatonnement. This is a stand-in for
  the full order book (asks/bids/expiries/matching + carryover), which is the
  larger Phase 6 build this module is the first stage of.
- Identification comes from the **content calendar**. A pack promo is a SUPPLY
  shifter (injects stock, traces demand); an SBC requirement is a DEMAND shifter
  (raises utility / consumes the card, traces supply). Randomizing *which goods*
  each event touches turns them into instruments.

The payoff (docs/10 §4): naive Berry inversion by OLS is biased — the unobserved
demand shock `xi` moves price and quantity together (classic simultaneity), and
it can even recover the wrong *sign* on price. 2SLS with the promo instrument
recovers `alpha` (price sensitivity) and `lam` (nest substitution) near truth, and
the recovered diversion matrix is graded against the oracle.

Observable vs. oracle boundary (sacred): estimators read only
{logp, q, s_j, s0, s_jg, x, promo, sbc}. `xi_true` is scoring-only.

Standard errors are **clustered by week** — one week's goods trade against a shared
denominator (interference), so they are not independent (CLAUDE.md, docs/04).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Truth that must never enter an estimator's feature set.
ORACLE_COLS = ["xi_true"]

# Columns an estimator is permitted to read.
OBSERVABLE_COLS = ["week", "good", "nest", "logp", "q", "s_j", "s0", "s_jg",
                   "x", "promo", "sbc"]


# --------------------------------------------------------------------------- #
# Configuration + ground truth (the oracle — known because we generate it)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketConfig:
    """docs/10 clearing-market DGP with dials. Truth: `alpha`, `lam`, `beta_x`."""

    n_nests: int = 3                 # quality x role bands, collapsed
    goods_per_nest: int = 4          # cards per nest
    n_weeks: int = 400               # time blocks; each yields one equilibrium/good
    population: float = 2.0e5        # M potential buyers per block
    alpha: float = 0.9               # TRUE price sensitivity (oracle)
    lam: float = 0.55                # TRUE nest param in (0,1]; smaller = tighter nest
    beta_x: float = 1.0              # TRUE taste coefficient on attribute x
    supply_elasticity: float = 1.3   # phi: sellers list more when price is high
    xi_sd: float = 0.35              # sd of the UNOBSERVED demand shock (endogeneity)
    omega_sd: float = 0.15           # sd of supply noise
    promo_boost: float = 0.8         # pack promo: +log stock on treated cards (instrument)
    sbc_boost: float = 0.7           # SBC: +utility on treated cards (observed demand shifter)
    treat_frac: float = 0.35         # share of cards an event touches each week
    seed: int = 42

    @property
    def n_goods(self) -> int:
        return self.n_nests * self.goods_per_nest


def _attributes(cfg: MarketConfig):
    """Fixed card attributes (nest membership, exogenous x, baseline log-stock)."""
    rng = np.random.default_rng(cfg.seed)
    nest_id = np.repeat(np.arange(cfg.n_nests), cfg.goods_per_nest)
    x = rng.normal(0.0, 1.0, cfg.n_goods)
    base_stock = rng.normal(0.0, 0.4, cfg.n_goods)
    return nest_id, x, base_stock


# --------------------------------------------------------------------------- #
# Nested-logit demand system
# --------------------------------------------------------------------------- #
def nested_logit_shares(delta: np.ndarray, nest_id: np.ndarray, lam: float):
    """Return (s_j, s0, s_j_given_g) for a nested logit with an outside good (u0=0)."""
    s_j = np.zeros_like(delta, dtype=float)
    s_jg = np.zeros_like(delta, dtype=float)
    nests = np.unique(nest_id)
    Dg_lam = np.zeros(len(nests))
    cache = []
    for g in nests:
        idx = nest_id == g
        e = np.exp(delta[idx] / lam)
        Dg = e.sum()
        cache.append((idx, e, Dg))
    for i, (idx, e, Dg) in enumerate(cache):
        Dg_lam[i] = Dg ** lam
    denom = 1.0 + Dg_lam.sum()               # +1 for the outside good
    s0 = 1.0 / denom
    for i, (idx, e, Dg) in enumerate(cache):
        s_given_g = e / Dg                   # P(j | g)
        s_g = Dg_lam[i] / denom              # P(g)
        s_j[idx] = s_given_g * s_g
        s_jg[idx] = s_given_g
    return s_j, float(s0), s_jg


def draw_nested_logit_utilities(delta: np.ndarray, nest_id: np.ndarray, lam: float,
                                n: int, rng: np.random.Generator):
    """Draw `n` individual random-utility vectors from the nested logit that
    `nested_logit_shares` describes, so agent-level simulators can give each agent
    their own utilities (market-sim task 2.2).

        u_ij = delta_j + eps_ij,  u_i0 = eps_i0   (outside good, delta_0 = 0)

    eps follows the nested-logit GEV distribution: standard Gumbel marginals, independent
    across nests and from the outside good, and correlated within a nest with
    dissimilarity `lam`. Construction (positive-stable / Gumbel-copula representation):

        eps_ij = lam * g_ij + lam * log V_ig,   g_ij ~ Gumbel(0, 1),  V_ig ~ PS(lam)

    where PS(lam) is positive stable with Laplace transform exp(-s**lam), drawn with
    Kanter's (1975) formula. Then P(eps_g <= x) = exp(-(sum_j exp(-x_j/lam))**lam),
    the nest's GEV term. The argmax over [u_i0, u_i1..J] reproduces
    `nested_logit_shares` (tests/test_phase6.py checks the choice frequencies).

    Returns (U, u0): U is n x J, u0 has length n.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    delta = np.asarray(delta, dtype=float)
    nest_id = np.asarray(nest_id)
    nests, inv = np.unique(nest_id, return_inverse=True)
    g = rng.gumbel(0.0, 1.0, (n, len(delta)))
    if lam == 1.0:
        log_v = np.zeros((n, len(nests)))                  # independent logit errors
    else:
        u = rng.uniform(0.0, 1.0, (n, len(nests))) * np.pi
        w = rng.exponential(1.0, (n, len(nests)))
        # Kanter: V = sin(lam u)/sin(u)^(1/lam) * (sin((1-lam) u)/w)^((1-lam)/lam)
        log_v = (np.log(np.sin(lam * u)) - np.log(np.sin(u)) / lam
                 + (1.0 - lam) / lam * (np.log(np.sin((1.0 - lam) * u)) - np.log(w)))
    eps = lam * g + lam * log_v[:, inv]
    u0 = rng.gumbel(0.0, 1.0, n)
    return delta + eps, u0


def diversion_matrix(delta: np.ndarray, nest_id: np.ndarray, lam: float,
                     alpha: float, eps: float = 1e-4) -> np.ndarray:
    """Numeric diversion matrix D[j,k]: of the demand good j loses when its price
    rises, the fraction that goes to k. Finite-differenced on the share function,
    so it works identically for oracle params and recovered params."""
    base, _, _ = nested_logit_shares(delta, nest_id, lam)
    J = len(delta)
    D = np.zeros((J, J))
    for j in range(J):
        d2 = delta.copy()
        d2[j] += -alpha * eps                # a small price bump on j lowers delta_j
        bumped, _, _ = nested_logit_shares(d2, nest_id, lam)
        dloss = base[j] - bumped[j]          # demand j loses (>0)
        gains = bumped - base
        gains[j] = 0.0
        if dloss > 0:
            D[j] = gains / dloss
    return D


# --------------------------------------------------------------------------- #
# Reduced-form market clearing (tatonnement stand-in for the order book)
# --------------------------------------------------------------------------- #
def clear_market(mean_util, alpha, log_stock, phi, omega, nest_id, lam, M,
                 iters: int = 600, eta: float = 0.25):
    """Find log-prices where nested-logit demand == upward-sloping supply per good.

        demand_j(p) = M * s_j(p)                              (nested logit)
        supply_j(p) = exp(log_stock_j + phi*logp_j + omega_j)

    Tatonnement: raise a good's price when it is in excess demand. Converges for
    the smooth nested-logit demand used here; the full order book replaces this
    with a matching engine and carryover across blocks (docs/10 §5).
    """
    logp = np.full_like(mean_util, 0.5, dtype=float)
    for _ in range(iters):
        delta = mean_util - alpha * logp
        s_j, _, _ = nested_logit_shares(delta, nest_id, lam)
        log_demand = np.log(np.maximum(M * s_j, 1e-9))
        log_supply = log_stock + phi * logp + omega
        logp = logp + eta * (log_demand - log_supply)   # up on excess demand
    delta = mean_util - alpha * logp
    s_j, s0, s_jg = nested_logit_shares(delta, nest_id, lam)
    return logp, M * s_j, s_j, s0, s_jg


# --------------------------------------------------------------------------- #
# Simulate a panel of weekly equilibria with randomized promo/SBC instruments
# --------------------------------------------------------------------------- #
def simulate_market(cfg: MarketConfig) -> pd.DataFrame:
    """Weekly equilibria. Observable: OBSERVABLE_COLS. Oracle: xi_true."""
    nest_id, x, base_stock = _attributes(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    J, T = cfg.n_goods, cfg.n_weeks
    rows = []
    for t in range(T):
        xi = rng.normal(0.0, cfg.xi_sd, J)                       # UNOBSERVED demand shock
        omega = rng.normal(0.0, cfg.omega_sd, J)                 # supply noise
        promo = (rng.random(J) < cfg.treat_frac).astype(float)   # pack: SUPPLY instrument
        sbc = (rng.random(J) < cfg.treat_frac).astype(float)     # SBC:  observed demand shifter

        mean_util = cfg.beta_x * x + xi + cfg.sbc_boost * sbc
        log_stock = base_stock + cfg.promo_boost * promo

        logp, q, s_j, s0, s_jg = clear_market(
            mean_util, cfg.alpha, log_stock, cfg.supply_elasticity,
            omega, nest_id, cfg.lam, cfg.population)

        for j in range(J):
            rows.append(dict(week=t, good=j, nest=int(nest_id[j]),
                             logp=logp[j], q=q[j], s_j=s_j[j], s0=s0,
                             s_jg=s_jg[j], x=x[j], promo=promo[j], sbc=sbc[j],
                             xi_true=xi[j]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Berry inversion — OLS (biased) vs 2SLS (identified), clustered by week
# --------------------------------------------------------------------------- #
def _berry_design(df: pd.DataFrame):
    """Assemble the Berry nested-logit inversion arrays.

        y = ln s_j - ln s0
          = b0 + beta*x + sbc_coef*sbc - alpha*logp + (1-lam)*ln s_{j|g} + xi

    Endogenous: logp, ln s_{j|g}. Excluded instruments: own promo (supply
    shifter), and rival promo / rival SBC within the nest-week (BLP-style).
    """
    d = df.copy()
    d["nest_promo"] = d.groupby(["week", "nest"])["promo"].transform("sum") - d["promo"]
    d["nest_sbc"] = d.groupby(["week", "nest"])["sbc"].transform("sum") - d["sbc"]
    y = (np.log(d["s_j"]) - np.log(d["s0"])).to_numpy()
    const = np.ones(len(d))
    # X: exogenous [const, x, sbc] + endogenous [logp, ln s_jg]
    X = np.column_stack([const, d["x"], d["sbc"], d["logp"], np.log(d["s_jg"])])
    # Z: exogenous [const, x, sbc] + excluded [promo, nest_promo, nest_sbc]
    Z = np.column_stack([const, d["x"], d["sbc"],
                         d["promo"], d["nest_promo"], d["nest_sbc"]])
    week = d["week"].to_numpy()
    return y, X, Z, week


def _cluster_vcov(bread, resid, regmat, groups):
    """Cluster-robust sandwich: bread @ (sum_g M_g' e e' M_g) @ bread."""
    k = regmat.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        m = groups == g
        Mg = regmat[m]
        eg = resid[m]
        s = Mg.T @ eg
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    n = len(resid)
    adj = (G / (G - 1)) * ((n - 1) / (n - k))       # standard small-sample factor
    return adj * bread @ meat @ bread


def _params(beta):
    """Map Berry coefficients -> structural (alpha, lam). Order: const,x,sbc,logp,ln_sjg."""
    return {"alpha": float(-beta[3]), "lam": float(1.0 - beta[4])}


def fit_berry_ols(df: pd.DataFrame) -> dict:
    """OLS Berry inversion — biased whenever the demand shock moves price (always here)."""
    y, X, _, week = _berry_design(df)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    V = _cluster_vcov(XtX_inv, resid, X, week)
    se = np.sqrt(np.diag(V))
    out = {"estimator": "ols", **_params(beta),
           "alpha_se": float(se[3]), "lam_se": float(se[4])}
    return out


def fit_berry_2sls(df: pd.DataFrame) -> dict:
    """2SLS Berry inversion — promo/rival instruments identify alpha and lam.

    Reports a first-stage F for the price equation on the excluded instruments as
    a weak-instrument guard (docs/05 convention: F < 10 => weak)."""
    y, X, Z, week = _berry_design(df)
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    Xhat = Pz @ X
    XhX_inv = np.linalg.inv(Xhat.T @ X)
    beta = XhX_inv @ (Xhat.T @ y)
    resid = y - X @ beta                              # structural residual (uses X, not Xhat)
    # 2SLS == OLS of y on Xhat; cluster-robust sandwich with that bread:
    bread = np.linalg.inv(Xhat.T @ Xhat)
    V = _cluster_vcov(bread, resid, Xhat, week)
    se = np.sqrt(np.diag(V))

    # first-stage F: logp ~ excluded instruments, partialling out exogenous cols
    exog = Z[:, :3]                                   # const, x, sbc
    excl = Z[:, 3:]                                   # promo, nest_promo, nest_sbc
    p = X[:, 3]
    def _resid_on(A, b):
        return b - A @ np.linalg.solve(A.T @ A, A.T @ b)
    p_perp = _resid_on(exog, p)
    excl_perp = np.column_stack([_resid_on(exog, excl[:, c]) for c in range(excl.shape[1])])
    gamma = np.linalg.solve(excl_perp.T @ excl_perp, excl_perp.T @ p_perp)
    fitted = excl_perp @ gamma
    ss_res = ((p_perp - fitted) ** 2).sum()
    n, m = len(p), excl.shape[1]
    f_stat = float((fitted @ fitted) / m / (ss_res / (n - 3 - m)))

    out = {"estimator": "2sls", **_params(beta),
           "alpha_se": float(se[3]), "lam_se": float(se[4]),
           "first_stage_F": f_stat, "weak_instrument": f_stat < 10.0}
    return out


# --------------------------------------------------------------------------- #
# Scoring against the oracle
# --------------------------------------------------------------------------- #
def reference_market(cfg: MarketConfig) -> pd.DataFrame:
    """ORACLE reference market: mean utility + the clearing log-price *at truth*.

    Scoring-only. The diversion matrix is a property of the demand system at a
    price point; we evaluate both the true and the recovered systems at this one
    common set of prices, so the comparison isolates demand-parameter error rather
    than mixing in an equilibrium re-solve. `mean_util` and `logp_ref` are built
    from truth, so this belongs on the oracle side of the boundary.
    """
    nest_id, x, base_stock = _attributes(cfg)
    mean_util = cfg.beta_x * x
    logp, *_ = clear_market(mean_util, cfg.alpha, base_stock, cfg.supply_elasticity,
                            np.zeros(cfg.n_goods), nest_id, cfg.lam, cfg.population)
    return pd.DataFrame({"good": np.arange(cfg.n_goods), "nest": nest_id,
                         "x": x, "mean_util": mean_util, "logp_ref": logp})


def oracle_diversion(cfg: MarketConfig) -> np.ndarray:
    """The true diversion matrix at the reference prices (scoring-only)."""
    ref = reference_market(cfg)
    mu, lp, nest_id = (ref["mean_util"].to_numpy(), ref["logp_ref"].to_numpy(),
                       ref["nest"].to_numpy())
    return diversion_matrix(mu - cfg.alpha * lp, nest_id, cfg.lam, cfg.alpha)


def score_diversion(cfg: MarketConfig, est: dict) -> dict:
    """Grade the recovered diversion matrix against the oracle at common prices."""
    ref = reference_market(cfg)
    mu, lp, nest_id = (ref["mean_util"].to_numpy(), ref["logp_ref"].to_numpy(),
                       ref["nest"].to_numpy())
    D_true = diversion_matrix(mu - cfg.alpha * lp, nest_id, cfg.lam, cfg.alpha)
    D_hat = diversion_matrix(mu - est["alpha"] * lp, nest_id, est["lam"], est["alpha"])
    off = ~np.eye(cfg.n_goods, dtype=bool)
    same = nest_id[:, None] == nest_id[None, :]
    return {"diversion_mae": float(np.abs(D_true[off] - D_hat[off]).mean()),
            "within_nest_share_true": float(D_true[off & same].sum() / D_true[off].sum()),
            "within_nest_share_hat": float(D_hat[off & same].sum() / D_hat[off].sum())}


def market_scorecard(cfg: MarketConfig, n_reps: int = 40) -> pd.DataFrame:
    """Monte Carlo over seeds: OLS vs 2SLS bias and 95% CI coverage for alpha
    (docs/10 §4). This is the Phase 6 analogue of `endogeneity_scorecard`."""
    from dataclasses import replace
    rows = []
    for r in range(n_reps):
        df = simulate_market(replace(cfg, seed=cfg.seed + r))
        for fit in (fit_berry_ols, fit_berry_2sls):
            e = fit(df)
            lo, hi = e["alpha"] - 1.96 * e["alpha_se"], e["alpha"] + 1.96 * e["alpha_se"]
            rows.append({"estimator": e["estimator"], "alpha_hat": e["alpha"],
                         "bias": e["alpha"] - cfg.alpha,
                         "covers": bool(lo <= cfg.alpha <= hi)})
    df = pd.DataFrame(rows)
    return (df.groupby("estimator")
              .agg(mean_alpha=("alpha_hat", "mean"),
                   mean_bias=("bias", "mean"),
                   rmse=("bias", lambda s: float(np.sqrt((s ** 2).mean()))),
                   coverage_95=("covers", "mean"))
              .reset_index())
