"""Export the datasets the analysis/ R Markdown walk-throughs consume.

Seed-reproducible (CLAUDE.md invariant). Observable and oracle content are
written to SEPARATE files so the .Rmds can be honest about which side of the
boundary each object sits on:

  observable (estimator inputs)          oracle (scoring-only)
  ------------------------------         --------------------------------
  fact_session.csv                       ground_truth_curves.csv
  surge_sessions.csv                     dim_segment_truth.csv
  endogenous_market.csv                  truth_params.csv
  fact_sales_daily.csv (Phase 2 CLI)

Usage:  python scripts/generate_analysis_datasets.py [--seed 42] [--outdir data]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace  # noqa: E402

from demand_sim import (EndogenousConfig, default_config, demand_curve,  # noqa: E402
                        run_simulation, run_surge_sessions,
                        simulate_endogenous_market, true_elasticity,
                        true_rd_jump)

RD_BASE_PRICE = 25.0
RD_CUT = 1.25
RD_STEP = 0.1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1 sessions: repeat visits + promos so mixed logit has panel
    # structure and a second attribute to price
    cfg = replace(default_config(), n_sessions=250_000, promo_prob=0.25,
                  sessions_per_consumer_mean=3.0, seed=args.seed)
    out = run_simulation(cfg)
    # gzipped: R's read.csv() decompresses .gz transparently
    out["sessions"].to_csv(outdir / "fact_session.csv.gz", index=False)

    # --- surge sessions for RD (matches rd.py test/demo configuration)
    surge = run_surge_sessions(cfg, n_sessions=400_000,
                               base_price=RD_BASE_PRICE, step=RD_STEP,
                               seed=3)
    surge.to_csv(outdir / "surge_sessions.csv.gz", index=False)

    # --- endogenous market (docs/05 DGP); u is oracle, so drop it
    endo_cfg = EndogenousConfig(n_periods=20_000, gamma=0.5, seed=7)
    endo = simulate_endogenous_market(endo_cfg)
    endo.drop(columns=["u"]).to_csv(outdir / "endogenous_market.csv",
                                    index=False)

    # --- ORACLE: ground-truth curves over a price grid (scoring-only)
    grid = np.linspace(5.0, 120.0, 116)
    curves = demand_curve(cfg, grid).merge(
        true_elasticity(cfg, grid), on="price")
    curves.to_csv(outdir / "ground_truth_curves.csv", index=False)

    # --- ORACLE: segment truth for mixed-logit validation (scoring-only)
    pd.DataFrame([{
        "segment": s.name, "share": s.share, "base_util": s.base_util,
        "price_coef": s.price_coef, "promo_uplift": s.promo_uplift,
    } for s in cfg.segments]).to_csv(outdir / "dim_segment_truth.csv",
                                     index=False)

    # --- ORACLE: scalar truths every .Rmd grades against (scoring-only)
    b_mix = float(np.average([s.price_coef for s in cfg.segments],
                             weights=[s.share for s in cfg.segments]))
    pd.DataFrame([
        {"dataset": "fact_session", "param": "price_coef_share_weighted",
         "value": b_mix},
        {"dataset": "endogenous_market", "param": "theta_true",
         "value": endo_cfg.theta},
        {"dataset": "endogenous_market", "param": "gamma", "value": endo_cfg.gamma},
        {"dataset": "endogenous_market", "param": "pi_instrument",
         "value": endo_cfg.pi},
        {"dataset": "surge_sessions", "param": "rd_cut", "value": RD_CUT},
        {"dataset": "surge_sessions", "param": "rd_step", "value": RD_STEP},
        {"dataset": "surge_sessions", "param": "base_price",
         "value": RD_BASE_PRICE},
        {"dataset": "surge_sessions", "param": "true_jump",
         "value": true_rd_jump(cfg, RD_BASE_PRICE, RD_CUT, RD_STEP)},
    ]).to_csv(outdir / "truth_params.csv", index=False)

    for f in ("fact_session.csv.gz", "surge_sessions.csv.gz",
              "endogenous_market.csv", "ground_truth_curves.csv",
              "dim_segment_truth.csv", "truth_params.csv"):
        print(f"{f}: {len(pd.read_csv(outdir / f)):,} rows")


if __name__ == "__main__":
    main()
