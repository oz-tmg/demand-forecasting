"""Phase 5 demo: the forecasting benchmark — training on sales vs demand.

Runs the rolling-origin benchmark on data/fact_sales_daily.csv and prints
the money table: censoring-blind models under-forecast true demand exactly
where censoring concentrates (promo-heavy, high-censoring series), look
calibrated against sales (the trap), and the stockout-aware treatments
close most of the gap to the oracle-lambda ceiling.

Run:  python examples/phase5_forecasting.py [--max-series N] [--origins K]
(full 100-series run takes a couple of minutes; --max-series 12 for a quick pass)
"""
import argparse
import pathlib
import sys
import warnings

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from demand_sim import ForecastBenchmarkConfig, run_forecast_benchmark  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--max-series", type=int, default=None)
ap.add_argument("--origins", type=int, default=1)
args = ap.parse_args()

data = pathlib.Path(__file__).resolve().parents[1] / "data" / "fact_sales_daily.csv"
panel = pd.read_csv(data, parse_dates=["date"])
if args.max_series:
    keys = (panel[["product_id", "store_id"]].drop_duplicates()
            .head(args.max_series))
    panel = panel.merge(keys, on=["product_id", "store_id"])

out = run_forecast_benchmark(panel, ForecastBenchmarkConfig(n_origins=args.origins))
sc = out["scorecard"]

n_series = panel.groupby(["product_id", "store_id"]).ngroups
print(f"series: {n_series}  origins: {args.origins}  horizon: 28d  "
      f"graded against ORACLE units_demanded\n")

for sl in ("all_series", "high_censoring"):
    show = sc[(sc.eval_target == "units_demanded") & (sc.series_slice == sl)]
    print(f"=== vs TRUE DEMAND [{sl}] ===")
    print(show.drop(columns=["eval_target", "series_slice"])
              .round(3).to_string(index=False), "\n")

blind = sc[(sc.model == "poisson_glm_blind") & (sc.series_slice == "high_censoring")]
vs_sales = float(blind[blind.eval_target == "units_sold"]["bias"].iloc[0])
vs_demand = float(blind[blind.eval_target == "units_demanded"]["bias"].iloc[0])
print("=== The trap ===")
print(f"censoring-blind GLM, high-censoring series: bias vs sales {vs_sales:+.2f} "
      f"(looks calibrated)\n"
      f"                                            bias vs demand {vs_demand:+.2f} "
      f"(the shortfall a real forecaster never sees)")
print("\nRead every rmse against the oracle_lambda row — that residual is pure "
      "Poisson noise,\nthe irreducible floor. The gap between blind and "
      "stockout-aware rows is the cost of\ntreating sales as demand; the oracle "
      "makes it measurable.")
