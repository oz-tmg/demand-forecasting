"""CLI entry point for the Phase 2 forecasting dataset generator.

Usage:
    python scripts/generate_phase2_dataset.py [--seed 42] [--outdir data]

Writes fact_sales_daily.csv (oracle view — includes ground truth),
dim_product_store.csv, and promo_calendar.csv. Use
demand_sim.observable_view() to strip oracle columns before handing the
panel to a forecast model.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from demand_sim import PanelConfig, generate_panel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = generate_panel(PanelConfig(seed=args.seed))
    panel = out["panel_oracle"]

    panel.to_csv(outdir / "fact_sales_daily.csv", index=False)
    out["dim_product_store"].to_csv(outdir / "dim_product_store.csv", index=False)
    out["promo_calendar"].to_csv(outdir / "promo_calendar.csv", index=False)

    n_series = panel.groupby(["product_id", "store_id"]).ngroups
    lost = panel.lost_sales.sum()
    print(f"rows: {len(panel):,}  series: {n_series}")
    print(f"stockout-day rate: {panel.stockout.mean():.3%}")
    print(f"censored (lost) units: {lost:,} "
          f"({lost / panel.units_demanded.sum():.2%} of true demand)")

    assert (panel.units_sold <= panel.units_demanded).all()
    assert (panel.inventory_start >= 0).all()
    assert (panel.units_sold <= panel.inventory_start).all()
    print("integrity checks passed: sold <= demanded, sold <= on-hand, stock >= 0")


if __name__ == "__main__":
    main()
