"""demand_sim — ground-truth demand simulator (Phases 1-3)."""
from .config import SegmentConfig, SimulationConfig, default_config
from .demand import demand_curve, true_elasticity, consumer_surplus
from .simulate import run_simulation
from .metrics import score_run, cell_conversion, arc_elasticity, power_two_proportions
from .panel import PanelConfig, generate_panel, observable_view
from .experiments import (ExperimentConfig, run_experiment, estimate_contrast,
                          true_contrast)
from .guardrails import run_guardrails, srm_check, covariate_balance, \
    dual_exposure, aa_battery
from .power import monte_carlo_power, experiment_scorecard
from .rd import run_surge_sessions, rd_estimate, true_rd_jump

__version__ = "0.3.0"
__all__ = [
    "SegmentConfig", "SimulationConfig", "default_config",
    "demand_curve", "true_elasticity", "consumer_surplus",
    "run_simulation", "score_run", "cell_conversion", "arc_elasticity",
    "power_two_proportions",
    "PanelConfig", "generate_panel", "observable_view",
    "ExperimentConfig", "run_experiment", "estimate_contrast", "true_contrast",
    "run_guardrails", "srm_check", "covariate_balance", "dual_exposure",
    "aa_battery", "monte_carlo_power", "experiment_scorecard",
    "run_surge_sessions", "rd_estimate", "true_rd_jump",
]
