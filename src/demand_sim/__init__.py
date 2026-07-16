"""demand_sim — ground-truth demand simulator (Phases 1-2)."""
from .config import SegmentConfig, SimulationConfig, default_config
from .demand import demand_curve, true_elasticity, consumer_surplus
from .simulate import run_simulation
from .metrics import score_run, cell_conversion, arc_elasticity, power_two_proportions
from .panel import PanelConfig, generate_panel, observable_view

__version__ = "0.2.0"
__all__ = [
    "SegmentConfig", "SimulationConfig", "default_config",
    "demand_curve", "true_elasticity", "consumer_surplus",
    "run_simulation", "score_run", "cell_conversion", "arc_elasticity",
    "power_two_proportions",
    "PanelConfig", "generate_panel", "observable_view",
]
