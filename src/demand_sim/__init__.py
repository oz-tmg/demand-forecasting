"""demand_sim — ground-truth demand simulator (Phase 1)."""
from .config import SegmentConfig, SimulationConfig, default_config
from .demand import demand_curve, true_elasticity, consumer_surplus
from .simulate import run_simulation
from .metrics import score_run, cell_conversion, arc_elasticity, power_two_proportions

__version__ = "0.1.0"
__all__ = [
    "SegmentConfig", "SimulationConfig", "default_config",
    "demand_curve", "true_elasticity", "consumer_surplus",
    "run_simulation", "score_run", "cell_conversion", "arc_elasticity",
    "power_two_proportions",
]
