"""Stage 3 metric plugins. Each module exports ``compute(scenario, route) -> float``."""

from . import success, travel_time, hazard_exposure

REGISTRY = {
    "success": success.compute,
    "travel_time": travel_time.compute,
    "hazard_exposure": hazard_exposure.compute,
}

__all__ = ["REGISTRY"]
