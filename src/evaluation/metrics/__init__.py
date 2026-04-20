"""Stage 3 metric plugins. Each module exports ``compute(scenario, route) -> float``."""

from . import distance, hazard_exposure, hazard_score, runtime, steps, success, travel_time

REGISTRY = {
    "success": success.compute,
    "travel_time": travel_time.compute,
    "hazard_exposure": hazard_exposure.compute,
    "hazard_score": hazard_score.compute,
    "steps": steps.compute,
    "distance": distance.compute,
    "runtime": runtime.compute,
}

__all__ = ["REGISTRY"]
