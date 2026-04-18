"""Total travel time in minutes for the route. NaN for failed runs so the
aggregator can exclude them (manuscript §3.6.1 B: computed only over
successful episodes)."""

from __future__ import annotations

import math

from ..schemas import Route, Scenario


def compute(scenario: Scenario, route: Route) -> float:
    if not route.success:
        return math.nan
    return float(sum(edge["travel_time"] for edge in route.per_edge))
