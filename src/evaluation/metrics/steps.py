"""Number of edge traversals (steps) in the executed route.

Counts every edge walked, including those added by NNA replans. Returns
``NaN`` on failure so the aggregator can exclude incomplete episodes from
per-RI means (same convention as ``travel_time`` and ``hazard_exposure``).
"""

from __future__ import annotations

import math

from ..schemas import Route, Scenario


def compute(scenario: Scenario, route: Route) -> float:
    if not route.success:
        return math.nan
    return float(len(route.per_edge))
