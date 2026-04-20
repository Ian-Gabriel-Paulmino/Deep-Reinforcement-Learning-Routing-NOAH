"""Total distance walked along the executed route, in meters.

Sums the ``length_m`` attribute across every traversed edge — this reflects
what the policy *actually* walked, not what it planned. NaN on failure.
"""

from __future__ import annotations

import math

from ..schemas import Route, Scenario


def compute(scenario: Scenario, route: Route) -> float:
    if not route.success:
        return math.nan
    return float(sum(edge["length_m"] for edge in route.per_edge))
