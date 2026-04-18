"""Success indicator (0/1). Aggregates to success rate via mean."""

from __future__ import annotations

from ..schemas import Route, Scenario


def compute(scenario: Scenario, route: Route) -> float:
    return 1.0 if route.success else 0.0
