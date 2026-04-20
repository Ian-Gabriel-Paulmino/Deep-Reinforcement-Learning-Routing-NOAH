"""Per-episode wall-clock runtime in milliseconds.

Thin re-export of ``route.wall_time_ms`` as a registered metric so it flows
through the same aggregation and CSV pipelines as the other metrics.
Unlike ``travel_time`` / ``hazard_exposure`` / ``steps`` / ``distance``,
this is **always defined** — even for failed episodes runtime is still
meaningful (e.g., "how long did the NNA spend before failing on a block?").
"""

from __future__ import annotations

from ..schemas import Route, Scenario


def compute(scenario: Scenario, route: Route) -> float:
    return float(route.wall_time_ms)
