"""Hazard exposure per manuscript §3.6.1 C.

    Hazard Exposure = Σ_{e ∈ route} w_e · L_e
    where w_e = w_f · H_f,e + w_l · H_l,e

Weights are fixed to the _det config reward-weight defaults
(``w_flood = 0.6``, ``w_landslide = 0.4``). If we later need to sweep these,
expose them via a registry-bound config object.
"""

from __future__ import annotations

import math

from ..schemas import Route, Scenario


W_FLOOD = 0.6
W_LANDSLIDE = 0.4


def compute(scenario: Scenario, route: Route) -> float:
    if not route.success:
        return math.nan
    total = 0.0
    for edge in route.per_edge:
        w_e = W_FLOOD * float(edge["hazard_flood"]) + W_LANDSLIDE * float(
            edge["hazard_landslide"]
        )
        total += w_e * float(edge["length_m"])
    return float(total)
