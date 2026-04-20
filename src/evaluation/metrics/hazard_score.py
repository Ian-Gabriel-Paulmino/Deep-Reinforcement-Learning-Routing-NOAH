"""Raw, unweighted hazard exposure along the executed route.

    hazard_score = Σ_{e ∈ route} (H_f,e + H_l,e) · L_e

Differs from :mod:`hazard_exposure` (manuscript §3.6.1 C) in that it
**drops the reward weights** (w_f, w_l) from the ``_det`` config. It asks
"how many meters of hazard did the policy traverse, treating flood and
landslide equally?", independent of how the training reward was shaped.

Use both metrics together to separate "policy-weighted cost"
(``hazard_exposure``) from "raw physical exposure" (``hazard_score``).
NaN on failure.
"""

from __future__ import annotations

import math

from ..schemas import Route, Scenario


def compute(scenario: Scenario, route: Route) -> float:
    if not route.success:
        return math.nan
    total = 0.0
    for edge in route.per_edge:
        hazard_sum = float(edge["hazard_flood"]) + float(edge["hazard_landslide"])
        total += hazard_sum * float(edge["length_m"])
    return float(total)
