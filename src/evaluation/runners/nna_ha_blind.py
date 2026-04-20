"""NNA-Dijkstra-HA-Blind runner — hazard-aware weighting, block-blind, no replan.

**Despite the HA suffix, this is NOT an oracle.** It plans on the **full**
base graph (including blocked edges) with the hazard-aware ``travel_time``
weight (λ drag from manuscript §B), and fails with
``failure_reason = "blocked"`` the first time the planned next edge is in
``scenario.blocked_set()``. The "HA" tag refers only to the weighting
function — the λ drag penalizes traversal of hazard-heavy edges — not to
block foresight. Contrast with :class:`NNADijkstraHA` (``runners/nna_ha.py``)
which plans on the passable subgraph and is a true oracle.

**Purpose in the thesis comparison.** Fills the "hazard-aware weights +
block-blind" cell of the 2×2 capability matrix left empty by the prior
tiering:

- ``NNA-Dijkstra-Blind``: hazard-blind + block-blind.
- ``NNA-Dijkstra``: hazard-blind + block-aware (reactive replan).
- **``NNA-Dijkstra-HA-Blind``: hazard-aware weights + block-blind.**
- ``NNA-Dijkstra-HA``: hazard-aware weights + block-aware (oracle).

The gap ``Blind → HA-Blind`` isolates *what hazard-aware weighting alone
buys you*; the gap ``HA-Blind → HA`` isolates *what block foresight buys
you on top of those weights*. Together they decompose the single
``Blind → HA`` jump into interpretable halves.

**Mid-RI dominance hypothesis (falsifiable).** HA-Blind should strictly
dominate Blind on success rate at RI2–RI3, where blocks are numerous but
the passable SCC is still rich. Hazard-aware weights push the planner
off high-λ edges, and high-λ edges (high hazard scores) are *also* the
ones most likely to be blocked under the thresholded block rule, so
HA-Blind incidentally avoids many blocks without knowing they exist.

At the tails the gap narrows:

- **RI1**: near-zero blocks → both variants near 100% success; HA-Blind
  may pay a small ``travel_time`` premium for unproductive detours.
- **RI5**: most edges blocked → the shortest ``travel_time`` path
  frequently still includes blocked edges; both variants fail heavily.

Uniform dominance across all RI would signal either a bug or a
misunderstanding of the block-rule/weighting alignment. The block rule is
discrete (``H_f ≥ θ_f(RI)``) while λ drag is continuous
(``1 + α_f·H_f + α_l·H_l``), so HA-Blind's incidental block-avoidance
works only where high-λ and high-block-probability overlap — which is the
mid-RI regime.

**A\\* heuristic admissibility note.** The Euclidean-minutes heuristic
(lower-bound at 30 km/h) used by the existing ``NNA-AStar-Blind`` remains
admissible for ``travel_time`` because ``travel_time ≥ base_time`` for
every edge (``μ(RI) ≤ 1`` and ``λ_hazard ≥ 1``). An A\\* companion
runner is straightforward but deferred per YAGNI — it would produce
byte-identical routes to this Dijkstra variant (A\\*-with-admissible-h ≡
Dijkstra on the same weight).

See :func:`src.evaluation.runners.base.run_nna_blind` for the shared
execution loop (identical to :class:`NNADijkstraBlind`; only the planning
substrate and weight change).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from ..schemas import Route, Scenario
from .base import GraphView, config_hash, run_nna_blind


def _dijkstra_path_fn(G: nx.DiGraph, s: str, t: str, weight: str):
    try:
        path = nx.dijkstra_path(G, s, t, weight=weight)
        cost = nx.dijkstra_path_length(G, s, t, weight=weight)
        return path, cost
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")


@dataclass
class NNADijkstraHABlind:
    algorithm_id: str = "NNA-Dijkstra-HA-Blind"
    plan_weight: str = "travel_time"
    policy_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy_metadata = dict(self.policy_metadata)
        self.policy_metadata.setdefault(
            "variant", "dijkstra_hazard_aware_no_replan"
        )
        self.policy_metadata.setdefault("plan_weight", self.plan_weight)
        self.policy_metadata.setdefault(
            "plan_substrate", "base_graph_with_travel_time"
        )
        self.algorithm_config_hash = config_hash(
            {"algorithm_id": self.algorithm_id, **self.policy_metadata}
        )

    def run(self, scenario: Scenario, view: GraphView) -> Route:
        return run_nna_blind(
            scenario=scenario,
            view=view,
            algorithm_id=self.algorithm_id,
            algorithm_config_hash=self.algorithm_config_hash,
            path_fn=_dijkstra_path_fn,
            plan_on=self.plan_weight,
            policy_metadata=self.policy_metadata,
            plan_graph=view.hazard_aware_full_graph,
        )
