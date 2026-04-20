"""NNA-Dijkstra-Blind runner — nearest-neighbor + Dijkstra + blind execution.

Planning is hazard- and block-blind (like :class:`NNADijkstra`), but there is
**no fair-replan loop**: the policy commits to its planned path and fails
with ``failure_reason = "blocked"`` on the first blocked edge it attempts to
cross. Positioned in the thesis comparison as a *lower bound*: "how badly
does a classical NNA do if it knows nothing about blockages at any stage?"

See :func:`src.evaluation.runners.base.run_nna_blind` for the shared
execution loop.
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
class NNADijkstraBlind:
    algorithm_id: str = "NNA-Dijkstra-Blind"
    plan_weight: str = "base_time"
    policy_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy_metadata = dict(self.policy_metadata)
        self.policy_metadata.setdefault(
            "variant", "dijkstra_hazard_blind_no_replan"
        )
        self.policy_metadata.setdefault("plan_weight", self.plan_weight)
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
        )
