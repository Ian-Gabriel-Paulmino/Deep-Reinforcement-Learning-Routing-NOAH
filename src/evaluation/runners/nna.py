"""NNA-family policies.

This module provides ``NNADijkstra``. NNA-A\\* and NNA-Dijkstra-HA are
scaffolded in the plan but not implemented in v1 of the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from ..schemas import Route, Scenario
from .base import GraphView, config_hash, run_nna_with_fair_replan


def _dijkstra_path_fn(G: nx.DiGraph, s: str, t: str, weight: str):
    try:
        path = nx.dijkstra_path(G, s, t, weight=weight)
        cost = nx.dijkstra_path_length(G, s, t, weight=weight)
        return path, cost
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")


@dataclass
class NNADijkstra:
    algorithm_id: str = "NNA-Dijkstra"
    plan_weight: str = "base_time"
    policy_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy_metadata = dict(self.policy_metadata)
        self.policy_metadata.setdefault("variant", "dijkstra_hazard_blind_with_fair_replan")
        self.policy_metadata.setdefault("plan_weight", self.plan_weight)
        self.algorithm_config_hash = config_hash(
            {"algorithm_id": self.algorithm_id, **self.policy_metadata}
        )

    def run(self, scenario: Scenario, view: GraphView) -> Route:
        return run_nna_with_fair_replan(
            scenario=scenario,
            view=view,
            algorithm_id=self.algorithm_id,
            algorithm_config_hash=self.algorithm_config_hash,
            path_fn=_dijkstra_path_fn,
            plan_on=self.plan_weight,
            policy_metadata=self.policy_metadata,
        )
