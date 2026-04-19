"""NNA-Dijkstra-HA runner — hazard-aware oracle baseline.

Positioned as the **upper bound** in the comparison (manuscript §3.5.2),
not a peer competitor to the DQN. It sees the full blocked-edges set
AND the RI-adjusted travel_time_map at plan time — realistic deployment
conditions don't afford this foresight. Interpretation: "how close does
the DQN get to an oracle?"

Planning substrate: ``view.activated_graph`` (passable subgraph with
``travel_time`` attr populated from ``scenario.travel_time_map``).
Weight: ``travel_time``. No replan is needed — blocked edges are never
admitted to the plan.

Only failure mode: ``trapped`` (no path from current to any unvisited
delivery on the activated graph). Should never happen on a cohort
produced by the feasibility-filtered ``scenario_generator`` — which
sampled starts/deliveries from the passable graph's largest SCC.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from ..schemas import Route, Scenario
from .base import GraphView, _edge_step_record, config_hash


@dataclass
class NNADijkstraHA:
    algorithm_id: str = "NNA-Dijkstra-HA"
    plan_weight: str = "travel_time"
    policy_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy_metadata = dict(self.policy_metadata)
        self.policy_metadata.setdefault(
            "variant", "dijkstra_hazard_aware_oracle_no_replan"
        )
        self.policy_metadata.setdefault("plan_weight", self.plan_weight)
        self.algorithm_config_hash = config_hash(
            {"algorithm_id": self.algorithm_id, **self.policy_metadata}
        )

    def run(self, scenario: Scenario, view: GraphView) -> Route:
        t0 = time.perf_counter()
        base_graph = view.base_graph
        activated = view.activated_graph

        current = scenario.start_node
        remaining = list(scenario.delivery_nodes)
        visit_order: list[str] = []
        edge_sequence: list[list[str]] = []
        per_edge: list[dict] = []
        step_idx = 0

        while remaining:
            best_target: Optional[str] = None
            best_plan: Optional[list[str]] = None
            best_cost = float("inf")
            for target in remaining:
                try:
                    path = nx.dijkstra_path(
                        activated, current, target, weight=self.plan_weight
                    )
                    cost = nx.dijkstra_path_length(
                        activated, current, target, weight=self.plan_weight
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_target = target
                    best_plan = path

            if best_target is None or best_plan is None:
                wall_ms = (time.perf_counter() - t0) * 1000.0
                return Route(
                    scenario_id=scenario.scenario_id,
                    algorithm_id=self.algorithm_id,
                    algorithm_config_hash=self.algorithm_config_hash,
                    visit_order=visit_order,
                    edge_sequence=edge_sequence,
                    per_edge=per_edge,
                    success=False,
                    failure_reason="trapped",
                    replan_count=0,
                    wall_time_ms=wall_ms,
                    policy_metadata=dict(self.policy_metadata),
                )

            cursor = current
            for nxt in best_plan[1:]:
                step_record = _edge_step_record(
                    cursor, nxt, step_idx, False, base_graph, scenario
                )
                per_edge.append(step_record.to_dict())
                edge_sequence.append([cursor, nxt])
                step_idx += 1
                cursor = nxt
                if step_idx > scenario.max_steps:
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    return Route(
                        scenario_id=scenario.scenario_id,
                        algorithm_id=self.algorithm_id,
                        algorithm_config_hash=self.algorithm_config_hash,
                        visit_order=visit_order,
                        edge_sequence=edge_sequence,
                        per_edge=per_edge,
                        success=False,
                        failure_reason="timeout",
                        replan_count=0,
                        wall_time_ms=wall_ms,
                        policy_metadata=dict(self.policy_metadata),
                    )

            visit_order.append(best_target)
            remaining.remove(best_target)
            current = best_target

        wall_ms = (time.perf_counter() - t0) * 1000.0
        return Route(
            scenario_id=scenario.scenario_id,
            algorithm_id=self.algorithm_id,
            algorithm_config_hash=self.algorithm_config_hash,
            visit_order=visit_order,
            edge_sequence=edge_sequence,
            per_edge=per_edge,
            success=True,
            failure_reason=None,
            replan_count=0,
            wall_time_ms=wall_ms,
            policy_metadata=dict(self.policy_metadata),
        )
