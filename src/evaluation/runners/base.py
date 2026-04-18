"""Shared runner infrastructure: the policy protocol and the env simulator.

Every policy (NNA-Dijkstra, NNA-A*, NNA-HA, DQN variants) implements the same
interface and interacts with the same simulator. This keeps the "only the
policy differs" invariant strict — shared failure modes, shared block-reveal
rules, shared travel-time accounting.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import networkx as nx

from ..schemas import EdgeStep, Route, Scenario, edge_key


# ---------------------------------------------------------------------------
# Graph registry (policies only see attributes they are allowed to see)
# ---------------------------------------------------------------------------


@dataclass
class GraphView:
    """A lightweight view over a base graph exposing only what a given
    policy is permitted to plan on.

    - ``base_graph`` is the full directed graph with hazard scores and
      ``base_time``. Used by hazard-blind baselines for planning and by all
      policies for edge-attribute lookup during execution.
    - ``passable_graph`` is ``base_graph`` with blocked edges removed; any
      policy may use it for replanning on encountered blockage.
    - ``activated_graph`` is ``passable_graph`` with ``travel_time`` set per
      the scenario's ``travel_time_map``. Oracle-only (NNA-HA).
    """

    base_graph: nx.DiGraph
    passable_graph: nx.DiGraph
    activated_graph: nx.DiGraph


def build_graph_view(base_graph: nx.DiGraph, scenario: Scenario) -> GraphView:
    blocked = scenario.blocked_set()
    passable = nx.DiGraph()
    passable.add_nodes_from(base_graph.nodes(data=True))
    for u, v, data in base_graph.edges(data=True):
        if (u, v) not in blocked:
            passable.add_edge(u, v, **data)

    activated = nx.DiGraph()
    activated.add_nodes_from(passable.nodes(data=True))
    for u, v, data in passable.edges(data=True):
        tt = scenario.travel_time_map.get(edge_key(u, v))
        if tt is None:
            continue  # defensive; shouldn't happen for non-blocked edges
        new_data = dict(data)
        new_data["travel_time"] = float(tt)
        activated.add_edge(u, v, **new_data)
    return GraphView(base_graph=base_graph, passable_graph=passable, activated_graph=activated)


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


class Policy(Protocol):
    algorithm_id: str
    algorithm_config_hash: str
    policy_metadata: dict

    def run(self, scenario: Scenario, view: GraphView) -> Route:
        """Execute the policy on a scenario and return a Route."""
        ...


def config_hash(payload: dict) -> str:
    import json as _json

    blob = _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Execution helpers shared by all NNA-style policies
# ---------------------------------------------------------------------------


def _edge_step_record(
    u: str,
    v: str,
    step: int,
    was_replan: bool,
    base_graph: nx.DiGraph,
    scenario: Scenario,
) -> EdgeStep:
    data = base_graph[u][v]
    return EdgeStep(
        u=u,
        v=v,
        step=step,
        was_replan=was_replan,
        travel_time=float(scenario.travel_time(u, v)),
        hazard_flood=float(data.get("flood_hazard", 0.0)),
        hazard_landslide=float(data.get("landslide_hazard", 0.0)),
        length_m=float(data.get("length", 0.0)),
    )


def run_nna_with_fair_replan(
    *,
    scenario: Scenario,
    view: GraphView,
    algorithm_id: str,
    algorithm_config_hash: str,
    path_fn,
    plan_on: str,
    policy_metadata: dict,
) -> Route:
    """Shared execution loop for NNA-Dijkstra and NNA-A\\*.

    Planning happens on ``view.base_graph`` (hazard-blind, full edges) using
    ``path_fn(G, s, t, weight) -> (path, cost)`` and the weight name in
    ``plan_on`` (typically ``"base_time"``).

    During execution, if the planned path's next edge is blocked, the policy
    **replans from the current node over the passable graph** to the same
    target and continues. If no passable path exists, the episode fails with
    ``failure_reason = "trapped"``. This mirrors the DQN's action-masking
    behavior so the only difference between policies is the policy, not the
    structural ability to avoid blocks (see plan §Locked decision 1).
    """
    t0 = time.perf_counter()
    base_graph = view.base_graph
    passable = view.passable_graph
    blocked = scenario.blocked_set()

    current = scenario.start_node
    remaining = list(scenario.delivery_nodes)
    visit_order: list[str] = []
    edge_sequence: list[list[str]] = []
    per_edge: list[dict] = []
    step_idx = 0
    replan_count = 0

    while remaining:
        # 1) Nearest unvisited delivery by hazard-blind planner.
        best_target: Optional[str] = None
        best_plan: Optional[list[str]] = None
        best_cost = float("inf")
        for target in remaining:
            path, cost = path_fn(base_graph, current, target, plan_on)
            if path is not None and cost < best_cost:
                best_cost = cost
                best_target = target
                best_plan = path

        if best_target is None or best_plan is None:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return Route(
                scenario_id=scenario.scenario_id,
                algorithm_id=algorithm_id,
                algorithm_config_hash=algorithm_config_hash,
                visit_order=visit_order,
                edge_sequence=edge_sequence,
                per_edge=per_edge,
                success=False,
                failure_reason="trapped",
                replan_count=replan_count,
                wall_time_ms=wall_ms,
                policy_metadata=dict(policy_metadata),
            )

        # 2) Execute edges along the plan; replan locally if a block is hit.
        cursor = current
        plan = best_plan
        while cursor != best_target:
            # Find next edge in current plan starting at cursor.
            idx = plan.index(cursor)
            if idx + 1 >= len(plan):
                break
            nxt = plan[idx + 1]

            if (cursor, nxt) in blocked:
                # Replan on passable graph from cursor to best_target.
                replan_count += 1
                if nxt not in passable.nodes or not nx.has_path(
                    passable, cursor, best_target
                ):
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    return Route(
                        scenario_id=scenario.scenario_id,
                        algorithm_id=algorithm_id,
                        algorithm_config_hash=algorithm_config_hash,
                        visit_order=visit_order,
                        edge_sequence=edge_sequence,
                        per_edge=per_edge,
                        success=False,
                        failure_reason="trapped",
                        replan_count=replan_count,
                        wall_time_ms=wall_ms,
                        policy_metadata=dict(policy_metadata),
                    )
                try:
                    plan = nx.dijkstra_path(
                        passable, cursor, best_target, weight="base_time"
                    )
                except nx.NetworkXNoPath:
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    return Route(
                        scenario_id=scenario.scenario_id,
                        algorithm_id=algorithm_id,
                        algorithm_config_hash=algorithm_config_hash,
                        visit_order=visit_order,
                        edge_sequence=edge_sequence,
                        per_edge=per_edge,
                        success=False,
                        failure_reason="trapped",
                        replan_count=replan_count,
                        wall_time_ms=wall_ms,
                        policy_metadata=dict(policy_metadata),
                    )
                continue  # re-enter while with replanned route

            # Traverse edge (cursor -> nxt).
            step_record = _edge_step_record(
                cursor, nxt, step_idx, was_replan=False, base_graph=base_graph, scenario=scenario
            )
            per_edge.append(step_record.to_dict())
            edge_sequence.append([cursor, nxt])
            step_idx += 1
            cursor = nxt

            if step_idx > scenario.max_steps:
                wall_ms = (time.perf_counter() - t0) * 1000.0
                return Route(
                    scenario_id=scenario.scenario_id,
                    algorithm_id=algorithm_id,
                    algorithm_config_hash=algorithm_config_hash,
                    visit_order=visit_order,
                    edge_sequence=edge_sequence,
                    per_edge=per_edge,
                    success=False,
                    failure_reason="timeout",
                    replan_count=replan_count,
                    wall_time_ms=wall_ms,
                    policy_metadata=dict(policy_metadata),
                )

        visit_order.append(best_target)
        remaining.remove(best_target)
        current = best_target

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return Route(
        scenario_id=scenario.scenario_id,
        algorithm_id=algorithm_id,
        algorithm_config_hash=algorithm_config_hash,
        visit_order=visit_order,
        edge_sequence=edge_sequence,
        per_edge=per_edge,
        success=True,
        failure_reason=None,
        replan_count=replan_count,
        wall_time_ms=wall_ms,
        policy_metadata=dict(policy_metadata),
    )
