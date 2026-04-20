"""NNA-AStar-Blind runner — nearest-neighbor + A* + blind execution.

Planning is hazard- and block-blind (like :class:`NNAAStar`), using the same
Euclidean-minutes heuristic. Execution commits to the plan and fails with
``failure_reason = "blocked"`` on first blocked-edge encounter — no replan.

See :func:`src.evaluation.runners.base.run_nna_blind` for the shared
execution loop and :mod:`src.evaluation.runners.nna_astar` for the heuristic
rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from ..schemas import Route, Scenario
from .base import GraphView, config_hash, run_nna_blind


# La Trinidad sits near 16.4°N, so a flat-earth approximation is accurate
# enough over the ~20 km extent of the graph. These are meters per degree.
_LAT_M_PER_DEG = 111_320.0
_LON_M_PER_DEG = 107_000.0

# Baseline speed (30 km/h) in meters per minute. Matches `prepare_data.py`
# §3.1.1 and `nna_astar.py`.
_M_PER_MIN_AT_30KMH = 500.0


def _make_astar_path_fn(G: nx.DiGraph):
    coords = {
        n: (float(d.get("x", 0.0)), float(d.get("y", 0.0)))
        for n, d in G.nodes(data=True)
    }

    def _heuristic(u: str, target: str) -> float:
        x1, y1 = coords[u]
        x2, y2 = coords[target]
        dx = (x2 - x1) * _LON_M_PER_DEG
        dy = (y2 - y1) * _LAT_M_PER_DEG
        dist_m = math.hypot(dx, dy)
        return dist_m / _M_PER_MIN_AT_30KMH

    def _astar_path_fn(G_view: nx.DiGraph, s: str, t: str, weight: str):
        try:
            path = nx.astar_path(
                G_view, s, t, heuristic=_heuristic, weight=weight
            )
            cost = nx.astar_path_length(
                G_view, s, t, heuristic=_heuristic, weight=weight
            )
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float("inf")

    return _astar_path_fn


@dataclass
class NNAAStarBlind:
    algorithm_id: str = "NNA-AStar-Blind"
    plan_weight: str = "base_time"
    policy_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy_metadata = dict(self.policy_metadata)
        self.policy_metadata.setdefault(
            "variant", "astar_hazard_blind_no_replan"
        )
        self.policy_metadata.setdefault("plan_weight", self.plan_weight)
        self.policy_metadata.setdefault(
            "heuristic", "euclidean_minutes_flat_earth_latN16"
        )
        self.algorithm_config_hash = config_hash(
            {"algorithm_id": self.algorithm_id, **self.policy_metadata}
        )

    def run(self, scenario: Scenario, view: GraphView) -> Route:
        path_fn = _make_astar_path_fn(view.base_graph)
        return run_nna_blind(
            scenario=scenario,
            view=view,
            algorithm_id=self.algorithm_id,
            algorithm_config_hash=self.algorithm_config_hash,
            path_fn=path_fn,
            plan_on=self.plan_weight,
            policy_metadata=self.policy_metadata,
        )
