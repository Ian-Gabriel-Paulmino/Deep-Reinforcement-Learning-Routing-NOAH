"""Activation strategies: how an RI maps to blocked edges and travel times.

The ``ActivationStrategy`` protocol decouples the *rule* for blocking edges
(deterministic threshold today, possibly dynamic/probabilistic tomorrow)
from the rest of the pipeline. The strategy name is recorded in the
benchmark manifest as a display label; the materialised
``blocked_edges`` and ``travel_time_map`` flow through unchanged.

This is the agnosticism guarantee the frontend depends on: swapping a
strategy means changing a label in ``benchmark.json``; the artifact
shape downstream stays identical.
"""

from __future__ import annotations

from typing import Protocol

import networkx as nx

from ..schemas import edge_key  # re-export for strategy implementations


class ActivationStrategy(Protocol):
    """Computes which edges are blocked at a given RI and the travel-time
    weighting for the remaining passable edges.
    """

    name: str  # e.g. "deterministic_v3", "dynamic_probabilistic"

    def compute_blocked_edges(
        self,
        graph: nx.DiGraph,
        ri_key: str,
        scenario_seed: int | None = None,
    ) -> set[tuple[str, str]]:
        ...

    def compute_travel_time_map(
        self,
        graph: nx.DiGraph,
        ri_key: str,
    ) -> dict[str, float]:
        ...


__all__ = ["ActivationStrategy", "edge_key"]
