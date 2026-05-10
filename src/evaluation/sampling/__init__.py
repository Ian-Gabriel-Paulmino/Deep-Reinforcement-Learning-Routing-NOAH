"""Scenario samplers: how (depot, stops, RI) tuples are drawn.

The ``ScenarioSampler`` protocol decouples the sampling rule from the rest
of the harness. The sampler name is recorded in the benchmark manifest;
the frontend renders it as a display label and never branches on it.
"""

from __future__ import annotations

from typing import Protocol

import networkx as nx

from ..activation import ActivationStrategy
from ..schemas import Scenario


class ScenarioSampler(Protocol):
    """Picks (depot, stops, RI) tuples for a benchmark."""

    name: str  # e.g. "uniform_open", "scc_restricted"

    def sample(
        self,
        graph: nx.DiGraph,
        *,
        n_scenarios: int,
        k_deliveries: int,
        master_seed: int,
        activation: ActivationStrategy,
        rain_intensities: list[str],
        benchmark_id: str,
        max_steps: int,
        longitudinal: bool,
    ) -> list[Scenario]:
        ...


__all__ = ["ScenarioSampler"]
