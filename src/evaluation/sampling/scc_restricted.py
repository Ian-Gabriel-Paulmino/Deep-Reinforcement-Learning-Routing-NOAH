"""SCCRestrictedSampler -- legacy stratified-by-RI sampler.

Lifts the per-RI SCC-restricted draw from the legacy ``generate_benchmark``
body without changing semantics. Retained for byte-equivalent reproduction
of pre-overhaul benchmarks (e.g. ``la_trinidad_mini``) and as a fallback
sampler when callers explicitly opt in via ``--sampler scc_restricted``.

Behaviour:
  * Targets ``n_scenarios / |rain_intensities|`` scenarios per RI bucket
    (last bucket absorbs the remainder).
  * Depot + stops drawn from the largest strongly-connected component of
    the per-RI passable graph.
  * Rejects draws that fail full mutual reachability via ``is_feasible``.
  * ``longitudinal=True`` is rejected; this sampler doesn't support shared
    (depot, stops) across RIs.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import networkx as nx

from ..activation import ActivationStrategy
from ..schemas import Scenario
from ..scenario_generator import build_passable_graph, is_feasible


logger = logging.getLogger("evaluation.sampling.scc_restricted")


@dataclass
class SCCRestrictedSampler:
    name: str = "scc_restricted"
    max_sample_attempts: int = 200

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
        if longitudinal:
            raise ValueError(
                "SCCRestrictedSampler does not support longitudinal=True; "
                "use UniformOpenSampler for shared (depot, stops) across RIs."
            )

        if n_scenarios % len(rain_intensities) != 0:
            logger.warning(
                "n_scenarios=%d not divisible by |RI|=%d; last bucket absorbs the remainder",
                n_scenarios,
                len(rain_intensities),
            )

        per_ri = n_scenarios // len(rain_intensities)
        remainder = n_scenarios - per_ri * len(rain_intensities)

        ri_state: dict[str, dict] = {}
        for ri in rain_intensities:
            seed = (
                activation.activation_seed(master_seed, ri)
                if hasattr(activation, "activation_seed")
                else master_seed + int(ri.replace("RI", ""))
            )
            blocked = activation.compute_blocked_edges(graph, ri, scenario_seed=seed)
            passable = build_passable_graph(graph, blocked)
            sccs = sorted(nx.strongly_connected_components(passable), key=len, reverse=True)
            largest_scc = list(sccs[0]) if sccs else []
            tmap = activation.compute_travel_time_map(graph, ri)
            ri_state[ri] = {
                "activation_seed": seed,
                "blocked": blocked,
                "passable": passable,
                "sampling_pool": largest_scc,
                "travel_time_map": tmap,
            }
            logger.info(
                "  %s: blocked %d/%d (%.1f%%); largest SCC = %d/%d",
                ri,
                len(blocked),
                graph.number_of_edges(),
                100.0 * len(blocked) / max(1, graph.number_of_edges()),
                len(largest_scc),
                graph.number_of_nodes(),
            )
            if len(largest_scc) < k_deliveries + 1:
                raise RuntimeError(
                    f"{ri} largest SCC has {len(largest_scc)} nodes "
                    f"but scenario requires start+{k_deliveries} = "
                    f"{k_deliveries+1} nodes."
                )

        rng = random.Random(master_seed)
        scenarios: list[Scenario] = []
        graph_id = graph.graph.get("graph_id") or benchmark_id

        for i, ri in enumerate(rain_intensities):
            target = per_ri + (remainder if i == len(rain_intensities) - 1 else 0)
            state = ri_state[ri]
            passable = state["passable"]
            pool = state["sampling_pool"]

            accepted = 0
            attempts = 0
            while accepted < target:
                if attempts >= self.max_sample_attempts * target:
                    raise RuntimeError(
                        f"Too many infeasible draws for {ri}: "
                        f"{accepted}/{target} after {attempts} attempts."
                    )
                attempts += 1
                sample_nodes = rng.sample(pool, k_deliveries + 1)
                start, deliveries = sample_nodes[0], sample_nodes[1:]
                if not is_feasible(passable, start, deliveries):
                    continue

                scenario_id = f"{benchmark_id}_{len(scenarios):06d}"
                scenarios.append(
                    Scenario(
                        scenario_id=scenario_id,
                        graph_id=graph_id,
                        rain_level=int(ri.replace("RI", "")),
                        activation_mode=activation.name,
                        activation_seed=state["activation_seed"],
                        start_node=start,
                        delivery_nodes=deliveries,
                        blocked_edges=[[u, v] for (u, v) in sorted(state["blocked"])],
                        travel_time_map=state["travel_time_map"],
                        max_steps=max_steps,
                        num_deliveries=k_deliveries,
                        metadata={
                            "generator_version": "2.0",
                            "master_seed": master_seed,
                            "ri_key": ri,
                            "sample_attempts_for_this_ri": attempts,
                            "sampler": self.name,
                        },
                    )
                )
                accepted += 1

        return scenarios


__all__ = ["SCCRestrictedSampler"]
