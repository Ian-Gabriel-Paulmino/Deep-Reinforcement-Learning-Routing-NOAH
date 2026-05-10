"""UniformOpenSampler -- longitudinal uniform sampler.

Sampling rule:
  * Depot has >= 1 passable outgoing edge at every rain intensity it
    will be evaluated at. This is the only feasibility constraint --
    no SCC restriction, no full mutual-reachability rejection.
  * Stops are drawn uniformly from the full graph (excluding the depot).
  * Longitudinal (default): one (depot, stops) tuple per ``scenario_idx``
    is rolled across every RI. Each scenarios.jsonl row shares the same
    depot+stops; only ``rain_level``, ``blocked_edges``, and
    ``travel_time_map`` differ per RI.
  * Non-longitudinal: each (depot, stops) is paired with one RI in
    round-robin order; total row count == ``n_scenarios``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import networkx as nx

from ..activation import ActivationStrategy
from ..schemas import Scenario
from ..scenario_generator import build_passable_graph


logger = logging.getLogger("evaluation.sampling.uniform_open")


@dataclass
class UniformOpenSampler:
    name: str = "uniform_open"
    max_depot_reroll_attempts: int = 200
    # When both are set, scenario_idx == 0 uses these instead of a random
    # draw. Validation: depot must be non-isolated at every RI; stops must
    # be distinct from each other and from depot. Configured via the
    # ``--inject-depot`` / ``--inject-stops`` CLI flags or the
    # ``bundle_name`` field on POST /api/v1/benchmarks.
    inject_depot: str | None = None
    inject_stops: tuple[str, ...] | None = None

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
        # Pre-compute per-RI state.
        ri_state: dict[str, dict] = {}
        for ri in rain_intensities:
            seed = (
                activation.activation_seed(master_seed, ri)
                if hasattr(activation, "activation_seed")
                else master_seed + int(ri.replace("RI", ""))
            )
            blocked = activation.compute_blocked_edges(graph, ri, scenario_seed=seed)
            passable = build_passable_graph(graph, blocked)
            tmap = activation.compute_travel_time_map(graph, ri)
            ri_state[ri] = {
                "activation_seed": seed,
                "blocked": blocked,
                "passable": passable,
                "travel_time_map": tmap,
            }
            logger.info(
                "  %s: blocked %d/%d edges (%.1f%%)",
                ri,
                len(blocked),
                graph.number_of_edges(),
                100.0 * len(blocked) / max(1, graph.number_of_edges()),
            )

        # Depot eligibility: longitudinal => intersect across all RIs;
        # non-longitudinal => per-RI.
        if longitudinal:
            eligible_depots: set[str] = set(graph.nodes())
            for ri in rain_intensities:
                passable = ri_state[ri]["passable"]
                eligible_depots &= {
                    n for n in graph.nodes() if passable.out_degree(n) >= 1
                }
            depot_pools_by_ri = {ri: list(eligible_depots) for ri in rain_intensities}
            logger.info(
                "longitudinal depot pool: %d nodes non-isolated across all %d RIs",
                len(eligible_depots),
                len(rain_intensities),
            )
            if len(eligible_depots) == 0:
                raise RuntimeError(
                    "No node is non-isolated across all rain intensities. "
                    "Use non-longitudinal mode or restrict the RI set."
                )
        else:
            depot_pools_by_ri = {
                ri: [
                    n for n in graph.nodes()
                    if ri_state[ri]["passable"].out_degree(n) >= 1
                ]
                for ri in rain_intensities
            }
            for ri, pool in depot_pools_by_ri.items():
                logger.info("  %s: %d non-isolated depot candidates", ri, len(pool))

        all_nodes = list(graph.nodes())
        graph_id = graph.graph.get("graph_id") or benchmark_id
        node_set = set(all_nodes)
        scenarios: list[Scenario] = []

        # Validate the injected (depot, stops) up-front, before any random
        # draws, so a bad bundle aborts the run before any work is wasted.
        inject_active = self.inject_depot is not None and self.inject_stops is not None
        if inject_active:
            depot_id = self.inject_depot
            stop_ids = list(self.inject_stops or ())
            if depot_id not in node_set:
                raise ValueError(
                    f"injected depot {depot_id!r} not in graph"
                )
            for s in stop_ids:
                if s not in node_set:
                    raise ValueError(f"injected stop {s!r} not in graph")
            if len(stop_ids) != k_deliveries:
                raise ValueError(
                    f"injected stops length {len(stop_ids)} does not match "
                    f"k_deliveries={k_deliveries}"
                )
            if depot_id in stop_ids:
                raise ValueError("injected depot must not appear in stops")
            if len(set(stop_ids)) != len(stop_ids):
                raise ValueError("injected stops must be distinct")
            # Depot non-isolation across the same constraint as the random
            # path: longitudinal => intersect of every RI's eligible set.
            for ri in rain_intensities:
                pool = (
                    depot_pools_by_ri[rain_intensities[0]]
                    if longitudinal
                    else depot_pools_by_ri[ri]
                )
                if depot_id not in pool:
                    raise ValueError(
                        f"injected depot {depot_id!r} is isolated at {ri} "
                        f"(no passable outgoing edge); pick a different bundle"
                    )
            logger.info(
                "injecting bundle: depot=%s, %d stops, into scenario_idx=0",
                depot_id,
                len(stop_ids),
            )

        for scenario_idx in range(n_scenarios):
            # Per-scenario RNG so adding more scenarios later doesn't shift
            # earlier draws.
            rng = random.Random(hash((master_seed, scenario_idx)) & 0xFFFFFFFF)
            use_injected = inject_active and scenario_idx == 0

            if longitudinal:
                depot_pool = depot_pools_by_ri[rain_intensities[0]]
                if not depot_pool:
                    raise RuntimeError(f"empty depot pool at scenario {scenario_idx}")
                if use_injected:
                    depot = self.inject_depot
                    stops = list(self.inject_stops or ())
                else:
                    depot = rng.choice(depot_pool)
                    stop_pool = [n for n in all_nodes if n != depot]
                    stops = rng.sample(stop_pool, k_deliveries)

                for ri in rain_intensities:
                    state = ri_state[ri]
                    scenario_id = f"{benchmark_id}_{scenario_idx:06d}_{ri}"
                    metadata = {
                        "generator_version": "2.0",
                        "master_seed": master_seed,
                        "ri_key": ri,
                        "scenario_idx": scenario_idx,
                        "sampler": self.name,
                        "longitudinal": True,
                    }
                    if use_injected:
                        metadata["injected_from_bundle"] = True
                    scenarios.append(
                        Scenario(
                            scenario_id=scenario_id,
                            graph_id=graph_id,
                            rain_level=int(ri.replace("RI", "")),
                            activation_mode=activation.name,
                            activation_seed=state["activation_seed"],
                            start_node=depot,
                            delivery_nodes=list(stops),
                            blocked_edges=[
                                [u, v] for (u, v) in sorted(state["blocked"])
                            ],
                            travel_time_map=state["travel_time_map"],
                            max_steps=max_steps,
                            num_deliveries=k_deliveries,
                            metadata=metadata,
                        )
                    )
            else:
                ri = rain_intensities[scenario_idx % len(rain_intensities)]
                state = ri_state[ri]
                depot_pool = depot_pools_by_ri[ri]
                if not depot_pool:
                    raise RuntimeError(f"empty depot pool for {ri}")
                if use_injected:
                    depot = self.inject_depot
                    stops = list(self.inject_stops or ())
                else:
                    depot = rng.choice(depot_pool)
                    stop_pool = [n for n in all_nodes if n != depot]
                    stops = rng.sample(stop_pool, k_deliveries)

                scenario_id = f"{benchmark_id}_{scenario_idx:06d}"
                metadata = {
                    "generator_version": "2.0",
                    "master_seed": master_seed,
                    "ri_key": ri,
                    "scenario_idx": scenario_idx,
                    "sampler": self.name,
                    "longitudinal": False,
                }
                if use_injected:
                    metadata["injected_from_bundle"] = True
                scenarios.append(
                    Scenario(
                        scenario_id=scenario_id,
                        graph_id=graph_id,
                        rain_level=int(ri.replace("RI", "")),
                        activation_mode=activation.name,
                        activation_seed=state["activation_seed"],
                        start_node=depot,
                        delivery_nodes=list(stops),
                        blocked_edges=[
                            [u, v] for (u, v) in sorted(state["blocked"])
                        ],
                        travel_time_map=state["travel_time_map"],
                        max_steps=max_steps,
                        num_deliveries=k_deliveries,
                        metadata=metadata,
                    )
                )

        return scenarios


__all__ = ["UniformOpenSampler"]
