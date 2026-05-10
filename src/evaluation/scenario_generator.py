"""Stage 1: generate a committed benchmark of scenarios.

Reads a graph (graphml) and a ``_det``-style config to extract per-RI
deterministic blocking thresholds, samples feasible ``(start, deliveries)``
tuples stratified by RI, and writes ``benchmark.json`` + ``scenarios.jsonl``.

Usage (run from the Benguet project root):
    python -m src.evaluation.scenario_generator \\
        --graph data/staged_subgraphs/selected_subgraph_n200.graphml \\
        --graph-id la_trinidad_subgraph_n200 \\
        --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \\
        --benchmark-id la_trinidad_mini \\
        --num-scenarios 100 \\
        --num-deliveries 5 \\
        --master-seed 42

The generator is deterministic given ``--master-seed``. Blocked-edge sets are
baked into each scenario, so policies never roll dice at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

from .schemas import (
    BENCHMARK_SCHEMA_VERSION,
    Benchmark,
    Scenario,
    edge_key,
    write_benchmark,
    write_jsonl,
)


logger = logging.getLogger("evaluation.scenario_generator")


# ---------------------------------------------------------------------------
# Graph loading + attribute normalization
# ---------------------------------------------------------------------------

# The two main graphml sources in the project use different edge-attribute
# names. Normalize to a single schema so downstream code doesn't branch.
ATTR_ALIASES = {
    "flood_hazard": ("flood_hazard", "flood_score"),
    "landslide_hazard": ("landslide_hazard", "landslide_score"),
    "base_time": ("base_time", "travel_time_min"),
    "length": ("length",),
}


def load_graph(path: Path) -> nx.DiGraph:
    """Load a graphml file and return a ``DiGraph`` with normalized hazard
    attributes.

    Accepts undirected, directed, or multi-directed graphml inputs. For
    undirected inputs every edge is symmetrized into both directions so
    downstream reachability checks (SCC, has_path) work consistently.
    """
    logger.info(f"Loading graph: {path}")
    G_raw = nx.read_graphml(str(path))

    G = nx.DiGraph()
    G.graph.update(G_raw.graph)
    G.add_nodes_from(G_raw.nodes(data=True))

    if G_raw.is_directed():
        # DiGraph or MultiDiGraph: collapse parallel edges, keep first.
        for u, v, data in G_raw.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)
    else:
        # Undirected: symmetrize.
        for u, v, data in G_raw.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)
            if not G.has_edge(v, u):
                G.add_edge(v, u, **data)

    for _, _, data in G.edges(data=True):
        for canonical, aliases in ATTR_ALIASES.items():
            for a in aliases:
                if a in data:
                    data[canonical] = float(data[a])
                    break
            if canonical not in data:
                data[canonical] = 0.0

    logger.info(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G


# ---------------------------------------------------------------------------
# Deterministic activation per _det config
# ---------------------------------------------------------------------------


def load_rain_levels_from_config(config_path: Path) -> dict[str, dict[str, float]]:
    """Extract ``hazard.rain_levels`` from a training/_det config JSON."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    rain = cfg.get("hazard", {}).get("rain_levels")
    if rain is None:
        raise ValueError(f"config {config_path} has no hazard.rain_levels")
    return rain


def load_time_weights_from_config(
    config_path: Path,
) -> tuple[Optional[float], Optional[float]]:
    """Extract ``hazard.flood_time_weight`` / ``hazard.landslide_time_weight``.

    These are the travel-time-drag weights (α_f, α_l from manuscript §B).
    Returns ``(None, None)`` when the keys are absent so the caller can fall
    back to module defaults. Distinct from ``reward.w_flood`` /
    ``reward.w_landslide`` (which are training reward weights, §D).
    """
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    hazard = cfg.get("hazard", {})
    af = hazard.get("flood_time_weight")
    al = hazard.get("landslide_time_weight")
    return (
        float(af) if af is not None else None,
        float(al) if al is not None else None,
    )


def compute_blocked_edges(
    G: nx.DiGraph,
    rain_cfg: dict[str, float],
    activation_mode: str,
    activation_seed: int,
) -> set[tuple[str, str]]:
    """Return the set of blocked edges for this RI and activation mode.

    Deterministic mode: edge blocked iff hazard score meets threshold AND
    the block probability is 1.0 (the ``_det`` config semantics). In
    deterministic mode, ``activation_seed`` is unused.

    Probabilistic mode: edge blocked iff ``random.random() < prob`` AND the
    threshold is met. Uses a local ``random.Random(activation_seed)`` so the
    realization is reproducible without touching the global RNG.
    """
    rng = random.Random(activation_seed)
    blocked: set[tuple[str, str]] = set()

    f_thr = float(rain_cfg["flood_block_threshold"])
    f_prob = float(rain_cfg["flood_block_prob"])
    l_thr_raw = rain_cfg["landslide_block_threshold"]
    l_thr = float(l_thr_raw) if l_thr_raw is not None else float("inf")
    l_prob = float(rain_cfg.get("landslide_block_prob", 0.0))

    deterministic = activation_mode == "deterministic_v3"

    for u, v, data in G.edges(data=True):
        hf = float(data.get("flood_hazard", 0.0))
        hl = float(data.get("landslide_hazard", 0.0))

        flood_block = False
        if hf >= f_thr:
            if deterministic:
                flood_block = f_prob >= 1.0
            else:
                flood_block = rng.random() < f_prob

        ls_block = False
        if hl >= l_thr:
            if deterministic:
                ls_block = l_prob >= 1.0
            else:
                ls_block = rng.random() < l_prob

        if flood_block or ls_block:
            blocked.add((u, v))

    return blocked


def build_passable_graph(G: nx.DiGraph, blocked: set[tuple[str, str]]) -> nx.DiGraph:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if (u, v) not in blocked:
            H.add_edge(u, v, **data)
    return H


def compute_travel_time_map(
    G: nx.DiGraph,
    rain_cfg: dict[str, float],
    alpha_flood: float,
    alpha_landslide: float,
) -> dict[str, float]:
    """Per-edge effective travel time in minutes (manuscript §B).

    T_e = L_e / (v_e * mu(RI)) * lambda_hazard(H_f, H_l)
        = base_time / speed_mult * (1 + alpha_f * H_f + alpha_l * H_l)

    alpha_flood / alpha_landslide are the travel-time-drag weights from
    manuscript §B. They are distinct from the reward-penalty weights
    w_f, w_l in §D — do not conflate them.
    """
    speed_mult = max(float(rain_cfg["speed_mult"]), 1e-6)
    inv_mu = 1.0 / speed_mult
    tmap: dict[str, float] = {}
    for u, v, data in G.edges(data=True):
        hf = float(data.get("flood_hazard", 0.0))
        hl = float(data.get("landslide_hazard", 0.0))
        base = float(data.get("base_time", 0.0))
        lam = 1.0 + alpha_flood * hf + alpha_landslide * hl
        tmap[edge_key(u, v)] = base * inv_mu * lam
    return tmap


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------


def is_feasible(G_pass: nx.DiGraph, start: str, deliveries: list[str]) -> bool:
    if start not in G_pass:
        return False
    for d in deliveries:
        if d not in G_pass:
            return False
        if not nx.has_path(G_pass, start, d):
            return False
    for i, src in enumerate(deliveries):
        for dst in deliveries[i + 1:]:
            if not nx.has_path(G_pass, src, dst) or not nx.has_path(G_pass, dst, src):
                return False
    return True


# ---------------------------------------------------------------------------
# Benchmark generation
# ---------------------------------------------------------------------------


# Travel-time drag weights (α_f, α_l from manuscript §B: λ = 1 + α_f·H_f + α_l·H_l).
# Empirically calibrated -- NOT the same as reward-penalty weights (w_f, w_l).
# See README §6 for the α-vs-w distinction.
DEFAULT_ALPHA_FLOOD = 0.5
DEFAULT_ALPHA_LANDSLIDE = 0.5


def generate_benchmark(
    *,
    graph_path: Path,
    graph_id: str,
    config_path: Path,
    benchmark_id: str,
    out_dir: Path,
    n_scenarios: int,
    k_deliveries: int,
    master_seed: int,
    activation_strategy: str = "deterministic_v3",
    sampler: str = "scc_restricted",
    longitudinal: bool = False,
    max_steps: int = 220,
    ri_keys: Optional[list[str]] = None,
    max_sample_attempts: int = 200,
    alpha_flood: Optional[float] = None,
    alpha_landslide: Optional[float] = None,
    inject_depot: Optional[str] = None,
    inject_stops: Optional[list[str]] = None,
) -> tuple[Benchmark, list[Scenario]]:
    """Generate a benchmark using the configured sampler.

    Two sampler paths:
      * ``scc_restricted`` -- legacy stratified-by-RI draw, inline below.
      * ``uniform_open`` -- delegates to ``UniformOpenSampler``; supports
        ``longitudinal=True`` (one tuple rolled across every RI).
    """
    if sampler == "uniform_open":
        return _generate_via_uniform_open(
            graph_path=graph_path,
            graph_id=graph_id,
            config_path=config_path,
            benchmark_id=benchmark_id,
            out_dir=out_dir,
            n_scenarios=n_scenarios,
            k_deliveries=k_deliveries,
            master_seed=master_seed,
            activation_strategy=activation_strategy,
            longitudinal=longitudinal,
            max_steps=max_steps,
            ri_keys=ri_keys,
            alpha_flood=alpha_flood,
            alpha_landslide=alpha_landslide,
            inject_depot=inject_depot,
            inject_stops=inject_stops,
        )

    if longitudinal:
        raise NotImplementedError(
            "longitudinal=True requires --sampler=uniform_open. The legacy "
            "scc_restricted sampler only supports per-RI stratified draws."
        )
    if sampler != "scc_restricted":
        raise ValueError(
            f"Unknown sampler {sampler!r}. Available: 'scc_restricted', 'uniform_open'."
        )
    if inject_depot is not None or inject_stops is not None:
        raise ValueError(
            "--inject-depot/--inject-stops are only supported with --sampler=uniform_open."
        )

    G = load_graph(graph_path)
    rain_levels = load_rain_levels_from_config(config_path)
    ri_keys = ri_keys or sorted(rain_levels.keys())

    # Resolve travel-time drag weights (α_f, α_l from manuscript §B).
    # Priority: explicit CLI/kwarg override > config > module default.
    cfg_af, cfg_al = load_time_weights_from_config(config_path)
    resolved_af = alpha_flood if alpha_flood is not None else (
        cfg_af if cfg_af is not None else DEFAULT_ALPHA_FLOOD
    )
    resolved_al = alpha_landslide if alpha_landslide is not None else (
        cfg_al if cfg_al is not None else DEFAULT_ALPHA_LANDSLIDE
    )
    af_source = (
        "override" if alpha_flood is not None
        else ("config" if cfg_af is not None else "default")
    )
    al_source = (
        "override" if alpha_landslide is not None
        else ("config" if cfg_al is not None else "default")
    )
    logger.info(
        f"  travel-time weights: alpha_flood={resolved_af} ({af_source}), "
        f"alpha_landslide={resolved_al} ({al_source})"
    )

    if n_scenarios % len(ri_keys) != 0:
        logger.warning(
            f"  n_scenarios={n_scenarios} not divisible by "
            f"|ri_keys|={len(ri_keys)}; last bucket will absorb the remainder"
        )

    per_ri = n_scenarios // len(ri_keys)
    remainder = n_scenarios - per_ri * len(ri_keys)

    # Pre-compute blocked sets, travel-time maps, and the largest SCC of the
    # passable graph per RI. Sampling is restricted to the SCC so every
    # drawn (start, deliveries) tuple is mutually reachable by construction
    # -- otherwise feasibility filtering rejects almost every draw on sparse
    # subgraphs (e.g. staged n=200 has many pendant nodes).
    ri_state: dict[str, dict] = {}
    for ri in ri_keys:
        rain_cfg = rain_levels[ri]
        activation_seed = master_seed + int(ri.replace("RI", ""))
        blocked = compute_blocked_edges(G, rain_cfg, activation_strategy, activation_seed)
        passable = build_passable_graph(G, blocked)
        sccs = sorted(nx.strongly_connected_components(passable), key=len, reverse=True)
        largest_scc = sccs[0] if sccs else set()
        tmap = compute_travel_time_map(G, rain_cfg, resolved_af, resolved_al)
        ri_state[ri] = {
            "rain_cfg": rain_cfg,
            "activation_seed": activation_seed,
            "blocked": blocked,
            "passable": passable,
            "sampling_pool": list(largest_scc),
            "travel_time_map": tmap,
        }
        logger.info(
            f"  {ri}: blocked {len(blocked)}/{G.number_of_edges()} edges "
            f"({100.0*len(blocked)/max(1,G.number_of_edges()):.1f}%); "
            f"largest SCC = {len(largest_scc)}/{G.number_of_nodes()} nodes"
        )
        if len(largest_scc) < k_deliveries + 1:
            raise RuntimeError(
                f"{ri} passable graph's largest SCC has {len(largest_scc)} nodes "
                f"but scenario requires start+{k_deliveries} deliveries = "
                f"{k_deliveries+1} nodes. Reduce --num-deliveries or use a "
                f"denser graph."
            )

    rng = random.Random(master_seed)
    scenarios: list[Scenario] = []
    ri_counts: dict[str, int] = {ri: 0 for ri in ri_keys}

    for i, ri in enumerate(ri_keys):
        target = per_ri + (remainder if i == len(ri_keys) - 1 else 0)
        state = ri_state[ri]
        passable = state["passable"]
        pool = state["sampling_pool"]

        accepted = 0
        attempts = 0
        while accepted < target:
            if attempts >= max_sample_attempts * target:
                raise RuntimeError(
                    f"Too many infeasible draws for {ri}: accepted "
                    f"{accepted}/{target} after {attempts} attempts. "
                    f"Graph may be too disconnected at this RI."
                )
            attempts += 1
            sample = rng.sample(pool, k_deliveries + 1)
            start, deliveries = sample[0], sample[1:]
            if not is_feasible(passable, start, deliveries):
                continue

            scenario_id = f"{benchmark_id}_{len(scenarios):06d}"
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    graph_id=graph_id,
                    rain_level=int(ri.replace("RI", "")),
                    activation_mode=activation_strategy,
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
                    },
                )
            )
            accepted += 1
            ri_counts[ri] += 1

    benchmark = Benchmark(
        benchmark_id=benchmark_id,
        schema_version=BENCHMARK_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        master_seed=master_seed,
        graph_id=graph_id,
        graph_path=str(graph_path),
        n_scenarios=len(scenarios),
        n_evaluations=len(scenarios),
        k_deliveries=k_deliveries,
        rain_intensities=list(ri_keys),
        activation_strategy=activation_strategy,
        sampler=sampler,
        longitudinal=longitudinal,
        ri_distribution=ri_counts,
        feasibility_filtered=True,
    )

    write_benchmark(out_dir, benchmark)
    write_jsonl(
        out_dir / benchmark.scenarios_path,
        [s.to_dict() for s in scenarios],
    )
    logger.info(
        f"  wrote {len(scenarios)} scenarios to {out_dir / benchmark.scenarios_path}"
    )
    return benchmark, scenarios


# ---------------------------------------------------------------------------
# UniformOpen path (Stage 2)
# ---------------------------------------------------------------------------


def _generate_via_uniform_open(
    *,
    graph_path: Path,
    graph_id: str,
    config_path: Path,
    benchmark_id: str,
    out_dir: Path,
    n_scenarios: int,
    k_deliveries: int,
    master_seed: int,
    activation_strategy: str,
    longitudinal: bool,
    max_steps: int,
    ri_keys: Optional[list[str]],
    alpha_flood: Optional[float],
    alpha_landslide: Optional[float],
    inject_depot: Optional[str] = None,
    inject_stops: Optional[list[str]] = None,
) -> tuple[Benchmark, list[Scenario]]:
    # Local imports to avoid circular dependency with sampling/__init__.py
    # (which imports Scenario from this module's schemas sibling).
    from .activation.deterministic_v3 import DeterministicV3Strategy
    from .sampling.uniform_open import UniformOpenSampler

    if activation_strategy != "deterministic_v3":
        raise ValueError(
            f"Activation strategy {activation_strategy!r} not implemented. "
            f"Available: 'deterministic_v3'."
        )

    G = load_graph(graph_path)
    G.graph["graph_id"] = graph_id

    activation = DeterministicV3Strategy(
        config_path=config_path,
        alpha_flood=alpha_flood,
        alpha_landslide=alpha_landslide,
    )
    rain_intensities = ri_keys or activation.rain_keys()

    logger.info(
        "  travel-time weights: alpha_flood=%s, alpha_landslide=%s",
        activation.alpha_flood_resolved(),
        activation.alpha_landslide_resolved(),
    )

    sampler_obj = UniformOpenSampler(
        inject_depot=inject_depot,
        inject_stops=tuple(inject_stops) if inject_stops else None,
    )
    scenarios = sampler_obj.sample(
        G,
        n_scenarios=n_scenarios,
        k_deliveries=k_deliveries,
        master_seed=master_seed,
        activation=activation,
        rain_intensities=rain_intensities,
        benchmark_id=benchmark_id,
        max_steps=max_steps,
        longitudinal=longitudinal,
    )

    ri_counts: dict[str, int] = {ri: 0 for ri in rain_intensities}
    for s in scenarios:
        ri_counts[f"RI{s.rain_level}"] = ri_counts.get(f"RI{s.rain_level}", 0) + 1

    benchmark = Benchmark(
        benchmark_id=benchmark_id,
        schema_version=BENCHMARK_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        master_seed=master_seed,
        graph_id=graph_id,
        graph_path=str(graph_path),
        n_scenarios=n_scenarios,
        n_evaluations=len(scenarios),
        k_deliveries=k_deliveries,
        rain_intensities=list(rain_intensities),
        activation_strategy=activation_strategy,
        sampler=sampler_obj.name,
        longitudinal=longitudinal,
        ri_distribution=ri_counts,
        feasibility_filtered=False,
    )

    write_benchmark(out_dir, benchmark)
    write_jsonl(
        out_dir / benchmark.scenarios_path,
        [s.to_dict() for s in scenarios],
    )
    logger.info(
        "  wrote %d scenarios (longitudinal=%s) to %s",
        len(scenarios),
        longitudinal,
        out_dir / benchmark.scenarios_path,
    )
    return benchmark, scenarios


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1: generate a benchmark of scenarios.")
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--graph-id", required=True)
    p.add_argument("--config", required=True, type=Path, help="_det training config")
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--num-scenarios", type=int, default=2500)
    p.add_argument("--num-deliveries", type=int, default=5)
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--activation-strategy", default="deterministic_v3")
    p.add_argument(
        "--sampler",
        default="scc_restricted",
        choices=["scc_restricted", "uniform_open"],
        help="Sampling strategy. 'uniform_open' requires Stage 2 wiring.",
    )
    p.add_argument(
        "--longitudinal",
        action="store_true",
        help="One (depot, stops) tuple per scenario_id, rolled across all RIs. "
        "Requires --sampler=uniform_open (Stage 2).",
    )
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--ri-keys", nargs="*", default=None)
    # Travel-time drag weights (α_f, α_l from manuscript §B). When omitted,
    # the generator reads `hazard.flood_time_weight` / `landslide_time_weight`
    # from the config JSON, falling back to the module defaults (0.5 / 0.5).
    p.add_argument("--alpha-flood", type=float, default=None,
                   help="Override hazard.flood_time_weight from config")
    p.add_argument("--alpha-landslide", type=float, default=None,
                   help="Override hazard.landslide_time_weight from config")
    # Optional bundle injection: when both flags are set, scenario_idx == 0
    # uses these (depot, stops) instead of a random draw. Required-together;
    # only valid with --sampler=uniform_open.
    p.add_argument(
        "--inject-depot",
        default=None,
        help="Pin scenario 0's depot to this node id. Requires --inject-stops.",
    )
    p.add_argument(
        "--inject-stops",
        nargs="+",
        default=None,
        help="Pin scenario 0's stops to these node ids (in order). "
        "Length must match --num-deliveries. Requires --inject-depot.",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    # Required-together validation for the inject flags.
    if (args.inject_depot is None) != (args.inject_stops is None):
        p.error(
            "--inject-depot and --inject-stops must be set together "
            "(or both omitted)."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    out_dir = args.out_dir or (
        Path(__file__).resolve().parent / "benchmarks" / args.benchmark_id
    )
    t0 = time.perf_counter()
    generate_benchmark(
        graph_path=args.graph,
        graph_id=args.graph_id,
        config_path=args.config,
        benchmark_id=args.benchmark_id,
        out_dir=out_dir,
        n_scenarios=args.num_scenarios,
        k_deliveries=args.num_deliveries,
        master_seed=args.master_seed,
        activation_strategy=args.activation_strategy,
        sampler=args.sampler,
        longitudinal=args.longitudinal,
        max_steps=args.max_steps,
        ri_keys=args.ri_keys,
        alpha_flood=args.alpha_flood,
        alpha_landslide=args.alpha_landslide,
        inject_depot=args.inject_depot,
        inject_stops=args.inject_stops,
    )
    logger.info(f"Benchmark generation took {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
