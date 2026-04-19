"""
Baseline Benchmarking Framework (DEPRECATED)
============================================
Retained for reproducibility of pre-2026-04-18 results only. For
thesis-reportable fairness-aware baselines and DQN evaluation, use the
3-stage harness at ``src/evaluation/`` instead — see
``src/evaluation/README.md``. That harness fixes two fairness problems in
this file:

- Structural asymmetry — NNAs here fail on blocked edges while the DQN
  action-masks them, so numbers aren't comparable.
- Uncontrolled scenario substrate — this file samples its own episodes,
  while the DQN eval samples separately, so algorithms run on
  non-identical worlds.

Do not add new features to this file. Bug fixes only, if at all.

----------------------------------------------------------------------

Implements the three NNA baseline algorithms from Section 3.5.2:
  1. NNA-Dijkstra      — hazard-blind, Dijkstra shortest path
  2. NNA-A*            — hazard-blind, A* with Euclidean heuristic
  3. NNA-Dijkstra-HA   — hazard-aware oracle, Dijkstra on activated graph

Evaluated within the same Monte Carlo environment as the DQN (Section 3.5.1),
using the same graph, same hazard activation logic (Table 3.2), and same metrics
(Section 3.6).

Key design: Each episode retries node sampling up to MAX_RETRIES times to find
a feasible scenario. The feasibility rate (how many attempts needed) is itself a
metric of network vulnerability at each RI level.

Usage:
    python -m src.benchmarks.monte_carlo
    python -m src.benchmarks.monte_carlo --deliveries 3 --episodes 200
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx
import numpy as np


# ============================================================
# 1. RAINFALL ACTIVATION PARAMETERS (Table 3.2)
# ============================================================

RAIN_PARAMS = {
    1: {
        "description": "Light",
        "speed_mult": 0.94,
        "flood_block_threshold": 1.0,
        "flood_block_prob": 0.10,
        "landslide_block_threshold": None,
        "landslide_block_prob": 0.0,
    },
    2: {
        "description": "Moderate",
        "speed_mult": 0.90,
        "flood_block_threshold": 1.0,
        "flood_block_prob": 0.30,
        "landslide_block_threshold": 0.8,
        "landslide_block_prob": 0.05,
    },
    3: {
        "description": "Heavy",
        "speed_mult": 0.85,
        "flood_block_threshold": 0.6,
        "flood_block_prob": 0.60,
        "landslide_block_threshold": 0.8,
        "landslide_block_prob": 0.15,
    },
    4: {
        "description": "Very Heavy",
        "speed_mult": 0.40,
        "flood_block_threshold": 0.6,
        "flood_block_prob": 0.90,
        "landslide_block_threshold": 0.5,
        "landslide_block_prob": 0.30,
    },
    5: {
        "description": "Extreme",
        "speed_mult": 0.20,
        "flood_block_threshold": 0.2,
        "flood_block_prob": 1.0,
        "landslide_block_threshold": 0.5,
        "landslide_block_prob": 1.0,
    },
}

ATTR_MAP = {
    "flood_hazard": "flood_score",
    "landslide_hazard": "landslide_score",
    "travel_time_min": "base_time",
}

MAX_RETRIES = 50


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class RouteResult:
    path: list
    visit_order: list
    total_distance: float
    total_time: float
    hazard_exposure: float
    success: bool
    algorithm: str
    failure_reason: str = ""


@dataclass
class EpisodeResult:
    rain_level: int
    start_node: str
    delivery_nodes: list
    route: RouteResult
    num_blocked_edges: int
    retries_needed: int
    structurally_infeasible: bool


@dataclass
class BenchmarkResults:
    algorithm: str
    total_episodes: int
    feasible_episodes: int
    structurally_infeasible: int
    success_rate: float
    success_rate_overall: float
    avg_travel_time: float
    std_travel_time: float
    avg_distance: float
    std_distance: float
    avg_hazard_exposure: float
    std_hazard_exposure: float
    robustness_score: float
    by_rain_level: dict
    wall_clock_seconds: float


# ============================================================
# 3. GRAPH LOADING
# ============================================================

def load_graph(path: str) -> nx.DiGraph:
    print(f"Loading graph from: {path}")
    G_raw = nx.read_graphml(path)

    if isinstance(G_raw, nx.MultiDiGraph):
        G = nx.DiGraph()
        G.graph.update(G_raw.graph)
        G.add_nodes_from(G_raw.nodes(data=True))
        for u, v, data in G_raw.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)
    else:
        G = G_raw

    for n, data in G.nodes(data=True):
        if "pos" in data:
            lon, lat = map(float, str(data["pos"]).split(","))
            data["lon"] = lon
            data["lat"] = lat
        elif "x" in data and "y" in data:
            data["lon"] = float(data["x"])
            data["lat"] = float(data["y"])

    for u, v, data in G.edges(data=True):
        for src, dst in ATTR_MAP.items():
            if src in data and dst not in data:
                data[dst] = data[src]
        data["flood_score"] = float(data.get("flood_score", 0))
        data["landslide_score"] = float(data.get("landslide_score", 0))
        data["base_time"] = float(data.get("base_time", 1.0))
        data["length"] = float(data.get("length", 100.0))

    n_edges = G.number_of_edges()
    flood_scores = [d["flood_score"] for _, _, d in G.edges(data=True)]
    landslide_scores = [d["landslide_score"] for _, _, d in G.edges(data=True)]

    print(f"  {G.number_of_nodes()} nodes, {n_edges} edges")
    print(f"  Flood:     {sum(1 for s in flood_scores if s > 0)}/{n_edges} "
          f"({sum(1 for s in flood_scores if s > 0)/n_edges*100:.1f}%)")
    print(f"  Landslide: {sum(1 for s in landslide_scores if s > 0)}/{n_edges} "
          f"({sum(1 for s in landslide_scores if s > 0)/n_edges*100:.1f}%)")
    print(f"  Edge/node ratio: {n_edges/G.number_of_nodes():.2f} "
          f"(~{n_edges//2} undirected for {G.number_of_nodes()} nodes, "
          f"tree would be {G.number_of_nodes()-1})")

    return G


# ============================================================
# 4. HAZARD ACTIVATION (Section 3.1.3, Equation 3.2 + 3.3)
# ============================================================

def activate_hazards(G: nx.DiGraph, rain_level: int) -> nx.DiGraph:
    params = RAIN_PARAMS[rain_level]
    G_active = G.copy()

    for u, v, data in G_active.edges(data=True):
        hf = data["flood_score"]
        hl = data["landslide_score"]

        flood_blocked = False
        if hf >= params["flood_block_threshold"]:
            flood_blocked = random.random() < params["flood_block_prob"]

        landslide_blocked = False
        if params["landslide_block_threshold"] is not None and hl >= params["landslide_block_threshold"]:
            landslide_blocked = random.random() < params["landslide_block_prob"]

        if flood_blocked or landslide_blocked:
            data["blocked"] = True
            data["travel_time"] = float("inf")
            data["hazard_cost"] = float("inf")
        else:
            data["blocked"] = False
            alpha_f, alpha_l = 0.5, 0.5
            lambda_hazard = 1.0 + alpha_f * hf + alpha_l * hl
            data["travel_time"] = data["base_time"] / params["speed_mult"] * lambda_hazard
            data["hazard_cost"] = hf + hl

    return G_active


# ============================================================
# 5. SHORTEST PATH HELPERS
# ============================================================

def _euclidean_dist(G, u, v):
    u_data, v_data = G.nodes[u], G.nodes[v]
    dx = float(u_data.get("lon", u_data.get("x", 0))) - float(v_data.get("lon", v_data.get("x", 0)))
    dy = float(u_data.get("lat", u_data.get("y", 0))) - float(v_data.get("lat", v_data.get("y", 0)))
    return math.sqrt(dx * dx + dy * dy)


def dijkstra_path_on_graph(G, source, target, weight="travel_time"):
    try:
        path = nx.dijkstra_path(G, source, target, weight=weight)
        cost = nx.dijkstra_path_length(G, source, target, weight=weight)
        return path, cost
    except nx.NetworkXNoPath:
        return None, float("inf")


def astar_path_on_graph(G, source, target, weight="travel_time"):
    try:
        heuristic = lambda u, v: _euclidean_dist(G, u, v)
        path = nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)
        cost = sum(G[path[i]][path[i + 1]][weight] for i in range(len(path) - 1))
        return path, cost
    except nx.NetworkXNoPath:
        return None, float("inf")


# ============================================================
# 6. PASSABLE GRAPH + FEASIBILITY
# ============================================================

def get_passable_graph(G_active: nx.DiGraph) -> nx.DiGraph:
    passable = nx.DiGraph()
    passable.add_nodes_from(G_active.nodes(data=True))
    for u, v, data in G_active.edges(data=True):
        if not data.get("blocked", False):
            passable.add_edge(u, v, **data)
    return passable


def check_feasibility(G_passable: nx.DiGraph, start: str, deliveries: list) -> bool:
    all_nodes = [start] + list(deliveries)
    for node in all_nodes:
        if node not in G_passable:
            return False
    for target in deliveries:
        if not nx.has_path(G_passable, start, target):
            return False
    for i, src in enumerate(deliveries):
        for dst in deliveries[i + 1:]:
            if not nx.has_path(G_passable, src, dst) or not nx.has_path(G_passable, dst, src):
                return False
    return True


# ============================================================
# 7. BASELINE ALGORITHMS (Section 3.5.2)
# ============================================================

def nna_dijkstra(G_active, start, deliveries) -> RouteResult:
    G_plan = G_active.copy()
    for u, v, data in G_plan.edges(data=True):
        data["plan_weight"] = data["base_time"]
    G_exec = get_passable_graph(G_active)
    return _run_nna(G_plan, G_exec, G_active, start, deliveries,
                    lambda g, s, t: dijkstra_path_on_graph(g, s, t, weight="plan_weight"),
                    "NNA-Dijkstra")


def nna_astar(G_active, start, deliveries) -> RouteResult:
    G_plan = G_active.copy()
    for u, v, data in G_plan.edges(data=True):
        data["plan_weight"] = data["base_time"]
    G_exec = get_passable_graph(G_active)
    return _run_nna(G_plan, G_exec, G_active, start, deliveries,
                    lambda g, s, t: astar_path_on_graph(g, s, t, weight="plan_weight"),
                    "NNA-A*")


def nna_dijkstra_ha(G_active, start, deliveries) -> RouteResult:
    G_exec = get_passable_graph(G_active)
    return _run_nna(G_exec, G_exec, G_active, start, deliveries,
                    lambda g, s, t: dijkstra_path_on_graph(g, s, t, weight="travel_time"),
                    "NNA-Dijkstra-HA")


def _run_nna(G_plan, G_exec, G_active, start, deliveries, path_fn, algorithm) -> RouteResult:
    current = start
    remaining = list(deliveries)
    visit_order, full_path = [], [current]
    total_time, total_distance, total_hazard = 0.0, 0.0, 0.0

    while remaining:
        best_node, best_cost = None, float("inf")
        for target in remaining:
            path, cost = path_fn(G_plan, current, target)
            if path is not None and cost < best_cost:
                best_cost, best_node = cost, target

        if best_node is None:
            return RouteResult(full_path, visit_order, total_distance, total_time,
                               total_hazard, False, algorithm, "no_reachable_delivery")

        exec_path, _ = dijkstra_path_on_graph(G_exec, current, best_node, weight="travel_time")
        if exec_path is None:
            return RouteResult(full_path, visit_order, total_distance, total_time,
                               total_hazard, False, algorithm, "execution_path_blocked")

        for i in range(len(exec_path) - 1):
            u, v = exec_path[i], exec_path[i + 1]
            edge_data = G_active[u][v]
            total_time += edge_data.get("travel_time", edge_data["base_time"])
            total_distance += edge_data.get("length", 0)
            total_hazard += edge_data.get("hazard_cost", 0)

        full_path.extend(exec_path[1:])
        visit_order.append(best_node)
        remaining.remove(best_node)
        current = best_node

    return RouteResult(full_path, visit_order, total_distance, total_time,
                       total_hazard, True, algorithm)


# ============================================================
# 8. MONTE CARLO RUNNER WITH RETRY SAMPLING
# ============================================================

def run_monte_carlo(G, algorithms, num_episodes=200, num_deliveries=5, seed=42):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    results = {name: [] for name in algorithms}

    print(f"\nMonte Carlo: {num_episodes} episodes x {len(algorithms)} algorithms")
    print(f"  Deliveries: {num_deliveries}, Max retries: {MAX_RETRIES}")
    print(f"  Sampling from: {len(nodes)} nodes")
    print()

    stats = {ri: {"attempts": 0, "found": 0, "blocked": []} for ri in range(1, 6)}

    for ep in range(num_episodes):
        rain_level = rng.randint(1, 5)

        random.seed(seed + ep)
        G_active = activate_hazards(G, rain_level)
        G_passable = get_passable_graph(G_active)
        num_blocked = sum(1 for _, _, d in G_active.edges(data=True) if d.get("blocked"))
        stats[rain_level]["blocked"].append(num_blocked)

        # Retry until feasible or exhausted
        feasible, retries = False, 0
        start_node, delivery_nodes = None, None

        for attempt in range(MAX_RETRIES):
            selected = rng.sample(nodes, num_deliveries + 1)
            stats[rain_level]["attempts"] += 1
            retries = attempt + 1

            if check_feasibility(G_passable, selected[0], selected[1:]):
                start_node, delivery_nodes = selected[0], selected[1:]
                feasible = True
                stats[rain_level]["found"] += 1
                break

        if not feasible:
            for name in algorithms:
                results[name].append(EpisodeResult(
                    rain_level=rain_level, start_node="", delivery_nodes=[],
                    route=RouteResult([], [], 0, 0, 0, False, name, "structurally_infeasible"),
                    num_blocked_edges=num_blocked, retries_needed=MAX_RETRIES,
                    structurally_infeasible=True,
                ))
            if (ep + 1) % 25 == 0 or ep == 0:
                print(f"  Ep {ep+1:>4}/{num_episodes} | RI={rain_level} | "
                      f"Blocked: {num_blocked:>4} ({num_blocked/G.number_of_edges()*100:.1f}%) | "
                      f"INFEASIBLE after {MAX_RETRIES} retries")
            continue

        for name, algo_fn in algorithms.items():
            route = algo_fn(G_active, start_node, delivery_nodes)
            results[name].append(EpisodeResult(
                rain_level=rain_level, start_node=start_node,
                delivery_nodes=delivery_nodes, route=route,
                num_blocked_edges=num_blocked, retries_needed=retries,
                structurally_infeasible=False,
            ))

        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  Ep {ep+1:>4}/{num_episodes} | RI={rain_level} | "
                  f"Blocked: {num_blocked:>4} ({num_blocked/G.number_of_edges()*100:.1f}%) | "
                  f"Feasible after {retries} retries")

    # Feasibility summary
    print(f"\n  {'RI':<5} {'Description':<12} {'AvgBlocked':>11} {'Feasibility':>12} {'AvgRetries':>11}")
    print("  " + "-" * 55)
    for ri in range(1, 6):
        s = stats[ri]
        if s["attempts"] > 0:
            n_eps = len(s["blocked"])
            avg_blk = np.mean(s["blocked"]) if s["blocked"] else 0
            feas_rate = s["found"] / n_eps * 100 if n_eps > 0 else 0
            avg_retries = s["attempts"] / n_eps if n_eps > 0 else 0
            print(f"  RI{ri}  {RAIN_PARAMS[ri]['description']:<12} "
                  f"{avg_blk:>8.0f} edges "
                  f"{feas_rate:>10.1f}% "
                  f"{avg_retries:>10.1f}")

    return results


# ============================================================
# 9. METRICS (Section 3.6)
# ============================================================

def compute_metrics(episodes, algorithm, wall_time):
    total = len(episodes)
    feasible_eps = [ep for ep in episodes if not ep.structurally_infeasible]
    infeasible_eps = [ep for ep in episodes if ep.structurally_infeasible]

    feasible_successes = [ep for ep in feasible_eps if ep.route.success]
    success_rate = len(feasible_successes) / len(feasible_eps) * 100 if feasible_eps else 0
    all_successes = [ep for ep in episodes if ep.route.success]
    success_rate_overall = len(all_successes) / total * 100 if total > 0 else 0

    times = [ep.route.total_time for ep in all_successes] if all_successes else [0]
    dists = [ep.route.total_distance for ep in all_successes] if all_successes else [0]
    hazards = [ep.route.hazard_exposure for ep in all_successes] if all_successes else [0]

    by_rain, rain_success_rates = {}, []

    for ri in range(1, 6):
        ri_eps = [ep for ep in episodes if ep.rain_level == ri]
        if not ri_eps:
            continue
        ri_feasible = [ep for ep in ri_eps if not ep.structurally_infeasible]
        ri_successes = [ep for ep in ri_feasible if ep.route.success]

        ri_sr = len(ri_successes) / len(ri_feasible) * 100 if ri_feasible else 0
        ri_sr_overall = len([ep for ep in ri_eps if ep.route.success]) / len(ri_eps) * 100

        ri_times = [ep.route.total_time for ep in ri_successes] if ri_successes else [0]
        ri_hazards = [ep.route.hazard_exposure for ep in ri_successes] if ri_successes else [0]

        by_rain[ri] = {
            "episodes": len(ri_eps),
            "feasible": len(ri_feasible),
            "infeasible": len(ri_eps) - len(ri_feasible),
            "feasibility_rate": round(len(ri_feasible) / len(ri_eps) * 100, 1),
            "success_rate": round(ri_sr, 1),
            "success_rate_overall": round(ri_sr_overall, 1),
            "avg_travel_time": round(float(np.mean(ri_times)), 1),
            "avg_hazard_exposure": round(float(np.mean(ri_hazards)), 2),
            "avg_blocked_edges": round(float(np.mean([ep.num_blocked_edges for ep in ri_eps])), 0),
            "avg_retries": round(float(np.mean([ep.retries_needed for ep in ri_eps])), 1),
        }
        rain_success_rates.append(ri_sr)

    if len(rain_success_rates) > 1 and np.mean(rain_success_rates) > 0:
        robustness = max(0.0, 1.0 - float(np.std(rain_success_rates) / np.mean(rain_success_rates)))
    else:
        robustness = 0.0

    return BenchmarkResults(
        algorithm=algorithm, total_episodes=total,
        feasible_episodes=len(feasible_eps),
        structurally_infeasible=len(infeasible_eps),
        success_rate=round(success_rate, 1),
        success_rate_overall=round(success_rate_overall, 1),
        avg_travel_time=round(float(np.mean(times)), 1),
        std_travel_time=round(float(np.std(times)), 1),
        avg_distance=round(float(np.mean(dists)), 1),
        std_distance=round(float(np.std(dists)), 1),
        avg_hazard_exposure=round(float(np.mean(hazards)), 2),
        std_hazard_exposure=round(float(np.std(hazards)), 2),
        robustness_score=round(robustness, 4),
        by_rain_level=by_rain,
        wall_clock_seconds=round(wall_time, 2),
    )


# ============================================================
# 10. REPORTING
# ============================================================

def print_results(all_results):
    print("\n" + "=" * 100)
    print("BASELINE BENCHMARK RESULTS")
    print("=" * 100)

    header = (f"{'Algorithm':<22} {'SR(feas)':>9} {'SR(all)':>8} "
              f"{'Feasible':>10} {'Infeasible':>10} "
              f"{'AvgTime':>9} {'AvgHazard':>10} {'Robust':>8}")
    print(header)
    print("-" * 100)

    for name, m in all_results.items():
        print(f"{m.algorithm:<22} "
              f"{m.success_rate:>8.1f}% "
              f"{m.success_rate_overall:>7.1f}% "
              f"{m.feasible_episodes:>5}/{m.total_episodes:<4} "
              f"{m.structurally_infeasible:>5}/{m.total_episodes:<4} "
              f"{m.avg_travel_time:>8.1f}m "
              f"{m.avg_hazard_exposure:>9.2f} "
              f"{m.robustness_score:>7.4f}")

    print("\n" + "-" * 100)
    print("BREAKDOWN BY RAIN INTENSITY")
    print("-" * 100)

    for ri in range(1, 6):
        first = list(all_results.values())[0]
        if ri not in first.by_rain_level:
            continue
        info = first.by_rain_level[ri]
        print(f"\n  RI{ri} ({RAIN_PARAMS[ri]['description']}) — "
              f"n={info['episodes']}, "
              f"feasible={info['feasible']}/{info['episodes']} ({info['feasibility_rate']:.0f}%), "
              f"avg blocked={info['avg_blocked_edges']:.0f}, "
              f"avg retries={info['avg_retries']:.1f}")

        for name, m in all_results.items():
            if ri in m.by_rain_level:
                d = m.by_rain_level[ri]
                print(f"    {m.algorithm:<20} "
                      f"SR={d['success_rate']:>5.1f}% (overall={d['success_rate_overall']:>5.1f}%)  "
                      f"Time={d['avg_travel_time']:>7.1f}m  "
                      f"Hazard={d['avg_hazard_exposure']:>6.2f}")

    print("\n" + "-" * 100)
    print("ALGORITHM FAILURE REASONS (feasible episodes only)")
    print("-" * 100)


def print_failure_reasons(episode_results):
    for name, episodes in episode_results.items():
        algo_failures = [ep for ep in episodes
                         if not ep.structurally_infeasible and not ep.route.success]
        if not algo_failures:
            print(f"  {name}: No algorithm failures")
            continue
        reasons = {}
        for ep in algo_failures:
            r = ep.route.failure_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        print(f"  {name}: {len(algo_failures)} failures")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"    {r}: {c}")


def save_results(all_results, path):
    with open(path, "w") as f:
        json.dump({n: asdict(m) for n, m in all_results.items()}, f, indent=2)
    print(f"\nResults saved to: {path}")


# ============================================================
# 11. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline Benchmarking Framework")
    parser.add_argument("--graph", default="data/la_trinidad_hazard_graph.graphml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--deliveries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()

    G = load_graph(args.graph)

    algorithms = {
        "NNA-Dijkstra": nna_dijkstra,
        "NNA-A*": nna_astar,
        "NNA-Dijkstra-HA": nna_dijkstra_ha,
    }

    t_start = time.time()
    episode_results = run_monte_carlo(
        G, algorithms,
        num_episodes=args.episodes,
        num_deliveries=args.deliveries,
        seed=args.seed,
    )
    t_total = time.time() - t_start

    all_metrics = {}
    for name in algorithms:
        m = compute_metrics(episode_results[name], name, t_total / len(algorithms))
        all_metrics[name] = m

    print_results(all_metrics)
    print_failure_reasons(episode_results)
    save_results(all_metrics, args.output)

    print(f"\nTotal wall clock: {t_total:.1f}s")


if __name__ == "__main__":
    main()
