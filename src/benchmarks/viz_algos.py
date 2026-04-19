"""
Episode Visualizer for Monte Carlo Benchmarks
===============================================
Generates interactive Folium maps for a sample of benchmark episodes.
Each map shows:
  - Full road network (gray)
  - Blocked edges (red dashed)
  - Start node (green marker)
  - Delivery nodes (blue markers)
  - Algorithm routes (colored paths)
  - Failure point (red X) if route failed

Usage:
    python -m src.benchmarks.visualize_episodes
    python -m src.benchmarks.visualize_episodes --episodes 3 --deliveries 5 --ri 2
"""

import argparse
import math
import random
from pathlib import Path

import folium
import networkx as nx
import numpy as np

# Reuse components from the benchmark module
from src.benchmarks.monte_carlo import (
    RAIN_PARAMS, ATTR_MAP, load_graph,
    activate_hazards, get_passable_graph, check_feasibility,
    nna_dijkstra, nna_astar, nna_dijkstra_ha,
)


# Route colors per algorithm
ALGO_COLORS = {
    "NNA-Dijkstra": "#378ADD",      # blue
    "NNA-A*": "#1D9E75",            # teal
    "NNA-Dijkstra-HA": "#7F77DD",   # purple
}


def get_node_coords(G, node):
    """Get (lat, lon) for a node."""
    data = G.nodes[node]
    lat = float(data.get("lat", data.get("y", 0)))
    lon = float(data.get("lon", data.get("x", 0)))
    return lat, lon


def get_edge_coords(G, u, v):
    """Get [(lat, lon), ...] for an edge."""
    lat_u, lon_u = get_node_coords(G, u)
    lat_v, lon_v = get_node_coords(G, v)
    return [(lat_u, lon_u), (lat_v, lon_v)]


def build_episode_map(
    G, G_active, rain_level, start_node, delivery_nodes,
    algo_results, episode_num, output_dir
):
    """Build a single Folium map for one episode."""

    # Center map on the network
    lats = [float(d.get("lat", d.get("y", 0))) for _, d in G.nodes(data=True)
            if d.get("lat") or d.get("y")]
    lons = [float(d.get("lon", d.get("x", 0))) for _, d in G.nodes(data=True)
            if d.get("lon") or d.get("x")]

    if not lats or not lons:
        print("  WARNING: No coordinates found on nodes")
        return

    center = [np.mean(lats), np.mean(lons)]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # Layer 1: All edges (light gray base network)
    base_group = folium.FeatureGroup(name="Road network", show=True)
    for u, v, data in G.edges(data=True):
        coords = get_edge_coords(G, u, v)
        folium.PolyLine(
            coords, weight=1, color="#FFD3AC", opacity=0.4
        ).add_to(base_group)
    base_group.add_to(m)

    # Layer 2: Blocked edges (red dashed)
    blocked_group = folium.FeatureGroup(name="Blocked edges", show=True)
    num_blocked = 0
    for u, v, data in G_active.edges(data=True):
        if data.get("blocked", False):
            num_blocked += 1
            coords = get_edge_coords(G_active, u, v)
            hf = float(data.get("flood_score", G[u][v].get("flood_score", 0)))
            hl = float(data.get("landslide_score", G[u][v].get("landslide_score", 0)))
            tooltip = f"BLOCKED | flood={hf:.1f} land={hl:.1f}"
            folium.PolyLine(
                coords, weight=3, color="#E24B4A", opacity=0.7,
                dash_array="6 4", tooltip=tooltip
            ).add_to(blocked_group)
    blocked_group.add_to(m)

    # Layer 3: Hazardous but passable edges (orange, subtle)
    hazard_group = folium.FeatureGroup(name="Hazardous (passable)", show=False)
    for u, v, data in G_active.edges(data=True):
        if not data.get("blocked", False):
            hf = float(data.get("flood_score", G[u][v].get("flood_score", 0)))
            hl = float(data.get("landslide_score", G[u][v].get("landslide_score", 0)))
            if hf > 0 or hl > 0:
                coords = get_edge_coords(G_active, u, v)
                folium.PolyLine(
                    coords, weight=2, color="#EF9F27", opacity=0.4,
                    tooltip=f"Passable | flood={hf:.1f} land={hl:.1f}"
                ).add_to(hazard_group)
    hazard_group.add_to(m)

    # Layer 4: Algorithm routes
    offsets = {"NNA-Dijkstra": -0.0001, "NNA-A*": 0, "NNA-Dijkstra-HA": 0.0001}

    for algo_name, route in algo_results.items():
        color = ALGO_COLORS.get(algo_name, "#333333")
        route_group = folium.FeatureGroup(
            name=f"{algo_name} ({'success' if route.success else 'FAILED'})",
            show=True
        )

        if route.path and len(route.path) > 1:
            offset = offsets.get(algo_name, 0)
            route_coords = []
            for node in route.path:
                lat, lon = get_node_coords(G_active, node)
                route_coords.append((lat + offset, lon + offset))

            folium.PolyLine(
                route_coords, weight=4, color=color, opacity=0.85,
                tooltip=(
                    f"{algo_name} | "
                    f"{'Success' if route.success else 'FAILED: ' + route.failure_reason} | "
                    f"Time={route.total_time:.1f}m | "
                    f"Hazard={route.hazard_exposure:.1f}"
                )
            ).add_to(route_group)

            # Mark failure point if route failed
            if not route.success and len(route.path) > 0:
                last_node = route.path[-1]
                lat, lon = get_node_coords(G_active, last_node)
                folium.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(html=(
                        f'<div style="font-size:18px;color:#E24B4A;font-weight:bold;'
                        f'text-shadow:1px 1px 2px white">X</div>'
                    )),
                    tooltip=f"{algo_name} failed here: {route.failure_reason}"
                ).add_to(route_group)

        route_group.add_to(m)

    # Layer 5: Start + delivery markers (on top)
    marker_group = folium.FeatureGroup(name="Start + deliveries", show=True)

    lat, lon = get_node_coords(G, start_node)
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip=f"START ({start_node})"
    ).add_to(marker_group)

    for i, d_node in enumerate(delivery_nodes):
        lat, lon = get_node_coords(G, d_node)
        visited = d_node in (algo_results.get("NNA-Dijkstra-HA") or algo_results.get("NNA-Dijkstra", type("", (), {"visit_order": []}))).visit_order
        color = "blue" if visited else "red"
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(color=color, icon="gift", prefix="fa"),
            tooltip=f"Delivery {i+1} ({d_node})"
        ).add_to(marker_group)

    marker_group.add_to(m)

    # Title box
    ri_desc = RAIN_PARAMS[rain_level]["description"]
    title_html = f"""
    <div style="position:fixed;top:10px;left:60px;z-index:9999;
         background:white;padding:12px 16px;border-radius:8px;
         border:1px solid #ccc;font-family:sans-serif;font-size:13px;
         box-shadow:0 2px 6px rgba(0,0,0,0.15)">
        <b>Episode {episode_num}</b> | RI{rain_level} ({ri_desc}) |
        Blocked: {num_blocked}/{G.number_of_edges()} ({100*num_blocked/G.number_of_edges():.1f}%)<br>
        Deliveries: {len(delivery_nodes)} |
        Start: {start_node}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    filename = f"episode_{episode_num:03d}_ri{rain_level}.html"
    filepath = Path(output_dir) / filename
    m.save(str(filepath))
    return filepath


def run_visual_episodes(
    G, num_episodes=5, num_deliveries=5,
    target_ri=None, seed=42, output_dir="episode_maps"
):
    """Run a small number of episodes and generate maps for each."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    nodes = list(G.nodes())

    algorithms = {
        "NNA-Dijkstra": nna_dijkstra,
        "NNA-A*": nna_astar,
        "NNA-Dijkstra-HA": nna_dijkstra_ha,
    }

    generated = 0
    attempt = 0
    max_attempts = num_episodes * 100

    print(f"\nGenerating {num_episodes} episode maps...")
    print(f"  Deliveries: {num_deliveries}")
    print(f"  Target RI: {'all' if target_ri is None else f'RI{target_ri}'}")
    print(f"  Output: {output_dir}/\n")

    while generated < num_episodes and attempt < max_attempts:
        attempt += 1

        if target_ri is not None:
            rain_level = target_ri
        else:
            rain_level = rng.randint(1, 5)

        random.seed(seed + attempt)
        G_active = activate_hazards(G, rain_level)
        G_passable = get_passable_graph(G_active)

        # Try to find a feasible scenario (allow infeasible too for visualization)
        selected = rng.sample(nodes, num_deliveries + 1)
        start_node = selected[0]
        delivery_nodes = selected[1:]

        is_feasible = check_feasibility(G_passable, start_node, delivery_nodes)

        # Run algorithms
        algo_results = {}
        for name, fn in algorithms.items():
            algo_results[name] = fn(G_active, start_node, delivery_nodes)

        # Generate map
        filepath = build_episode_map(
            G, G_active, rain_level, start_node, delivery_nodes,
            algo_results, generated + 1, output_dir
        )

        status = "FEASIBLE" if is_feasible else "INFEASIBLE"
        num_blocked = sum(1 for _, _, d in G_active.edges(data=True) if d.get("blocked"))
        success_str = " | ".join(
            f"{n}: {'ok' if r.success else r.failure_reason}"
            for n, r in algo_results.items()
        )

        print(f"  Map {generated+1}: RI{rain_level} | {status} | "
              f"Blocked={num_blocked} | {success_str}")
        print(f"    -> {filepath}")

        generated += 1

    print(f"\nDone. {generated} maps saved to {output_dir}/")
    print("Open the .html files in a browser to explore.")


def main():
    parser = argparse.ArgumentParser(description="Visualize Monte Carlo episodes")
    parser.add_argument("--graph", default="data/la_trinidad_hazard_graph.graphml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deliveries", type=int, default=5)
    parser.add_argument("--ri", type=int, default=None,
                        help="Fix rain level (1-5). Default: random.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="episode_maps")
    args = parser.parse_args()

    G = load_graph(args.graph)

    run_visual_episodes(
        G,
        num_episodes=args.episodes,
        num_deliveries=args.deliveries,
        target_ri=args.ri,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()