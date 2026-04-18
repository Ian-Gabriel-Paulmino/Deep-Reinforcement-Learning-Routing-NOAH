"""
Hazard-Aware Routing — Data Preprocessing (v3)

Builds the La Trinidad road network enriched with NOAH flood and landslide
hazard scores. Writes a graphml ready for RL training, baseline benchmarks,
and visualization.

Key differences from v1/v2 (both archived under ../../archive/):

  * Landslide mapping is the NOAH-aligned canonical-3 scheme
    {1: 0.2, 2: 0.6, 3: 1.0}, identical to the flood mapping.
  * Edge attributes now include `flood_class` and `landslide_class` (raw class
    per edge) and `sample_count`, useful for debugging and reanalysis.
  * Writes a machine-readable `preprocessing_manifest.json` with every score
    distribution, blocking preview, and mapping snapshot.
  * Uses the `logging` module with both console and file handlers; no bare
    print() calls.
  * Defensive assertions on score ranges and class values.

Usage:
    python -m src.data.prepare_data                # standard run
    python -m src.data.prepare_data --debug        # verbose logging
    python -m src.data.prepare_data --no-viz       # skip visualization hand-off
    python -m src.data.prepare_data --cache-osm    # reuse cached OSM graph

Output files (relative to project root):
    data/la_trinidad_hazard_graph.graphml
    data/preprocessing_manifest.json
    data/preprocessing_v3.log

References:
    HAZARD_VALUES_DIRECTION_2026-04-17.md (project root)
    PIPELINE_V3_GUIDE.md (project root)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================


def _immutable(m: dict) -> Mapping:
    """Return a read-only view of a dict so Config stays effectively frozen."""
    return MappingProxyType(dict(m))


CANONICAL_FLOOD_SCORES: Mapping[int, float] = _immutable({1: 0.2, 2: 0.6, 3: 1.0})
CANONICAL_LANDSLIDE_SCORES: Mapping[int, float] = _immutable({1: 0.2, 2: 0.6, 3: 1.0})


@dataclass(frozen=True)
class Config:
    place_name: str = "La Trinidad, Benguet, Philippines"
    network_type: str = "drive"
    base_speed_kph: float = 30.0

    flood_shapefile: str = "data/raw/noah/flood_hazard/Benguet_Flood_25year.shp"
    landslide_shapefile: str = "data/raw/noah/landslide_susceptibility/Benguet_LandslideHazards.shp"
    flood_column: str = "Var"
    landslide_column: str = "LH"

    flood_scores: Mapping[int, float] = field(default_factory=lambda: CANONICAL_FLOOD_SCORES)
    landslide_scores: Mapping[int, float] = field(default_factory=lambda: CANONICAL_LANDSLIDE_SCORES)

    sample_interval_meters: float = 10.0

    osm_cache_path: str = "cache/osm_raw_la_trinidad.graphml"
    output_graphml: str = "data/la_trinidad_hazard_graph.graphml"
    output_manifest: str = "data/preprocessing_manifest.json"
    output_log: str = "data/preprocessing_v3.log"

    pipeline_version: str = "v3"
    pipeline_description: str = (
        "Canonical-3 landslide mapping; parallels flood; NOAH-aligned 3-class scheme"
    )


# Per-RI deterministic thresholds used by the RL framework's *_det.json configs.
# Included here for the blocking preview only; the benchmarks and training
# scripts still own the runtime copy of these numbers.
DET_RI_THRESHOLDS: Mapping[int, Mapping[str, float]] = _immutable({
    1: {"speed_mult": 0.94, "flood_block_threshold": 1.1, "landslide_block_threshold": 1.1},
    2: {"speed_mult": 0.90, "flood_block_threshold": 1.0, "landslide_block_threshold": 1.1},
    3: {"speed_mult": 0.85, "flood_block_threshold": 0.6, "landslide_block_threshold": 1.1},
    4: {"speed_mult": 0.40, "flood_block_threshold": 0.6, "landslide_block_threshold": 1.0},
    5: {"speed_mult": 0.20, "flood_block_threshold": 0.6, "landslide_block_threshold": 0.6},
})


# =============================================================================
# LOGGING
# =============================================================================


def configure_logging(log_path: Path, debug: bool = False) -> logging.Logger:
    """Set up a module logger with console + file handlers. Called once from main()."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prepare_data")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
    detailed_fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    stream.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(detailed_fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def log_header(logger: logging.Logger, text: str) -> None:
    logger.info("=" * 72)
    logger.info(text)
    logger.info("=" * 72)


def log_step(logger: logging.Logger, step_num: int, text: str) -> None:
    logger.info("")
    logger.info(f"[Step {step_num}] {text}")
    logger.info("-" * 72)


# =============================================================================
# ROAD NETWORK EXTRACTION
# =============================================================================


def download_road_network(cfg: Config, logger: logging.Logger, use_cache: bool) -> nx.MultiDiGraph:
    """Download (or reuse cached) La Trinidad road network via OSMnx."""
    log_step(logger, 1, "Download road network from OpenStreetMap")
    logger.info(f"  Place:             {cfg.place_name}")
    logger.info(f"  Network type:      {cfg.network_type}")
    logger.info(f"  Base speed:        {cfg.base_speed_kph} km/h")

    cache_path = Path(cfg.osm_cache_path)

    if use_cache and cache_path.exists():
        logger.info(f"  Using cached OSM graph: {cache_path}")
        G = ox.load_graphml(filepath=cache_path)
    else:
        t0 = time.perf_counter()
        G = ox.graph_from_place(
            cfg.place_name, network_type=cfg.network_type, simplify=True
        )
        logger.info(f"  OSM download complete in {time.perf_counter() - t0:.1f}s")
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(G, filepath=cache_path)
            logger.info(f"  Cached OSM graph to {cache_path}")

    # Ensure every edge has geometry + base_time (minutes under normal conditions)
    for u, v, _, data in G.edges(keys=True, data=True):
        if "geometry" not in data or data["geometry"] is None:
            u_xy = (G.nodes[u]["x"], G.nodes[u]["y"])
            v_xy = (G.nodes[v]["x"], G.nodes[v]["y"])
            data["geometry"] = LineString([u_xy, v_xy])

        length_m = float(data.get("length", 0.0))
        data["base_time"] = (length_m / 1000.0) / cfg.base_speed_kph * 60.0  # minutes

    total_km = sum(float(d.get("length", 0.0)) for _, _, d in G.edges(data=True)) / 1000.0
    logger.info(f"  Nodes:             {G.number_of_nodes():,}")
    logger.info(f"  Edges:             {G.number_of_edges():,}")
    logger.info(f"  Total road length: {total_km:.1f} km")
    return G


# =============================================================================
# HAZARD DATA LOADING
# =============================================================================


@dataclass(frozen=True)
class HazardData:
    name: str
    polygons: gpd.GeoDataFrame
    spatial_index: STRtree
    geometries: tuple
    column: str
    scores: Mapping[int, float]
    classes_present: tuple
    polygon_count: int
    source_path: str


def load_hazard_shapefile(
    path_str: str, name: str, column: str, scores: Mapping[int, float], logger: logging.Logger
) -> Optional[HazardData]:
    """Load a NOAH shapefile; build an R-tree spatial index."""
    path = Path(path_str)
    if not path.exists():
        logger.error(f"  Missing shapefile for {name}: {path}")
        return None

    logger.info(f"  Loading {name} shapefile: {path}")
    gdf = gpd.read_file(path)
    logger.info(f"    {len(gdf):,} polygons loaded")

    if gdf.crs is None:
        logger.warning("    CRS not declared — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs) != "EPSG:4326":
        logger.info(f"    Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    if column not in gdf.columns:
        logger.error(
            f"    Column {column!r} not found; available: {list(gdf.columns)}"
        )
        return None

    classes_present = sorted({int(v) for v in gdf[column].dropna().tolist()})
    logger.info(f"    Classes present: {classes_present}")
    for c in classes_present:
        count = int((gdf[column].astype(int) == c).sum())
        mapped = scores.get(c)
        if mapped is None:
            logger.warning(
                f"      class {c}: {count} polygons -> UNMAPPED (score will be 0.0)"
            )
        else:
            logger.info(f"      class {c}: {count} polygons -> score {mapped}")

    geometries = tuple(gdf.geometry)
    spatial_index = STRtree(list(geometries))

    return HazardData(
        name=name,
        polygons=gdf,
        spatial_index=spatial_index,
        geometries=geometries,
        column=column,
        scores=scores,
        classes_present=tuple(classes_present),
        polygon_count=len(gdf),
        source_path=str(path),
    )


def load_all_hazard_data(cfg: Config, logger: logging.Logger) -> tuple[Optional[HazardData], Optional[HazardData]]:
    log_step(logger, 2, "Load NOAH hazard shapefiles")
    flood = load_hazard_shapefile(
        cfg.flood_shapefile, "flood", cfg.flood_column, cfg.flood_scores, logger
    )
    landslide = load_hazard_shapefile(
        cfg.landslide_shapefile, "landslide", cfg.landslide_column, cfg.landslide_scores, logger
    )
    return flood, landslide


# =============================================================================
# SPATIAL OVERLAY
# =============================================================================


def sample_points_along_edge(geometry: LineString, interval_m: float) -> list[Point]:
    """Place evenly-spaced sample points along a LineString, min 3 samples."""
    coords = list(geometry.coords)
    total_length_m = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat_m = (lat2 - lat1) * 111_000.0
        dlon_m = (lon2 - lon1) * 111_000.0 * np.cos(np.radians((lat1 + lat2) / 2.0))
        total_length_m += float(np.sqrt(dlat_m**2 + dlon_m**2))

    num_samples = max(3, int(total_length_m / interval_m) + 1)
    return [
        geometry.interpolate(i / (num_samples - 1) if num_samples > 1 else 0.5, normalized=True)
        for i in range(num_samples)
    ]


def query_hazard_at_point(point: Point, hazard: Optional[HazardData]) -> tuple[int, float]:
    """
    Return (raw_class, score) at a point. raw_class == 0 means "no hazard here".
    If a point falls in multiple polygons, pick the higher class (max score).
    """
    if hazard is None:
        return 0, 0.0

    candidate_indices = hazard.spatial_index.query(point)
    best_class = 0
    best_score = 0.0
    for idx in candidate_indices:
        if hazard.geometries[idx].contains(point):
            try:
                raw = int(hazard.polygons.iloc[idx][hazard.column])
            except (ValueError, TypeError):
                continue
            score = float(hazard.scores.get(raw, 0.0))
            if score > best_score:
                best_score = score
                best_class = raw
    return best_class, best_score


def enrich_graph_with_hazards(
    G: nx.MultiDiGraph,
    flood: Optional[HazardData],
    landslide: Optional[HazardData],
    cfg: Config,
    logger: logging.Logger,
) -> nx.MultiDiGraph:
    """Overlay hazard scores onto every edge via 10-meter sampling + max aggregation (Eq. 3.1)."""
    log_step(logger, 3, "Overlay hazard data on road network")
    logger.info(f"  Sampling interval: {cfg.sample_interval_meters:.1f} m")

    total_edges = G.number_of_edges()
    total_samples = 0
    stats = {"flood_nonzero": 0, "landslide_nonzero": 0, "any_nonzero": 0}
    t0 = time.perf_counter()

    for _, _, _, data in tqdm(
        G.edges(keys=True, data=True), total=total_edges, desc="  Overlay edges", leave=False
    ):
        geometry = data.get("geometry")
        if geometry is None:
            data["flood_score"] = 0.0
            data["landslide_score"] = 0.0
            data["flood_class"] = 0
            data["landslide_class"] = 0
            data["sample_count"] = 0
            continue

        pts = sample_points_along_edge(geometry, cfg.sample_interval_meters)
        total_samples += len(pts)

        # Query each sample for flood and landslide class + score
        f_best_class, f_best_score = 0, 0.0
        l_best_class, l_best_score = 0, 0.0
        for p in pts:
            fc, fs = query_hazard_at_point(p, flood)
            lc, ls = query_hazard_at_point(p, landslide)
            if fs > f_best_score:
                f_best_class, f_best_score = fc, fs
            if ls > l_best_score:
                l_best_class, l_best_score = lc, ls

        # Defensive checks
        assert 0.0 <= f_best_score <= 1.0, f"Bad flood_score: {f_best_score}"
        assert 0.0 <= l_best_score <= 1.0, f"Bad landslide_score: {l_best_score}"

        data["flood_score"] = f_best_score
        data["landslide_score"] = l_best_score
        data["flood_class"] = f_best_class
        data["landslide_class"] = l_best_class
        data["sample_count"] = len(pts)

        if f_best_score > 0:
            stats["flood_nonzero"] += 1
        if l_best_score > 0:
            stats["landslide_nonzero"] += 1
        if f_best_score > 0 or l_best_score > 0:
            stats["any_nonzero"] += 1

    elapsed = time.perf_counter() - t0
    logger.info("")
    logger.info(f"  Overlay complete in {elapsed:.1f}s")
    logger.info(f"  Total samples queried:      {total_samples:,}")
    logger.info(
        f"  Edges with flood hazard:    "
        f"{stats['flood_nonzero']:>4} ({100.0*stats['flood_nonzero']/total_edges:.1f}%)"
    )
    logger.info(
        f"  Edges with landslide:       "
        f"{stats['landslide_nonzero']:>4} ({100.0*stats['landslide_nonzero']/total_edges:.1f}%)"
    )
    logger.info(
        f"  Edges with any hazard:      "
        f"{stats['any_nonzero']:>4} ({100.0*stats['any_nonzero']/total_edges:.1f}%)"
    )

    return G


# =============================================================================
# SAVE GRAPHML + MANIFEST
# =============================================================================


def save_graphml(
    G_enriched: nx.MultiDiGraph, cfg: Config, logger: logging.Logger
) -> tuple[nx.DiGraph, int]:
    """
    Collapse MultiDiGraph -> DiGraph (keep first edge per pair),
    strip non-serializable attrs, write graphml.
    Returns (DiGraph, parallel_edges_collapsed_count).
    """
    log_step(logger, 4, "Save graph as GraphML")

    G_out = nx.DiGraph()

    # Graph-level metadata
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    G_out.graph.update({
        "pipeline_version": cfg.pipeline_version,
        "pipeline_description": cfg.pipeline_description,
        "preprocessing_date": now,
        "flood_mapping_json": json.dumps({str(k): v for k, v in cfg.flood_scores.items()}),
        "landslide_mapping_json": json.dumps(
            {str(k): v for k, v in cfg.landslide_scores.items()}
        ),
        "sample_interval_meters": cfg.sample_interval_meters,
        "base_speed_kph": cfg.base_speed_kph,
        "noah_flood_source": str(Path(cfg.flood_shapefile).name),
        "noah_landslide_source": str(Path(cfg.landslide_shapefile).name),
    })

    # Nodes: canonical x, y, pos
    for node, data in G_enriched.nodes(data=True):
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        G_out.add_node(node, x=x, y=y, pos=f"{x},{y}")

    # Edges: canonical flat attribute set; collapse parallel edges, keep first
    parallel_collapsed = 0
    for u, v, data in G_enriched.edges(data=True):
        if G_out.has_edge(u, v):
            parallel_collapsed += 1
            continue
        length_m = float(data.get("length", 0.0))
        G_out.add_edge(
            u,
            v,
            flood_score=float(data.get("flood_score", 0.0)),
            landslide_score=float(data.get("landslide_score", 0.0)),
            flood_class=int(data.get("flood_class", 0)),
            landslide_class=int(data.get("landslide_class", 0)),
            sample_count=int(data.get("sample_count", 0)),
            base_time=float(data.get("base_time", 0.0)),
            length=length_m,
        )

    out_path = Path(cfg.output_graphml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G_out, str(out_path))

    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    logger.info(f"  Output:                     {out_path}")
    logger.info(f"  Format:                     graphml (DiGraph)")
    logger.info(f"  Nodes:                      {G_out.number_of_nodes():,}")
    logger.info(
        f"  Edges:                      {G_out.number_of_edges():,} "
        f"({parallel_collapsed} parallel collapsed)"
    )
    logger.info(f"  File size:                  {size_mb:.2f} MB")
    return G_out, parallel_collapsed


# =============================================================================
# VERIFICATION + ANALYSIS + MANIFEST
# =============================================================================


def _score_distribution(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0, "nonzero_count": 0, "nonzero_pct": 0.0, "by_value": {}}
    uniq, counts = np.unique(np.round(arr, 4), return_counts=True)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "nonzero_count": int((arr > 0).sum()),
        "nonzero_pct": float(100.0 * (arr > 0).sum() / arr.size),
        "by_value": {f"{v:.2f}": int(c) for v, c in zip(uniq, counts)},
    }


def verify_and_analyze(
    G: nx.DiGraph, cfg: Config, logger: logging.Logger
) -> dict:
    """
    Check attribute presence, compute score distributions, build blocking preview,
    and return a manifest-ready dict.
    """
    log_step(logger, 5, "Verify graph + compute score distributions")

    # Attribute presence check
    sample = next(iter(G.edges(data=True)))[2]
    required = ("flood_score", "landslide_score", "flood_class", "landslide_class",
                "sample_count", "base_time", "length")
    for attr in required:
        status = "ok" if attr in sample else "MISSING"
        logger.info(f"  edge attr {attr:<18} {status}")

    flood_scores = [float(d["flood_score"]) for _, _, d in G.edges(data=True)]
    landslide_scores = [float(d["landslide_score"]) for _, _, d in G.edges(data=True)]
    flood_dist = _score_distribution(flood_scores)
    ls_dist = _score_distribution(landslide_scores)

    logger.info("")
    logger.info(
        f"  flood scores:      mean={flood_dist['mean']:.3f}  max={flood_dist['max']:.2f}  "
        f"nonzero={flood_dist['nonzero_count']}/{flood_dist['count']} ({flood_dist['nonzero_pct']:.1f}%)"
    )
    logger.info(f"    distribution:    {flood_dist['by_value']}")
    logger.info(
        f"  landslide scores:  mean={ls_dist['mean']:.3f}  max={ls_dist['max']:.2f}  "
        f"nonzero={ls_dist['nonzero_count']}/{ls_dist['count']} ({ls_dist['nonzero_pct']:.1f}%)"
    )
    logger.info(f"    distribution:    {ls_dist['by_value']}")

    # Blocking preview under the deterministic RI thresholds
    logger.info("")
    logger.info(
        "  Blocking preview (deterministic RI thresholds; eligible = score >= threshold):"
    )
    logger.info(
        f"    {'RI':<4} {'theta_f':<8} {'theta_l':<8} {'Flood elig':>10} "
        f"{'LS elig':>10} {'Union':>10}"
    )
    n_edges = len(flood_scores)
    blocking_preview = {}
    for ri in (1, 2, 3, 4, 5):
        params = DET_RI_THRESHOLDS[ri]
        tf = params["flood_block_threshold"]
        tl = params["landslide_block_threshold"]
        f_elig = sum(1 for s in flood_scores if s >= tf)
        l_elig = sum(1 for s in landslide_scores if s >= tl)
        union = sum(
            1 for fs, ls in zip(flood_scores, landslide_scores) if fs >= tf or ls >= tl
        )
        logger.info(
            f"    RI{ri:<3} {tf:<8.2f} {tl:<8.2f} {f_elig:>10} {l_elig:>10} {union:>10}"
        )
        blocking_preview[f"RI{ri}"] = {
            "flood_threshold": tf,
            "landslide_threshold": tl,
            "speed_mult": params["speed_mult"],
            "flood_eligible": f_elig,
            "landslide_eligible": l_elig,
            "union_eligible": union,
            "flood_eligible_pct": 100.0 * f_elig / n_edges,
            "landslide_eligible_pct": 100.0 * l_elig / n_edges,
            "union_eligible_pct": 100.0 * union / n_edges,
        }

    # Connectivity
    sccs = list(nx.strongly_connected_components(G))
    largest_scc = max(sccs, key=len)
    logger.info("")
    logger.info(
        f"  Connectivity: {len(sccs)} SCCs; largest has {len(largest_scc):,} nodes "
        f"({100.0*len(largest_scc)/G.number_of_nodes():.1f}% of graph)"
    )
    logger.info(f"  Edge / node ratio: {G.number_of_edges() / G.number_of_nodes():.2f}")

    return {
        "flood_distribution": flood_dist,
        "landslide_distribution": ls_dist,
        "blocking_preview": blocking_preview,
        "scc_count": len(sccs),
        "largest_scc_nodes": len(largest_scc),
    }


def write_manifest(
    cfg: Config,
    G: nx.DiGraph,
    flood: Optional[HazardData],
    landslide: Optional[HazardData],
    analysis: dict,
    parallel_collapsed: int,
    logger: logging.Logger,
) -> None:
    """Write the preprocessing manifest JSON sidecar."""
    log_step(logger, 6, "Write preprocessing manifest")

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    manifest = {
        "pipeline_version": cfg.pipeline_version,
        "pipeline_description": cfg.pipeline_description,
        "preprocessing_date": now,
        "study_area": cfg.place_name,
        "base_speed_kph": cfg.base_speed_kph,
        "sample_interval_meters": cfg.sample_interval_meters,
        "mappings": {
            "flood": {str(k): v for k, v in cfg.flood_scores.items()},
            "landslide": {str(k): v for k, v in cfg.landslide_scores.items()},
        },
        "noah_sources": {
            "flood": {
                "path": flood.source_path if flood else cfg.flood_shapefile,
                "polygon_count": flood.polygon_count if flood else 0,
                "classes_present": list(flood.classes_present) if flood else [],
            },
            "landslide": {
                "path": landslide.source_path if landslide else cfg.landslide_shapefile,
                "polygon_count": landslide.polygon_count if landslide else 0,
                "classes_present": list(landslide.classes_present) if landslide else [],
            },
        },
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "edge_node_ratio": G.number_of_edges() / max(1, G.number_of_nodes()),
            "scc_count": analysis["scc_count"],
            "largest_scc_nodes": analysis["largest_scc_nodes"],
            "parallel_edges_collapsed": parallel_collapsed,
        },
        "score_distributions": {
            "flood": analysis["flood_distribution"],
            "landslide": analysis["landslide_distribution"],
        },
        "blocking_preview": analysis["blocking_preview"],
        "outputs": {
            "graphml": str(Path(cfg.output_graphml)),
            "log": str(Path(cfg.output_log)),
        },
    }

    out = Path(cfg.output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Wrote: {out}")


# =============================================================================
# MAIN
# =============================================================================


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess the La Trinidad hazard graph (v3, canonical-3 mapping)"
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument(
        "--no-viz", action="store_true", help="skip the post-run visualization hand-off"
    )
    parser.add_argument(
        "--cache-osm", action="store_true", help="reuse cached OSM graph if present"
    )
    args = parser.parse_args(argv)

    cfg = Config()
    logger = configure_logging(Path(cfg.output_log), debug=args.debug)

    log_header(logger, f"HAZARD-AWARE ROUTING — DATA PREPROCESSING ({cfg.pipeline_version})")
    logger.info(f"  Description:   {cfg.pipeline_description}")
    logger.info(f"  Output graphml: {cfg.output_graphml}")
    logger.info(f"  Manifest:       {cfg.output_manifest}")
    logger.info(f"  Log:            {cfg.output_log}")

    t0 = time.perf_counter()
    try:
        G = download_road_network(cfg, logger, use_cache=args.cache_osm)
        flood, landslide = load_all_hazard_data(cfg, logger)
        G = enrich_graph_with_hazards(G, flood, landslide, cfg, logger)
        G_out, parallel_collapsed = save_graphml(G, cfg, logger)
        analysis = verify_and_analyze(G_out, cfg, logger)
        write_manifest(cfg, G_out, flood, landslide, analysis, parallel_collapsed, logger)
    except Exception as exc:
        logger.exception(f"Preprocessing failed: {exc}")
        return 1

    elapsed = time.perf_counter() - t0
    log_header(logger, f"PREPROCESSING COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Ready:   {cfg.output_graphml}")

    if not args.no_viz:
        logger.info("")
        logger.info("  Next: python -m src.data.visualize_hazards")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
