# Benguet Flood and Landslide Data

Preprocessing for the BSCS thesis
**Hazard-Aware DRL Routing for La Trinidad, Benguet**
(Janiola, Paulmino, Lluch).

This sub-project turns raw data into a hazard-enriched road network graph
that the sibling repos consume. Its scope is **preprocessing only**:
produce the GraphML hazard graph that the REST API consumes. As of
2026-05-11 the fair-evaluation harness was vendored out to
`web/Hazard-Aware-Routing-REST-API/`, and as of 2026-05-16 the leftover
`src/benchmarks/monte_carlo.py` + `src/benchmarks/viz_algos.py` baseline
scripts (deprecated since 2026-04-18) and their migration helpers in
`scripts/_archive/` have been removed. For benchmarking, live inference,
and the fair-evaluation harness, see the API repo.

## What goes in

- **Road network:** downloaded from OpenStreetMap via OSMnx (place:
  La Trinidad, Benguet, Philippines).
- **NOAH flood hazard:** 3 polygons in
  `data/raw/noah/flood_hazard/Benguet_Flood_25year.shp`, classified by
  depth (1=Low 0-0.5m, 2=Moderate 0.5-1.5m, 3=High >1.5m).
- **NOAH landslide susceptibility:** 3 polygons in
  `data/raw/noah/landslide_susceptibility/Benguet_LandslideHazards.shp`,
  classified as 1=Low/Yellow, 2=Moderate/Orange, 3=High/Red (no-dwelling).

## What comes out

- **Hazard graph (graphml)** — `data/la_trinidad_hazard_graph.graphml`.
  1,447 nodes, 3,112 directed edges. Each edge carries `flood_score`,
  `landslide_score`, `flood_class`, `landslide_class`, `sample_count`,
  `base_time`, `length`.
- **Machine-readable manifest** — `data/preprocessing_manifest.json`.
  Full run snapshot (mappings used, score distributions, blocking preview,
  connectivity). Read this programmatically; don't hardcode stats.
- **Structured log** — `data/preprocessing_v3.log`.
- **Interactive Folium map** — `data/la_trinidad_hazard_map_v3.html`. Six
  toggleable layers: hazard-colored roads, NOAH flood polygons, NOAH
  landslide polygons, per-RI blocked-edge layers (RI1..RI5), 10m
  sample-point markers on five curated edges.

## Quick start

```bash
# From this directory, install dependencies (uv; Python 3.13+ per pyproject.toml)
uv sync

# Regenerate the hazard graph (~10 min; dominated by the overlay step)
python -m src.data.prepare_data

# Optional: reuse cached OSM graph for faster reruns
python -m src.data.prepare_data --cache-osm

# Render the Folium map (~30 s)
python -m src.data.visualize_hazards
# open data/la_trinidad_hazard_map_v3.html in a browser
```

## Hazard score mapping (v3, canonical-3, NOAH-aligned)

| Class | Flood `Var` | Landslide `LH` | NOAH description |
|-------|-------------|----------------|------------------|
| 1 | 0.2 | 0.2 | Low / Yellow |
| 2 | 0.6 | 0.6 | Moderate / Orange |
| 3 | 1.0 | 1.0 | High / Red (no-dwelling) |

Per-edge score is the maximum across 10m samples along the road geometry
(Equation 3.1, "weakest-link").

## Where to read next

- **`../CLAUDE.md`** (workspace root) — monorepo overview: sub-project map,
  canonical vs deprecated boundaries, locked thesis direction.
- **`../MACRO_DDQN_IMPLEMENTATION_2026-05-14.md`** — end-to-end reference
  for the macro-level DDQN system that consumes this folder's graph.
- **`../HANDOFF_2026-05-14_visualization.md`** — sequenced upgrade plan
  for the REST API + visualisation tool.
- **`../final_thesis_masterguide.md`** — locked manuscript stance.
- **`archive/README.md`** — why v1 and v2 outputs are archived (gitignored
  but present on disk for historical reference).

## Technology

- Python 3.13+ with `uv` for package management (per `pyproject.toml`).
- OSMnx + NetworkX for road network extraction and graph manipulation.
- GeoPandas + Shapely (STRtree spatial index) for NOAH polygon overlay.
- Folium for interactive visualization.
- `logging` module for structured logging; JSON sidecar for machine-readable
  run metadata.
