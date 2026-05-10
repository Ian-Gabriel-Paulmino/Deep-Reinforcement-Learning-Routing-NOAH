# Benguet Flood and Landslide Data

Preprocessing + baseline benchmarking for the BSCS thesis
**Hazard-Aware DRL Routing for La Trinidad, Benguet**
(Janiola, Paulmino, Lluch).

This sub-project turns raw data into a hazard-enriched road network graph
that the sibling repos consume. As of 2026-05-11, the **fair-evaluation
harness** that used to live at `src/evaluation/` has been vendored into
`web/Hazard-Aware-Routing-REST-API/src/evaluation/` — that repo is now
the canonical source-of-truth for benchmarking and live inference. This
sub-project's scope is now narrower: produce the GraphML hazard graph
that the API consumes. The older, fairness-unsafe
`src/benchmarks/monte_carlo.py` is retained for pre-2026-04-18
reproducibility only and is marked deprecated.

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
# From this directory, install dependencies (uv; Python 3.10+)
uv sync

# Regenerate the hazard graph (~10 min; dominated by the overlay step)
python -m src.data.prepare_data

# Optional: reuse cached OSM graph for faster reruns
python -m src.data.prepare_data --cache-osm

# Render the Folium map (~30 s)
python -m src.data.visualize_hazards
# open data/la_trinidad_hazard_map_v3.html in a browser

# Legacy baseline (DEPRECATED — kept for pre-2026-04-18 reproducibility only):
python -m src.benchmarks.monte_carlo --episodes 300 --deliveries 5
```

For benchmarking, live inference, and the fair-evaluation harness, see
the API repo at `web/Hazard-Aware-Routing-REST-API/`. As of 2026-05-11
the entire `src/evaluation/` tree was vendored there; this sub-project
no longer owns benchmarking. See `Thesis/HANDOFF_2026-05-11.md` for the
post-vendor architecture summary.

## Hazard score mapping (v3, canonical-3, NOAH-aligned)

| Class | Flood `Var` | Landslide `LH` | NOAH description |
|-------|-------------|----------------|------------------|
| 1 | 0.2 | 0.2 | Low / Yellow |
| 2 | 0.6 | 0.6 | Moderate / Orange |
| 3 | 1.0 | 1.0 | High / Red (no-dwelling) |

Per-edge score is the maximum across 10m samples along the road geometry
(Equation 3.1, "weakest-link").

## Where to read next

- **`../PIPELINE_V3_GUIDE.md`** (workspace root) — team-oriented walk-through
  of the pipeline, conflicts, and open decisions. Start here if you've been
  away from the project for a while.
- **`../CHANGES_V3_2026-04-17.md`** — one-page summary of what moved where in
  the v3 refactor.
- **`../HAZARD_VALUES_DIRECTION_2026-04-17.md`** — the investigation report
  that motivated v3 (NOAH 3-class confirmation, mapping comparison, DQN
  impact analysis).
- **`CLAUDE.md`** — project overview for Claude Code agents.
- **`docs/findings.md`** — authoritative pipeline reference (keep in step with
  the manifest).
- **`docs/modelling.md`** — detailed trace from NOAH rasters to edge scores
  to travel-time modelling.
- **`archive/README.md`** — why v1 and v2 are archived and what each folder
  contains.

## Technology

- Python 3.10+ with `uv` for package management.
- OSMnx + NetworkX for road network extraction and graph manipulation.
- GeoPandas + Shapely (STRtree spatial index) for NOAH polygon overlay.
- Folium for interactive visualization.
- `logging` module for structured logging; JSON sidecar for machine-readable
  run metadata.
