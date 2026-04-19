# Benguet Flood and Landslide Data

Preprocessing + baseline benchmarking for the BSCS thesis
**Hazard-Aware DRL Routing for La Trinidad, Benguet**
(Janiola, Paulmino, Lluch).

This sub-project turns raw data into a hazard-enriched road network graph
that the sibling `RL_framework/` project consumes. It also owns the
**fair-evaluation harness** (`src/evaluation/`, see its README) that
benchmarks the DQN against classical NNA baselines on a shared, committed
cohort of scenarios. The older, fairness-unsafe
`src/benchmarks/monte_carlo.py` is retained for reproducibility only and
has been marked deprecated.

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

# Fair-evaluation harness (thesis-reportable path):
# Stage 1 — generate a committed cohort of scenarios
python -m src.evaluation.scenario_generator \
    --graph data/staged_subgraphs/selected_subgraph_n200.graphml \
    --graph-id la_trinidad_subgraph_n200 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_mini \
    --num-scenarios 100 --num-deliveries 5 --master-seed 42

# Stage 2 — run policies against that cohort
python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra

# Stage 3 — aggregate metrics
python -m src.evaluation.evaluator --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

See `src/evaluation/README.md` for the full contract (schemas, algorithm
notes, fairness argument, extension patterns).

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
