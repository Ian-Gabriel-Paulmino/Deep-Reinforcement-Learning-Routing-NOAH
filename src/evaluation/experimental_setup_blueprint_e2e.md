# Experimental Setup Blueprint — End-to-End

> A single-source guide for running the Fair Evaluation Harness across every
> experimental configuration the thesis needs, plus manual verification
> checklists and troubleshooting. Companion to `README.md` — the README
> explains *why*, this file explains *what to run, in what order, and what
> to expect back*.

---

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [Prerequisites](#2-prerequisites)
3. [Stage 1 — Scenario Generation Setups](#3-stage-1--scenario-generation-setups)
4. [Stage 2 — Policy Execution Setups](#4-stage-2--policy-execution-setups)
5. [Stage 3 — Evaluation Setups](#5-stage-3--evaluation-setups)
6. [Manual Verification Checklists](#6-manual-verification-checklists)
7. [Data-Gathering Workflow for the Thesis](#7-data-gathering-workflow-for-the-thesis)
8. [Known Caveats & Interpretation Warnings](#8-known-caveats--interpretation-warnings)
9. [Scaling & Code-Cleaning Notes](#9-scaling--code-cleaning-notes)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Mental Model

The harness separates three concerns that were previously entangled in
`monte_carlo.py`:

```
Stage 1 (scenario_generator)    — DEFINE the exam questions
   ↓ cohort.json + scenarios.jsonl
Stage 2 (run_policies)          — ASK each policy the questions
   ↓ routes/<algo>.jsonl
Stage 3 (evaluator)             — SCORE the answers
   ↓ report/metrics.json
```

Two properties fall out of this split:

- **Reproducibility**: Stage 1 is the only place RNG lives; given the same
  `master_seed` + graph + config, you get byte-identical scenarios.jsonl.
- **Incremental re-use**: adding a new DQN variant reruns Stage 2 for that
  variant only. Redefining a metric reruns Stage 3 only. No refitting, no
  regeneration.

Every experiment in this guide follows the same pattern: **generate once,
run any policies you want against it, evaluate any metrics you want on
the routes**.

---

## 2. Prerequisites

### One Python venv (self-contained)

A single uv-managed venv under
`Benguet Flood and Landslide Data/.venv` covers every stage. Install
with the `dqn` extra so torch is available for DQN runs:

```bash
cd "Benguet Flood and Landslide Data"
uv sync --extra dqn
```

| Subset | Base deps | Extra |
|---|---|---|
| Scenario gen + NNA runners + evaluator | networkx, numpy | — |
| DQN runner (`DQN@*` algorithms) | — (above, plus) | `--extra dqn` (torch) |

The harness lazy-imports torch inside `run_policies._make_dqn_runner`,
so NNA-only workflows run fine without `--extra dqn`.

### Bundled RL backend + checkpoints

The DQN runner depends only on files inside this project — no external
clone:

```
Benguet Flood and Landslide Data/
├── src/evaluation/rl_backend/
│   ├── rl_routing_wCUDA_wCheckP.py      (inference + env module)
│   └── utils/graph_utils.py             (helper imported by the above)
└── models/rl_checkpoints/
    ├── balanced_HF/
    │   ├── stage_200_balanced_HF_RI1_det/best_model.pt
    │   ├── stage_200_balanced_HF_RI2_det/best_model.pt
    │   ├── stage_200_balanced_HF_RI3_det/best_model.pt
    │   ├── stage_200_balanced_HF_RI4_det/best_model.pt
    │   └── stage_200_balanced_HF_RI5_det/best_model.pt
    ├── fast_HF/       (same 5-RI layout)
    └── safe_HF/       (same 5-RI layout)
```

No hardcoded cross-repo paths remain. See
[§9](#9-scaling--code-cleaning-notes) for notes on retraining /
swapping the bundled checkpoints.

### Configs

Cohort generation reads `hazard.rain_levels` and `hazard.flood_time_weight`
/ `hazard.landslide_time_weight` (α values, manuscript §B) from any
`_det.json` config under:

```
src/evaluation/configs/hazard_training_final/{balanced,fast,safe}_HF/
  ├── stage_200_{profile}_HF_RI2_det.json
  ├── stage_200_{profile}_HF_RI3_det.json
  ├── stage_200_{profile}_HF_RI4_det.json
  └── stage_200_{profile}_HF_RI5_det.json
```

Any of these works for Stage 1 — they share the same `hazard.rain_levels`
structure. The DQN runner also uses these for model architecture config
(it picks the RI3 config of the profile by default; same arch across RIs
within a profile).

### Terminal working directory

All commands assume you are in:

```bash
cd "Benguet Flood and Landslide Data"
```

Python is invoked via the Benguet venv (created by `uv sync --extra dqn`):

```bash
PY="uv run python"       # preferred — works on all platforms
# or, if you've already activated the venv manually:
# PY=".venv/Scripts/python.exe"       # Windows
# PY=".venv/bin/python"                # Linux/macOS
```

All subsequent commands use `$PY` (adjust based on your shell).

---

## 3. Stage 1 — Scenario Generation Setups

### 3.1 Smoke test — `la_trinidad_mini` (default)

**Purpose:** fast sanity check after any harness change. 100 scenarios on
the 200-node subgraph. Useful for iterating on runners/metrics without
waiting.

```bash
$PY -m src.evaluation.scenario_generator \
    --graph data/staged_subgraphs/selected_subgraph_n200.graphml \
    --graph-id la_trinidad_subgraph_n200 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_mini \
    --num-scenarios 100 \
    --num-deliveries 5 \
    --master-seed 42 \
    --max-steps 220
```

**Expected wall-clock:** ~1 second.

**Expected output (stdout):**

```
17:21:32  INFO     Loading graph: data/staged_subgraphs/selected_subgraph_n200.graphml
17:21:33  INFO       nodes=200 edges=417
17:21:33  INFO       travel-time weights: alpha_flood=0.5 (config), alpha_landslide=0.5 (config)
17:21:33  INFO       RI1: blocked 0/417 edges (0.0%); largest SCC = 200/200 nodes
17:21:33  INFO       RI2: blocked 18/417 edges (4.3%); largest SCC = 189/200 nodes
17:21:33  INFO       RI3: blocked 54/417 edges (12.9%); largest SCC = 96/200 nodes
17:21:33  INFO       RI4: blocked 159/417 edges (38.1%); largest SCC = 68/200 nodes
17:21:33  INFO       RI5: blocked 277/417 edges (66.4%); largest SCC = 10/200 nodes
17:21:33  INFO       wrote 100 scenarios to ...\cohorts\la_trinidad_mini\scenarios.jsonl
17:21:33  INFO     Cohort generation took 1.0s
```

**Expected files on disk:**

```
src/evaluation/cohorts/la_trinidad_mini/
├── cohort.json          (~350 bytes, metadata)
└── scenarios.jsonl      (~800 KB, 100 lines, one scenario per line)
```

**Things to look for in the log:**
- `alpha_flood=0.5 (config)` — confirms α sourcing is working. If this
  says `(default)`, your config is missing `hazard.flood_time_weight`.
- Blocked-edge counts increase monotonically with RI (0 → 18 → 54 → 159 → 277).
- SCC sizes decrease monotonically with RI. RI5 collapses to ~10 nodes
  on this subgraph — expected; it's why the 200-node graph is only for
  smoke tests.

### 3.2 Canonical thesis cohort — `la_trinidad_v1`

**Purpose:** the numbers you'll report in the manuscript. 2500 scenarios
on the full 1447-node hazard graph.

```bash
$PY -m src.evaluation.scenario_generator \
    --graph data/la_trinidad_hazard_graph.graphml \
    --graph-id la_trinidad_full \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_v1 \
    --num-scenarios 2500 \
    --num-deliveries 5 \
    --master-seed 42 \
    --max-steps 500
```

**Expected wall-clock:** ~30–60 seconds (dominated by per-RI SCC
computation on the full graph).

**Expected file sizes:**
- `cohort.json`: ~400 bytes
- `scenarios.jsonl`: ~50 MB (2500 scenarios × ~20 KB each, including
  the travel_time_map)

**Notes:**
- `--max-steps 500` is larger than the smoke test's 220 because a
  full-graph path can plausibly exceed 220 edges.
- 500 scenarios per RI × 5 RIs = 2500. Clean division, no remainder bias.

**Skip-to-verify:** after generation, immediately open
`cohort.json` — `ri_distribution` should be
`{"RI1": 500, "RI2": 500, "RI3": 500, "RI4": 500, "RI5": 500}`.

### 3.3 num_deliveries sensitivity sweep

**Purpose:** if you ever retrain the DQN at `num_deliveries=5` (closing
the current OOD gap — see §8), you'll want a cohort matching training.

```bash
# num_deliveries=2 (matches the bundled checkpoints' training run)
$PY -m src.evaluation.scenario_generator \
    --graph data/staged_subgraphs/selected_subgraph_n200.graphml \
    --graph-id la_trinidad_subgraph_n200 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_mini_nd2 \
    --num-scenarios 100 --num-deliveries 2 --master-seed 42 --max-steps 220

# num_deliveries=10 (stress test — long-horizon routing)
$PY -m src.evaluation.scenario_generator \
    --graph data/la_trinidad_hazard_graph.graphml \
    --graph-id la_trinidad_full \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_full_nd10 \
    --num-scenarios 500 --num-deliveries 10 --master-seed 42 --max-steps 800
```

**Risk note:** a high `--num-deliveries` on the small subgraph will fail
Stage 1's SCC pre-check at high RIs (you need at least
`num_deliveries+1` nodes in the RI5 SCC, and the n=200 subgraph only has
~10 passable nodes at RI5). If you get `RuntimeError` mentioning "largest
SCC has X nodes but scenario requires ...", pick a denser graph or a
smaller num_deliveries.

### 3.4 RI-subset cohort (single-RI stress test)

**Purpose:** isolate a single RI when debugging a policy's failure mode.

```bash
# RI5-only cohort (all extreme rainfall)
$PY -m src.evaluation.scenario_generator \
    --graph data/la_trinidad_hazard_graph.graphml \
    --graph-id la_trinidad_full \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_v1_ri5only \
    --num-scenarios 500 \
    --num-deliveries 5 \
    --master-seed 42 \
    --max-steps 500 \
    --ri-keys RI5
```

The `--ri-keys` flag accepts any subset. Useful combinations:
- `--ri-keys RI1` — dry baseline
- `--ri-keys RI4 RI5` — extreme only
- `--ri-keys RI1 RI3 RI5` — coarse sweep

### 3.5 Probabilistic-mode legacy cohort

**Purpose:** reproduce pre-2026-04-18 comparisons. Not recommended for
thesis — use deterministic_v3 as the canonical mode.

```bash
$PY -m src.evaluation.scenario_generator \
    --graph data/la_trinidad_hazard_graph.graphml \
    --graph-id la_trinidad_full \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_v1_probabilistic \
    --num-scenarios 500 --num-deliveries 5 --master-seed 42 \
    --activation-mode probabilistic_v1
```

### 3.6 α-weight override (sensitivity analysis)

**Purpose:** if you want to probe how travel-time drag calibration
(α_f, α_l) affects DQN-vs-baseline ordering.

```bash
# Explicit α override (overrides config's flood_time_weight/landslide_time_weight)
$PY -m src.evaluation.scenario_generator \
    --graph data/staged_subgraphs/selected_subgraph_n200.graphml \
    --graph-id la_trinidad_subgraph_n200 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_mini_alpha_high \
    --num-scenarios 100 --num-deliveries 5 --master-seed 42 \
    --alpha-flood 0.8 --alpha-landslide 0.6
```

Expect the log to now print:
`travel-time weights: alpha_flood=0.8 (override), alpha_landslide=0.6 (override)`.

This only affects `travel_time_map` values in the scenarios. Blocked-edge
set and SCC structure are α-independent.

---

## 4. Stage 2 — Policy Execution Setups

All commands assume a cohort exists at
`src/evaluation/cohorts/<cohort_id>/` from Stage 1.

### 4.1 Baselines-only (no torch required)

**Purpose:** fastest feedback loop when iterating on NNA runners,
robustness metric, or the evaluator.

```bash
$PY -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra NNA-AStar NNA-Dijkstra-HA
```

**Expected wall-clock:** ~1.5s on 100 scenarios (all three algos combined).

**Expected output (stdout):**

```
17:26:26  INFO     Cohort: la_trinidad_mini  (100 scenarios)
17:26:26  INFO     Running NNA-Dijkstra (sha256:7e58c93e37baa189)
17:26:27  INFO       NNA-Dijkstra: 100 routes in 0.58s -> .../routes/NNA-Dijkstra.jsonl
17:26:27  INFO     Running NNA-AStar (sha256:202ce52852d241a3)
17:26:27  INFO       NNA-AStar: 100 routes in 0.46s -> .../routes/NNA-AStar.jsonl
17:26:27  INFO     Running NNA-Dijkstra-HA (sha256:36e98801ee83d548)
17:26:28  INFO       NNA-Dijkstra-HA: 100 routes in 0.67s -> .../routes/NNA-Dijkstra-HA.jsonl
```

**Expected files:**

```
src/evaluation/cohorts/la_trinidad_mini/routes/
├── NNA-Dijkstra.jsonl         (100 lines, one route each, ~1 MB)
├── NNA-AStar.jsonl            (~1 MB)
└── NNA-Dijkstra-HA.jsonl      (~1 MB)
```

### 4.2 Single DQN profile (iterative dev)

**Purpose:** when debugging the DQN runner, run one profile to isolate
issues without waiting for all three.

```bash
$PY -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms DQN@balanced_HF
```

**Expected wall-clock:** ~3.5s on 100 scenarios (includes lazy-loading
up to 5 checkpoints, ~0.6s each).

**Expected output (stdout):**

```
17:43:25  INFO     Running DQN@balanced_HF (sha256:ecbe159ec53c2b41)
17:43:25  INFO       [DQN@balanced_HF] 1/100
17:43:27  INFO       [DQN@balanced_HF] 50/100
17:43:28  INFO       [DQN@balanced_HF] 100/100
17:43:28  INFO       DQN@balanced_HF: 100 routes in 3.28s -> .../routes/DQN@balanced_HF.jsonl
```

### 4.3 Full 6-method comparison (the thesis run)

**Purpose:** produce the complete route set for the manuscript's
comparison tables.

```bash
$PY -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra NNA-AStar NNA-Dijkstra-HA \
                  DQN@balanced_HF DQN@fast_HF DQN@safe_HF
```

**Expected wall-clock on la_trinidad_mini (100 scenarios):** ~13s total.
On `la_trinidad_v1` (2500 scenarios full graph): ~6–10 minutes (DQN
dominates; each DQN profile takes ~2 minutes at 2500 scenarios).

**Expected files:**

```
src/evaluation/cohorts/la_trinidad_mini/routes/
├── DQN@balanced_HF.jsonl
├── DQN@fast_HF.jsonl
├── DQN@safe_HF.jsonl
├── NNA-AStar.jsonl
├── NNA-Dijkstra.jsonl
└── NNA-Dijkstra-HA.jsonl
```

### 4.4 Re-run only a specific algorithm

Running Stage 2 with the same `--algorithms` flag **overwrites** the
existing `<algo>.jsonl` file — other algos' files are untouched. So to
re-run a single algo without disturbing the rest:

```bash
# Overwrite DQN@balanced_HF only (after tweaking the runner)
$PY -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms DQN@balanced_HF
```

Other `routes/*.jsonl` files remain, so Stage 3 still sees the full
6-method set.

### 4.5 Per-profile DQN ablation

**Purpose:** isolate which profile handles which RI best. Useful for the
"which profile per deployment context" discussion in the thesis.

```bash
for profile in balanced_HF fast_HF safe_HF; do
    $PY -m src.evaluation.run_policies \
        --cohort-dir src/evaluation/cohorts/la_trinidad_v1 \
        --algorithms DQN@${profile}
done
```

---

## 5. Stage 3 — Evaluation Setups

### 5.1 Full evaluation

**Purpose:** produce the report JSON + console summary after Stage 2.

```bash
$PY -m src.evaluation.evaluator \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

**Expected wall-clock:** <1 second on 600 total routes (100 × 6 algos).

**Expected console summary (abbreviated — from the smoke test in this
session):**

```
========================================================================
Cohort: la_trinidad_mini  (100 scenarios)
Graph:  la_trinidad_subgraph_n200
Mode:   deterministic_v3
========================================================================

  DQN@balanced_HF  (scored 100)
    success_rate     =  88.00%  (100 episodes)
    travel_time(min) = mean=21.3  std=12.6  over 88 successful
    hazard_exposure  = mean=1858.12  std=1663.01  over 88 successful
    replan_count     = mean=0.00  max=0
    failures: timeout=12
    by RI:   RI1= 95.0%  RI2= 75.0%  RI3= 90.0%  RI4= 85.0%  RI5= 95.0%
    robustness:  success=0.915  travel_time=0.609  hazard_exposure=0.282

  DQN@fast_HF  (scored 100)
    success_rate     =  87.00%  (100 episodes)
    ...

  NNA-Dijkstra-HA  (scored 100)
    success_rate     = 100.00%  (100 episodes)
    travel_time(min) = mean=21.0  std=11.1  over 100 successful
    hazard_exposure  = mean=1784.54  std=1522.70  over 100 successful
    replan_count     = mean=0.00  max=0
    by RI:   RI1=100.0%  RI2=100.0%  RI3=100.0%  RI4=100.0%  RI5=100.0%
    robustness:  success=1.000  travel_time=0.587  hazard_exposure=0.269

  ... (3 more algorithms)
```

**Expected files:**

```
src/evaluation/cohorts/la_trinidad_mini/report/
└── metrics.json    (~30 KB, full per-(algorithm, RI) aggregates)
```

### 5.2 Metric-only rerun (add a new metric, skip policies)

**Purpose:** the whole point of the 3-stage split. Add a new metric file
under `metrics/<name>.py`, register it in `metrics/__init__.py::REGISTRY`,
and rerun Stage 3 only. No regeneration, no policy re-execution.

```bash
# After editing metrics/__init__.py to add your new metric
$PY -m src.evaluation.evaluator \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

Stage 3 reads `routes/*.jsonl` as-is. New metric appears in
`report/metrics.json` automatically.

### 5.3 Report schema (for downstream analysis)

`report/metrics.json` top-level shape:

```json
{
  "cohort_id": "la_trinidad_mini",
  "num_scenarios": 100,
  "graph_id": "la_trinidad_subgraph_n200",
  "activation_mode": "deterministic_v3",
  "algorithms": {
    "DQN@balanced_HF": {
      "routes_file": "routes/DQN@balanced_HF.jsonl",
      "routes_scored": 100,
      "failure_counts": {"timeout": 12},
      "replan_count_stats": {"n": 100, "mean": 0.0, ...},
      "wall_time_ms_stats": {"n": 100, "mean": 30.4, ...},
      "metrics": {
        "success":         {"all": {...}, "RI1": {...}, ...},
        "travel_time":     {"all": {...}, "RI1": {...}, ...},
        "hazard_exposure": {"all": {...}, "RI1": {...}, ...}
      },
      "robustness": {
        "success": 0.915,
        "travel_time": 0.609,
        "hazard_exposure": 0.282
      }
    },
    "NNA-Dijkstra": {...},
    ...
  }
}
```

Each `_safe_stats` entry has keys: `n, mean, stdev, min, max`. NaN values
filtered out of aggregates.

---

## 6. Manual Verification Checklists

Run these after every major change. Catches most regressions.

### 6.1 After Stage 1

- [ ] Log shows `travel-time weights: alpha_flood=0.5 (config), ...`.
      If `(default)`, your config is missing the time-weight keys.
- [ ] `ri_distribution` in `cohort.json` is balanced per RI (e.g.,
      20/20/20/20/20 for 100 scenarios, 500/500/500/500/500 for 2500).
- [ ] Blocked-edge counts in log increase monotonically with RI.
- [ ] Largest SCC sizes in log decrease monotonically with RI.
- [ ] Pick a random line in `scenarios.jsonl`. Verify it has keys
      `scenario_id, rain_level, start_node, delivery_nodes,
      blocked_edges, travel_time_map, max_steps, num_deliveries,
      metadata`.
- [ ] A scenario with `rain_level=1` should have `blocked_edges: []`
      (on the det-v3 config where RI1 blocks nothing).
- [ ] `metadata.sample_attempts_for_this_ri` should be 1 for almost every
      scenario. If you see values >5, the SCC sampling is hitting
      infeasibility more than expected — investigate.

### 6.2 After Stage 2

- [ ] One `<algorithm_id>.jsonl` per algorithm passed. File count equals
      number of algorithms.
- [ ] Each route file has exactly `cohort.num_scenarios` lines.
- [ ] Pick a random route. Verify:
  - `algorithm_id` matches the filename (sans `.jsonl`).
  - `algorithm_config_hash` starts with `sha256:`.
  - `per_edge` length equals `len(edge_sequence)`.
  - `per_edge[0].step == 0`, `per_edge[i].step == i`.
  - `visit_order` is a subset of `scenario.delivery_nodes`.
  - On `success=true`: `visit_order` covers all `delivery_nodes`.
  - On `success=false`: `failure_reason` is one of `trapped, timeout,
    invalid_action, no_route`.
- [ ] NNA-Dijkstra and NNA-AStar should produce **identical** metrics
      on cohorts without block-replan activity — A*'s admissible
      heuristic finds the same shortest paths as Dijkstra.
- [ ] NNA-Dijkstra-HA's `replan_count` should be 0 everywhere (oracle
      doesn't replan).
- [ ] DQN runners' `replan_count` should be 0 everywhere (action mask,
      not replan).

### 6.3 After Stage 3

- [ ] 6 entries under `report["algorithms"]` for the full thesis run.
- [ ] Each algorithm has `metrics` keys: `success, travel_time,
      hazard_exposure`. Each has `all` + 5 RI keys.
- [ ] `robustness` dict has 3 numbers (one per metric). `success`
      robustness near 1.0 for algorithms with 100% success at every RI.
- [ ] Ordering check (on the full canonical cohort — small subgraph
      can invert due to SCC collapse):
  - `NNA-Dijkstra-HA.travel_time.mean ≤ DQN.travel_time.mean ≤
    NNA-Dijkstra.travel_time.mean` (oracle lower bound).
  - `DQN.hazard_exposure.mean < NNA-Dijkstra.hazard_exposure.mean` at
    RI≥3 (DQN should avoid hazardous edges; baselines are hazard-blind).
- [ ] Success rates at RI1 are approximately in the ballpark of the
      per-profile reference CSV (`rl_profiles_200n_ri1_overall.csv` if
      you're on the n=200 subgraph — see the training run's artifacts).

If any of these checks fails unexpectedly, see [§10 Troubleshooting](#10-troubleshooting).

---

## 7. Data-Gathering Workflow for the Thesis

Plan your experiment runs so each thesis claim is backed by a specific
`report/metrics.json`.

### 7.1 Primary comparison (§3.5 of the manuscript)

```bash
# 1. Generate the canonical cohort
$PY -m src.evaluation.scenario_generator \
    --graph data/la_trinidad_hazard_graph.graphml \
    --graph-id la_trinidad_full \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_v1 \
    --num-scenarios 2500 --num-deliveries 5 --master-seed 42 --max-steps 500

# 2. Run all 6 methods
$PY -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_v1 \
    --algorithms NNA-Dijkstra NNA-AStar NNA-Dijkstra-HA \
                  DQN@balanced_HF DQN@fast_HF DQN@safe_HF

# 3. Evaluate
$PY -m src.evaluation.evaluator \
    --cohort-dir src/evaluation/cohorts/la_trinidad_v1
```

**Manuscript artifacts from this run:**
- Main comparison table (success_rate, travel_time, hazard_exposure per
  algorithm × RI) → `report/metrics.json::algorithms.*.metrics`.
- Robustness table → `report/metrics.json::algorithms.*.robustness`.
- Caption: "All 6 methods evaluated on the same 2500-scenario cohort
  generated with master_seed=42."

### 7.2 Subgraph validation (§3.5.x or appendix)

Rerun the full comparison on `la_trinidad_mini` to show the numbers are
consistent across scale. Use the same `--master-seed` so the scenarios
overlap where the subgraph permits.

### 7.3 Per-RI deep dive (§3.6 individual RI analysis)

The main `report/metrics.json` already has per-RI stats. For specific
prose about "what happens at RI5", pull the `metrics.success.RI5` (and
analogous) cells from each algorithm.

If you want a narrower cohort isolated to a single RI:

```bash
# RI5-only cohort for deep RI5 analysis
$PY -m src.evaluation.scenario_generator \
    --graph data/la_trinidad_hazard_graph.graphml \
    --graph-id la_trinidad_full \
    --cohort-id la_trinidad_v1_ri5 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --num-scenarios 500 --num-deliveries 5 --master-seed 42 \
    --max-steps 500 --ri-keys RI5
# ... then Stage 2 + Stage 3 as normal on this cohort.
```

### 7.4 Reproducibility appendix

Paste the exact commands into the thesis appendix. Readers can rerun
with the same seed and get byte-identical scenarios.jsonl. Stage 2 and
Stage 3 output will match within machine-precision floats.

### 7.5 Computing additional statistics

`report/metrics.json` is the authoritative input for any downstream
analysis. For significance tests, confidence intervals, or Pareto plots:

```python
import json
from pathlib import Path

report = json.loads(
    Path("src/evaluation/cohorts/la_trinidad_v1/report/metrics.json").read_text()
)

for algo_id, info in report["algorithms"].items():
    m = info["metrics"]
    print(algo_id, m["success"]["all"]["mean"],
          m["travel_time"]["all"]["mean"],
          m["hazard_exposure"]["all"]["mean"])
```

Per-scenario data is in `routes/<algo>.jsonl` — load line-by-line for
any custom cross-scenario analysis (e.g., "DQN vs NNA-Dijkstra paired
comparison").

---

## 8. Known Caveats & Interpretation Warnings

### 8.1 num_deliveries=2 training OOD

Teammate's checkpoints were trained at `num_deliveries=2`; our canonical
cohorts use 5. The DQN's MLP is count-invariant (masked-mean pool), so
checkpoint loading is safe — but pooled-embedding statistics shift
early in each episode when 5 unvisited deliveries are pooled instead of
at-most-2. Empirical cost on `la_trinidad_mini`: **12–22% timeout rate**
across the three DQN profiles vs 0% for baselines.

**Mitigation path:** retrain DQN at `num_deliveries=5`. Out of scope for
the runner session; documented in README §10.

**When writing the thesis:** add a footnote under §3.5 acknowledging
the OOD tax and quantifying it from the smoke-test numbers.

### 8.2 Success-conditional averaging bias

`travel_time` and `hazard_exposure` means are taken over **successful
routes only** (per manuscript §3.6.1 B). If Algorithm A succeeds on
100% of scenarios and Algorithm B on 78%, B's mean is over a
potentially-easier sub-cohort — the 22% it failed on were presumably
the hard ones.

**Concrete example from this session's smoke test:**
`DQN@safe_HF` has the lowest travel_time (20.9) AND lowest
hazard_exposure (1682) — but also the lowest success rate (78%). Don't
conclude "safe_HF is best" from those averages alone.

**Mitigation:** always report success rate AND the conditional averages
together. Ideally add Pareto plots (deferred to `visualization.py`).

### 8.3 SCC collapse on the subgraph at high RI

At RI5, `la_trinidad_subgraph_n200` collapses to a 10-node SCC. Every
scenario at RI5 draws 5 deliveries + 1 start from this 10-node pool —
many scenarios share overlapping starts/deliveries.

**Consequence:** RI5 stats on the subgraph are NOT a representative
sample of "full-graph RI5 performance." Only the full-graph cohort
(`la_trinidad_v1`) should be cited for RI5 conclusions in the
manuscript.

### 8.4 Probabilistic-mode is legacy

`--activation-mode probabilistic_v1` exists for pre-2026-04-18
reproducibility only. The thesis's locked mode is `deterministic_v3`
(per CLAUDE.md invariants). Don't mix modes within a single
comparison table.

### 8.5 α-weight fix is post-hoc

Any cohort generated before this session's α fix (commit introducing
`DEFAULT_ALPHA_FLOOD = 0.5`) used `w_f=0.6, w_l=0.4` in the
`travel_time_map` — *wrong* weights per manuscript §B. Regenerate any
such cohort before citing its numbers. The smoke-test §8 transcript in
`README.md` is flagged as pre-α-fix and should not be used for
reference post-session.

---

## 9. Scaling & Code-Cleaning Notes

### 9.1 Standalone install (no external repos)

The DQN runner is fully self-contained:

| Asset | In-project location |
|---|---|
| Inference / env module | `src/evaluation/rl_backend/rl_routing_wCUDA_wCheckP.py` |
| Graph helper | `src/evaluation/rl_backend/utils/graph_utils.py` |
| Re-export adapter | `src/evaluation/runners/_rl_backend.py` |
| DQN checkpoint dir | `models/rl_checkpoints/{profile}/stage_200_{profile}_RI{1..5}_det/best_model.pt` |

There is no `sys.path` manipulation and no dependency on a sibling
clone. A fresh environment boots up with:

```bash
cd "Benguet Flood and Landslide Data"
uv sync --extra dqn
uv run python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms DQN@balanced_HF DQN@fast_HF DQN@safe_HF
```

### 9.2 Upstream-merge of the vendored backend

`src/evaluation/rl_backend/rl_routing_wCUDA_wCheckP.py` is a **vendored
snapshot**. If upstream training code (in whatever repo it lives) adds
a behavior we need, re-copy the file and update the snapshot notes in
`src/evaluation/rl_backend/README.md`. Keep edits out of the vendored
file itself; patches we want go into `runners/_rl_backend.py` or
`runners/dqn.py`.

The one local edit already applied: the vendored file uses
`from .utils.graph_utils import ...` (relative) with a fallback to the
original absolute form, so it resolves cleanly inside the package
without needing `sys.path` shenanigans.

### 9.3 Checkpoint-graph vs cohort-graph matching

The DQN runner builds a `str → int` node mapping by matching
`(x, y)` positions between the cohort graphml and the env's graph
(loaded from `ckpt["base_graph_node_link"]`). This assumes:
- Both graphs have ~identical nodes with the same coordinates.
- Coordinate precision is at least 6 decimal places.

If a future retrain uses a graph derived differently (e.g., different
BFS seed in `to_training_graph`), position matching may miss some
nodes and raise `RuntimeError: OSM node X at (x,y) has no matching
position in the checkpoint's graph`. **Recovery:** regenerate the
cohort with a graphml derived from the same OSM snapshot used for
training, or retrain against the current graph and replace the
bundled checkpoints.

### 9.4 When/how to add a new DQN profile

Say a fourth profile `eco_HF` is trained. To add it:

1. Drop checkpoints at
   `models/rl_checkpoints/eco_HF/stage_200_eco_HF_RI{1..5}_det/best_model.pt`
   (15 files total — one per RI).
2. Copy an existing RI3 config to
   `src/evaluation/configs/hazard_training_final/eco_HF/stage_200_eco_HF_RI3_det.json`
   (used for arch only).
3. In `run_policies.py`'s `POLICY_FACTORIES`, add:
   ```python
   "DQN@eco_HF": lambda: _make_dqn_runner("eco_HF"),
   ```
4. Invoke via `--algorithms DQN@eco_HF`.

No other file changes needed — the `_make_dqn_runner` helper handles
path construction from the profile name.

### 9.5 When/how to add a new metric

1. Create `metrics/<name>.py` with
   `def compute(scenario, route) -> float`.
2. Register in `metrics/__init__.py::REGISTRY`.
3. Rerun Stage 3 only.

For second-order metrics (like robustness — computed across per-RI
aggregates rather than per-route), follow the `robustness.py` pattern:
skip the REGISTRY, add a post-aggregation hook in `evaluator.py`.

### 9.6 When/how to add a new baseline

For hazard-blind variants (e.g., NNA-Greedy-BFS), create a new
`runners/<name>.py` that wraps `run_nna_with_fair_replan` with a
different `path_fn`. Two dozen LoC. See `runners/nna_astar.py` for a
template.

---

## 10. Troubleshooting

### 10.1 `ModuleNotFoundError: No module named 'rl_routing_wCUDA_wCheckP'`

The vendored backend isn't where the adapter expects. Verify the
snapshot is present:

```bash
ls src/evaluation/rl_backend/rl_routing_wCUDA_wCheckP.py
ls src/evaluation/rl_backend/utils/graph_utils.py
```

If missing, restore from the upstream source (see
`src/evaluation/rl_backend/README.md` for the snapshot policy) and
make sure `__init__.py` exists in both `rl_backend/` and
`rl_backend/utils/`.

### 10.2 `ModuleNotFoundError: No module named 'torch'`

You installed the venv without the `dqn` extra. Re-run:

```bash
uv sync --extra dqn
```

NNA-only workflows don't need this — torch is only required when a
`DQN@*` algorithm is passed to `run_policies`.

### 10.3 `RuntimeError: OSM node X at (x,y) has no matching position in the checkpoint's graph`

Cohort graph and checkpoint graph don't match. See §9.3. Regenerate
the cohort using the same graphml that produced the bundled
checkpoints, or replace the bundled checkpoints with a retrain on
your current graph.

### 10.4 `OSError: [Errno 22] Invalid argument: '...\NNA-A*.jsonl'`

Windows can't create files with `*` in the name. We already fixed this
by renaming the algorithm_id to `NNA-AStar`. If you see this error with
a new algorithm you added, pick a filesystem-safe `algorithm_id`
(avoid `* / \ : ? " < > |`).

### 10.5 All DQN scenarios fail with `failure_reason = "timeout"`

Expected OOD tax from num_deliveries=2 training at 5-delivery eval. If
the timeout rate is >50% (not just 10–20%), something worse is
happening. Check:
- Node-id mapping built correctly (see log for "OSM node ... at (x,y)
  has no matching" warnings at init).
- Scenario rain_level correctly passed into env.rain_onehot.
- Checkpoint path resolves to an actual file (not a different profile).

### 10.6 `RuntimeError: {RI} passable graph's largest SCC has N nodes but scenario requires start+M deliveries = M+1 nodes.`

Stage 1 pre-check caught an infeasible request. At high RI on sparse
graphs, the passable SCC collapses. Reduce `--num-deliveries` or use a
denser `--graph`.

### 10.7 Robustness scores are all `null`

Check that the cohort has at least 2 RIs (the formula needs ≥2 per-RI
means). Single-RI cohorts produce `null` robustness by design — it's
not a bug. If your cohort has 5 RIs but robustness is still null, the
aggregation is producing `mean: null` for all RIs (metric never applied
successfully) — check the routes for anomalies.

### 10.8 Travel-time weights log says `(default)` instead of `(config)`

Your `_det` config is missing `hazard.flood_time_weight` /
`hazard.landslide_time_weight`. Fix the config (or accept the default
0.5/0.5, which matches the empirical calibration locked in for the
manuscript).

---

## Related documents

- `README.md` — design rationale, schemas, fairness argument,
  algorithm deep dives.
- `CLAUDE.md` (parent repo) — top-level project overview and
  invariants.
- Thesis manuscript — §3.1.3 B (travel-time formula), §3.5
  (comparison methodology), §3.6 (metric definitions).
