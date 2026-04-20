# Fair Evaluation Harness — Technical README

> Purpose, architecture, algorithms, formulas, and operational guide for the
> evaluation harness under `Benguet Flood and Landslide Data/src/evaluation/`.
> Intended for thesis-team reviewers of the design and future Claude Code
> sessions extending it. (The harness originated in `RL_framework/evaluation/`
> on 2026-04-18 and was migrated to its current location shortly after.)

---

## 1. Why this exists

The thesis compares a learned Deep Q-Learning routing policy against three
classical baselines (NNA-Dijkstra, NNA-A\*, NNA-Dijkstra-HA oracle) under
flood/landslide-induced road blockages. Earlier the comparison was wired
through `Benguet Flood and Landslide Data/src/benchmarks/monte_carlo.py`.
Two independent fairness problems made those numbers untrustworthy:

**A. Structural asymmetry — block-avoidance capability, not policy quality.**
The DQN's environment (see
`../RL_framework/rl_routing_wCUDA_wCheckP_latest.py:593–599`) masks blocked
neighbors out of the valid action set and uses `q_values[mask == 0] = -1e9`
before `argmax` (line 755). The DQN *cannot* attempt a blocked edge — the
architecture filters them. Meanwhile the old `monte_carlo.py`
NNA baselines plan a path on the hazard-blind base graph; when execution
encounters a blocked edge, the whole episode fails (`_run_nna` line 328–
331, `failure_reason="execution_path_blocked"`). That isn't a difference in
*policy*, it's a difference in *ability to avoid blocks at all*. A fair
comparison must equalize this capability first, then measure the residual
policy gap.

**B. Uncontrolled scenario substrate.** The DQN eval loop and
`monte_carlo.py` each generate their own episodes. Even with the same seed,
different RNG paths, different retry logic, and different feasibility
filters mean the two pipelines run on non-identical worlds. Any observed
difference could be scenario-set bias rather than policy quality.

This harness addresses both: every algorithm runs against the **same
pre-committed cohort of scenarios**, with the **same shared environment
simulator**, with **matching block-avoidance capability**.

---

## 2. Architecture

Three stages, each a separately-runnable Python module:

```
Stage 1  scenario_generator.py  ─►  cohorts/<cohort_id>/cohort.json
                                    cohorts/<cohort_id>/scenarios.jsonl

Stage 2  run_policies.py        ─►  cohorts/<cohort_id>/routes/<algorithm_id>.jsonl
         (one run per algorithm, all reading the same scenarios.jsonl.
          Nine algorithms registered today: NNA-Dijkstra, NNA-AStar,
          NNA-Dijkstra-Blind, NNA-AStar-Blind, NNA-Dijkstra-HA,
          NNA-Dijkstra-HA-Blind, DQN@balanced_HF, DQN@fast_HF,
          DQN@safe_HF.)

Stage 3  evaluator.py           ─►  cohorts/<cohort_id>/report/metrics.json
                                    cohorts/<cohort_id>/report/raw_metrics.csv
                                    cohorts/<cohort_id>/report/overall_metrics.csv
         (reads all routes/*.jsonl; aggregates per (algorithm, RI);
          computes robustness post-aggregation; emits one JSON +
          two CSVs — raw per-episode and wide-format per (algo, RI).
          PNGs + HTMLs in future work.)
```

Each stage reads and writes plain JSONL. Adding a new DQN variant reruns
only Stage 2 for that variant. Redefining a metric reruns only Stage 3. No
policy re-execution is required to iterate on analysis.

> **Step-by-step commands, expected outputs, and manual verification
> checklists** for every stage live in
> [`experimental_setup_blueprint_e2e.md`](./experimental_setup_blueprint_e2e.md).
> That document is the operational companion to this design README.

### Directory layout

```
Benguet Flood and Landslide Data/src/evaluation/
├── __init__.py
├── README.md                              <-- this file (design & rationale)
├── experimental_setup_blueprint_e2e.md    <-- operational commands + verification
├── schemas.py                  Scenario, Route, Cohort, EdgeStep + JSONL I/O
├── scenario_generator.py       Stage 1 entry point
├── run_policies.py             Stage 2 entry point (9 algorithms wired)
├── evaluator.py                Stage 3 entry point (+ robustness + CSV outputs)
├── runners/
│   ├── __init__.py
│   ├── base.py                 GraphView, Policy protocol, fair-replan + blind loops
│   ├── nna.py                  NNA-Dijkstra (fair replan)
│   ├── nna_astar.py            NNA-AStar  (A* + fair replan)
│   ├── nna_blind.py            NNA-Dijkstra-Blind (Dijkstra plan, NO replan)
│   ├── nna_astar_blind.py      NNA-AStar-Blind    (A* plan, NO replan)
│   ├── nna_ha.py               NNA-Dijkstra-HA (hazard-aware oracle)
│   ├── nna_ha_blind.py         NNA-Dijkstra-HA-Blind (HA weights, block-blind, NO replan)
│   ├── _rl_backend.py          re-exports of the vendored RL backend
│   └── dqn.py                  DQNRunner (per-profile, per-RI dispatch)
├── rl_backend/                 vendored RL inference + env module
│   ├── __init__.py
│   ├── README.md               snapshot policy, architecture invariants
│   ├── rl_routing_wCUDA_wCheckP.py
│   └── utils/graph_utils.py
├── metrics/
│   ├── __init__.py             metric registry (7 metrics)
│   ├── success.py
│   ├── travel_time.py
│   ├── hazard_exposure.py      manuscript §3.6.1 C — reward-weighted
│   ├── hazard_score.py         raw, unweighted length × hazard sum
│   ├── steps.py                edge traversal count
│   ├── distance.py             meters walked
│   ├── runtime.py              wall-clock ms (defined even on failure)
│   └── robustness.py           post-aggregation only (not in REGISTRY)
├── configs/hazard_training_final/
│   ├── balanced_HF/stage_200_balanced_HF_RI{1..5}_det.json
│   ├── fast_HF/stage_200_fast_HF_RI{1..5}_det.json
│   └── safe_HF/stage_200_safe_HF_RI{1..5}_det.json
└── cohorts/
    └── <cohort_id>/
        ├── cohort.json
        ├── scenarios.jsonl
        ├── routes/<algo_id>.jsonl   (one file per algorithm)
        └── report/
            ├── metrics.json         aggregated JSON (for machines)
            ├── raw_metrics.csv      one row per (scenario, algo)
            └── overall_metrics.csv  one row per (algo, RI|all); wide format
```

Outside `src/evaluation/` but used at runtime:

```
Benguet Flood and Landslide Data/models/rl_checkpoints/
├── README.md                  15-checkpoint provenance / architecture
├── balanced_HF/stage_200_balanced_HF_RI{1..5}_det/best_model.pt
├── fast_HF/stage_200_fast_HF_RI{1..5}_det/best_model.pt
└── safe_HF/stage_200_safe_HF_RI{1..5}_det/best_model.pt
```

> **Historical artifact note.** `cohorts/*/routes/DQN@*.jsonl` files
> produced before the backend was vendored still have old
> `policy_metadata.checkpoint_root` paths baked in. That field is
> informational only; re-running Stage 2 overwrites those files with
> the new in-project paths. Not a bug.

---

## 3. Data contract (schemas)

All persistent artifacts are newline-delimited JSON. Edges are keyed
`"u|v"` (the pipe separator never appears in OSM node ids).

### `Cohort` (`cohort.json`)

One object describing a batch of scenarios.

| Field | Meaning |
|---|---|
| `cohort_id` | Unique string; directory name. |
| `generated_at` | ISO 8601 timestamp. |
| `master_seed` | Integer seed for the generator. All stochastic choices in Stage 1 are derived from this. |
| `graph_id` | Human-readable id for the base graph (e.g. `la_trinidad_subgraph_n200`). |
| `graph_path` | Path to the graphml file, relative to the cwd at Stage 2 time (i.e. the Benguet project root). |
| `num_scenarios` | Total scenarios (should match `len(scenarios.jsonl)`). |
| `sampling_policy` | Currently `stratified_by_RI`. |
| `ri_distribution` | Count of scenarios per RI key (`{"RI1": 20, ...}`). |
| `num_deliveries` | Number of delivery nodes per scenario (start is additional). |
| `activation_mode` | `deterministic_v3` or `probabilistic_v1`. |
| `feasibility_filtered` | Always `true` — infeasible tuples are dropped at generation time. |
| `scenarios_path` | Relative path to the scenarios JSONL (default `"scenarios.jsonl"`). |

### `Scenario` (one line in `scenarios.jsonl`)

One fully-specified exam question.

| Field | Meaning |
|---|---|
| `scenario_id` | Unique string, typically `"<cohort_id>_<6-digit-index>"`. |
| `graph_id` | Matches the cohort's `graph_id`. |
| `rain_level` | Integer 1–5. |
| `activation_mode` | Matches the cohort. |
| `activation_seed` | Used for probabilistic activation; unused in deterministic mode. |
| `start_node` | OSM node id (string). |
| `delivery_nodes` | Array of OSM node ids to visit. |
| `blocked_edges` | `[[u, v], ...]` — the edges that are blocked in this scenario. **Precomputed and persisted**, so no algorithm rolls dice. |
| `travel_time_map` | `{"u|v": minutes}` — effective travel time for every non-blocked edge, with the RI's speed multiplier and hazard penalty already applied. |
| `max_steps` | Maximum number of edge traversals before `timeout`. |
| `num_deliveries` | For convenience; matches `len(delivery_nodes)`. |
| `metadata.master_seed` | Passed through from the cohort. |
| `metadata.ri_key` | String form of `rain_level` (`"RI3"`). |
| `metadata.sample_attempts_for_this_ri` | How many draws were needed before this scenario was accepted (diagnostic). |

### `Route` (one line in `routes/<algorithm_id>.jsonl`)

The output of one algorithm on one scenario.

| Field | Meaning |
|---|---|
| `scenario_id` | Links back to the scenario. |
| `algorithm_id` | The runner's identifier. |
| `algorithm_config_hash` | SHA-256 of the policy's config, for reproducibility across reruns. |
| `visit_order` | Deliveries in the order the policy visited them. |
| `edge_sequence` | `[[u, v], ...]` of every edge traversed. |
| `per_edge` | Array of `EdgeStep` dicts — step index, was_replan flag, travel_time, hazard_flood, hazard_landslide, length_m. This is what the Stage-3 metrics read. |
| `success` | Boolean. |
| `failure_reason` | `null` on success; otherwise one of `trapped`, `timeout`, `invalid_action`, `no_route`, `blocked` (blind NNAs only — planned edge was in `blocked_edges`). |
| `replan_count` | Number of times the policy had to locally repair its plan (see §5 below). |
| `wall_time_ms` | Wall-clock time for this single scenario run. |
| `policy_metadata` | Free-form dict for policy-specific context (e.g. checkpoint step, eval mode). |

---

## 4. How to run

All commands assume `cd "Benguet Flood and Landslide Data"` with the venv
activated. Every path needed by the harness — the hazard graphs under
`data/`, the `_det` activation configs under `src/evaluation/configs/`, and
the committed cohorts under `src/evaluation/cohorts/` — is Benguet-local, so
the repo can be cloned and run without the sibling `RL_framework/` repo
being on disk.

### Stage 1 — generate a cohort

```bash
python -m src.evaluation.scenario_generator \
    --graph data/staged_subgraphs/selected_subgraph_n200.graphml \
    --graph-id la_trinidad_subgraph_n200 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_mini \
    --num-scenarios 100 \
    --num-deliveries 5 \
    --master-seed 42 \
    --max-steps 220
```

Notes:
- `--graph` is any graphml with edge attrs `flood_hazard` / `flood_score`,
  `landslide_hazard` / `landslide_score`, `length`, `base_time` /
  `travel_time_min`. Undirected graphs are symmetrized to a DiGraph on
  load. Use `data/la_trinidad_hazard_graph.graphml` for the full 1447-node
  network or `data/staged_subgraphs/selected_subgraph_n200.graphml` for the
  smoke-test subgraph.
- `--config` points at any training JSON that contains `hazard.rain_levels`
  (it's read for thresholds and speed multipliers only; the generator
  doesn't train anything). Any `_det.json` under
  `src/evaluation/configs/hazard_training_final/` works.
- Output lands at `src/evaluation/cohorts/<cohort-id>/`.

### Stage 2 — run policies

```bash
python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra NNA-AStar \
                  NNA-Dijkstra-Blind NNA-AStar-Blind \
                  NNA-Dijkstra-HA \
                  DQN@balanced_HF DQN@fast_HF DQN@safe_HF
```

Eight algorithms are registered today in `POLICY_FACTORIES`
(`run_policies.py`): the two replan-capable NNAs (`NNA-Dijkstra`,
`NNA-AStar`), the two **blind** NNAs (`NNA-Dijkstra-Blind`,
`NNA-AStar-Blind` — plan once, no replan, fail on first blocked
edge; see §5.1b), the hazard-aware oracle (`NNA-Dijkstra-HA`), and
three DQN profiles (`DQN@balanced_HF`, `DQN@fast_HF`,
`DQN@safe_HF`). Each DQN runner internally dispatches to the
RI-matched specialist checkpoint (see §5.4). Extension pattern:

- Add a new runner class under `runners/` implementing the `Policy`
  protocol in `runners/base.py`.
- Register it in `POLICY_FACTORIES` in `run_policies.py`.
- Pass its id on the command line.

Each algorithm produces a separate `routes/<algorithm_id>.jsonl` file.
Invoke from inside `RL_framework/`'s venv so torch is available for
the DQN runners (NNA-only subsets work without torch). See
[`experimental_setup_blueprint_e2e.md`](./experimental_setup_blueprint_e2e.md)
§4 for concrete command recipes per experiment type.

### Stage 3 — evaluate

```bash
python -m src.evaluation.evaluator --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

Reads every `routes/*.jsonl` in the cohort, applies every metric in
`metrics.REGISTRY`, aggregates by `(algorithm_id, RI)`, and writes
three artifacts into `report/`:

- `metrics.json` — nested JSON mirror of the aggregation, indent-2
  formatted. Machine-readable.
- `raw_metrics.csv` — one row per `(scenario_id, algorithm_id)`. Fixed
  prefix columns (`scenario_id`, `RI`, `algorithm_id`, `failure_reason`,
  `replan_count`) + one column per registered metric. Failed-episode
  metric columns are blank (except `runtime`, which is always defined).
  This is the file to load into pandas / Excel for ad-hoc analysis.
- `overall_metrics.csv` — wide-format aggregated stats, one row per
  `(algorithm_id, bucket)` where bucket ∈ `{RI1..RI5, "all"}`. For each
  metric, four columns: `<metric>_mean`, `<metric>_stdev`, `<metric>_min`,
  `<metric>_max`. Robustness columns are populated only on the
  `bucket="all"` row; a `failure_counts` column carries `reason=N`
  strings on the same row.

Also prints a human summary to stdout.

---

## 5. Algorithm deep dives

This is the crucial section for the fairness argument. For each algorithm,
we describe (a) what it sees, (b) how it plans, (c) how it executes, (d)
its failure modes, and (e) why the harness makes the comparison fair.

### 5.1 NNA-Dijkstra (hazard-blind greedy, fair replan)  — **implemented**

**What it sees at plan time:** the full base graph with edge attribute
`base_time` (unaffected by RI or hazards). It does **not** see the
`blocked_edges` set or the RI-adjusted `travel_time_map`.

**What it sees at execution time:** when stepping from `cursor` to `nxt`,
if `(cursor, nxt)` is in the scenario's `blocked_edges`, the policy is
allowed to inspect the passable subgraph for local repair. It cannot
preview blocked edges globally.

**Planning loop.** For each unvisited delivery `target`:

1. Call `nx.dijkstra_path(base_graph, current, target, weight="base_time")`.
2. Pick the delivery with the lowest `base_time` cost.
3. Traverse the returned path edge-by-edge.

**Fair-replan protocol (the key fairness fix).** When the next edge in the
current plan is blocked:

1. Check `nx.has_path(passable, cursor, best_target)`. If no, fail with
   `failure_reason = "trapped"`.
2. Otherwise call `nx.dijkstra_path(passable, cursor, best_target,
   weight="base_time")`. This is a local repair: the planner still does
   not see hazards; it only sees which edges are currently traversable.
3. Continue along the repaired path. Increment `replan_count`.

**Failure modes:** `trapped` (no passable path exists), `timeout`
(`step_idx > scenario.max_steps`). The impossible mode `invalid_action`
would indicate a bug — the policy should never propose an edge that isn't
in the base graph.

**Why this is fair vs the DQN.** Without replan, the DQN (action-masked)
gets to avoid blocks structurally while the NNA gets penalized every time
its hazard-blind plan hits a block. With replan, both policies have the
same "cannot traverse blocked edges" property; they differ only in which
*next node* they prefer — DQN via learned Q-values, NNA-Dijkstra via
shortest-`base_time` greedy.

**Diagnostic signal.** `replan_count` tells you how often the NNA's
hazard-blind plan had to be repaired. A high mean `replan_count` at high
RI indicates the NNA is making plans that conflict with the actual block
realization — which is exactly what we'd expect from a hazard-blind
algorithm and what the DQN should outperform.

### 5.2 NNA-A\* (algorithm_id: `NNA-AStar`)  — **implemented**

Identical to NNA-Dijkstra except `nx.astar_path` replaces `nx.dijkstra_path`
for the initial plan. Same fair-replan contract — replan on blocked edges
still uses Dijkstra internally (the helper in `runners/base.py` hardcodes
Dijkstra for local repair, which is fine for our purposes).

The heuristic is a flat-earth Euclidean-minutes estimate tuned for La
Trinidad's latitude (~16.4°N). It uses `x`/`y` node attributes (OSM
longitude/latitude) and divides straight-line distance by 30 km/h to
produce a lower-bound time estimate — keeping the heuristic admissible
for the `base_time` weight.

**algorithm_id note:** persisted as `NNA-AStar` (not `NNA-A*`) because
asterisks are invalid in Windows filenames, and the id is used as the
routes-file basename. Docs and presentations can still call it "NNA-A*".

Implementation: `runners/nna_astar.py`.

### 5.2b NNA-Dijkstra-Blind / NNA-AStar-Blind (hazard-blind, **no replan**)  — **implemented**

Same hazard-blind planning as §5.1 and §5.2 (shortest `base_time` path on
the full base graph, using Dijkstra or A* respectively). **The only
difference: no fair-replan loop.** The policy commits to its planned path
and the episode fails with `failure_reason = "blocked"` on the first
blocked edge it attempts to cross.

**What it sees at plan time:** the full base graph, hazard-blind. Same as
the replan-capable variants.

**What it sees at execution time:** nothing about blockages. Per-edge
traversal proceeds along the plan until either (a) the plan is finished
and every delivery is visited (success) or (b) the next edge is in
`scenario.blocked_set()` (immediate failure — no local repair, no
"walk around the block" search).

**Failure modes:** `blocked` (the signature failure — plan crossed a
blocked edge), `timeout` (shouldn't happen on the smoke cohort since the
plan is shortest-`base_time`; included for completeness), `no_route` (no
base-graph path from current to any unvisited delivery — pathological).
`replan_count` is always 0.

**Why this is in the harness.** Purpose is to establish a worst-case
classical baseline ("fast but blind" — knows nothing about blockages at
any stage) for the thesis comparison:

1. **Blind** — this variant. No block awareness. Worst case.
2. **Replan** — §5.1 / §5.2. Plan blind, repair on block. Fairness-safe
   competitor to the DQN.
3. **HA oracle** — §5.3. Plan with full foresight. Upper bound.

The DQN sits between Replan and HA. Without the Blind variant, it was hard
to answer *"how much does the replan loop help the classical baselines?"*
— now the gap between Blind and Replan quantifies exactly that.

**Identical stats between NNA-Dijkstra-Blind and NNA-AStar-Blind are
expected.** A* with an admissible heuristic on `base_time` finds the same
shortest paths as Dijkstra, and since neither does any local repair,
their realized routes are identical on every scenario. The two algorithms
stay distinct in the registry so the planning-cost contrast (A* heuristic
vs. Dijkstra sweep) can be measured via the `runtime` metric.

Implementation: `runners/nna_blind.py`, `runners/nna_astar_blind.py`,
shared execution in `runners/base.py::run_nna_blind`.

### 5.3 NNA-Dijkstra-HA (hazard-aware oracle)  — **implemented**

**What it sees at plan time:** the **activated** graph — the passable
subgraph with `travel_time` set to the RI-adjusted, hazard-weighted
`travel_time` from `scenario.travel_time_map`. This is the oracle:
it knows which edges are blocked AND the exact travel time for every
unblocked edge, including the speed multiplier and λ_hazard.

**Planning loop.** For each unvisited delivery, Dijkstra on the activated
graph with `weight="travel_time"`. Pick the one with lowest cost. Traverse
the planned path.

**Why no replan.** The oracle plans on the exact graph it will execute on.
Blocked edges were never admitted to the planner in the first place.

**Failure mode:** only `trapped` (when the activated graph disconnects
current from target — which, at Stage 1 feasibility filtering, shouldn't
happen because Stage 1 already checks this).

**Fairness framing.** NNA-HA is explicitly positioned as the **upper
bound**, not as a "competitor" to the DQN. Per manuscript §3.5.2, it
assumes perfect hazard foresight — an unrealistic condition in real-world
deployment. The fair head-to-head is DQN vs NNA-Dijkstra / NNA-A\*. NNA-HA
tells the reader "how close does the DQN get to an oracle?".

### 5.3b NNA-Dijkstra-HA-Blind (hazard-aware weighting, block-blind, no replan)  — **implemented**

**Despite the HA suffix, this is NOT an oracle.** The policy plans on
the **full** base graph (including blocked edges) with the hazard-aware
``travel_time`` weight (λ drag from manuscript §B), and fails with
``failure_reason = "blocked"`` the first time the planned next edge is
in ``scenario.blocked_set()``. The "HA" tag refers only to the weighting
function — not to block foresight. Contrast with §5.3 (true oracle).

**What it sees at plan time:** the full base graph — same edge set as
the hazard-blind Blind variants (§5.2b), but with ``travel_time``
attributes populated from ``scenario.travel_time_map`` (hazard-drag
+ RI speed reduction applied). The planner is block-blind; it sees all
edges including blocked ones, and the λ drag makes high-hazard edges
look expensive but never infeasible.

**What it sees at execution time:** nothing about blockages. Exactly
like the Blind variants — proceeds along the planned path until either
(a) every delivery is visited (success), or (b) the next edge is in
``scenario.blocked_set()`` (immediate failure, ``"blocked"``).

**Planning loop.** For each unvisited delivery, Dijkstra on
``view.hazard_aware_full_graph`` with ``weight="travel_time"``. Pick the
lowest-cost delivery. Traverse the planned path until success or block.

**Failure modes:** ``blocked`` (signature failure), ``timeout`` (should
not happen on the smoke cohort since the plan is shortest-``travel_time``;
included for completeness), ``no_route`` (no path in the full graph —
pathological). ``replan_count`` is always 0.

**Why this is in the harness.** Fills the "hazard-aware weights +
block-blind" cell of the 2×2 capability matrix left empty by the prior
tiering. Together with §5.2b (Blind), §5.1 (Replan), and §5.3 (HA):

| Runner | Hazard-aware weights? | Block-aware? |
|---|---|---|
| ``NNA-Dijkstra-Blind`` | No (`base_time`) | No — fail on block |
| ``NNA-Dijkstra`` | No (`base_time`) | Yes — reactive replan |
| **``NNA-Dijkstra-HA-Blind`` (new)** | **Yes (`travel_time`)** | **No — fail on block** |
| ``NNA-Dijkstra-HA`` | Yes (`travel_time`) | Yes — plans on passable subgraph |

The gap ``Blind → HA-Blind`` isolates *what hazard-aware weighting alone
buys you* when block foresight is absent. The gap ``HA-Blind → HA``
isolates *what block foresight buys you on top of hazard-aware weights*.
Together they decompose the single ``Blind → HA`` jump into
interpretable halves — and the ordering is no longer a 1D ladder, it's
a 2D capability matrix (see §7).

**Mid-RI dominance hypothesis (falsifiable).** HA-Blind should strictly
dominate Blind on success rate at **mid-RI (RI2–RI4)**, where blocks
are numerous enough to matter but the passable SCC is still rich.
Hazard-aware weights push the planner off high-λ edges, and high-λ
edges (high hazard scores) are *also* the ones most likely to be
blocked under the thresholded block rule, so HA-Blind incidentally
avoids many blocks without knowing they exist.

At the tails the gap narrows:

- **RI1**: near-zero blocks → both variants near 100% success;
  HA-Blind may pay a small ``travel_time`` premium for unproductive
  detours.
- **RI5**: most edges blocked → the subgraph collapses to a small
  SCC; scenarios reduce to short trips with few block opportunities,
  so Blind and HA-Blind converge.

On the `la_trinidad_mini` cohort this hypothesis holds:
`NNA-Dijkstra-Blind` vs `NNA-Dijkstra-HA-Blind` success rate is
100/100 at RI1, **25/40 at RI2** (+15pp), **85/90 at RI3** (+5pp),
**95/100 at RI4** (+5pp), 100/100 at RI5 — overall 81% → 86%. See
`docs/blind_nna_and_new_metrics_guide.md` for the full results
transcript.

Uniform dominance across RI would signal either a bug or a
misunderstanding of the block-rule/weighting alignment — the block
rule is discrete (``H_f ≥ θ_f(RI)``) while λ drag is continuous
(``1 + α_f·H_f + α_l·H_l``), so HA-Blind's incidental block-avoidance
works only where high-λ and high-block-probability overlap.

Implementation: `runners/nna_ha_blind.py`, shared execution in
`runners/base.py::run_nna_blind` (reused — the new ``plan_graph``
parameter lets the runner swap in ``view.hazard_aware_full_graph``
instead of ``view.base_graph``).

### 5.4 DQN (algorithm_ids: `DQN@balanced_HF`, `DQN@fast_HF`, `DQN@safe_HF`)  — **implemented**

Three runners, one per reward-profile variant. Each runner is a thin
wrapper (`runners/dqn.py`) around the `HazardRoutingEnv` and `DQN`
classes in the **vendored RL backend** at
`src/evaluation/rl_backend/rl_routing_wCUDA_wCheckP.py` (re-exported
via `runners/_rl_backend.py`). The backend ships inside the project —
no external `sys.path` splicing; install torch once with
`uv sync --extra dqn`.

**Per-RI checkpoint dispatch.** Each profile has 5 RI-specialist
checkpoints (RI1..RI5), all fine-tuned from the same profile
pretrain. At inference, each runner loads checkpoints lazily
(first scenario at a given RI triggers the load) and selects by
`scenario.rain_level`. So `DQN@balanced_HF` is a single algorithm_id
in the report but internally routes each scenario through the
RI-matched specialist.

**Graph / node-id translation.** The backend's env uses integer node
indices (`nx.convert_node_labels_to_integers`), while our harness
persists OSM string ids. The runner builds the `str → int` map at
first-run time by matching `(x, y)` positions between the env's graph
(loaded from `ckpt["base_graph_node_link"]`) and our cohort's graph.

**num_deliveries caveat.** The bundled checkpoints were trained with
`num_deliveries=2`; our canonical cohort uses 5. The DQN's MLP
weights are architecturally count-invariant (pooled unvisited is a
masked mean), so the checkpoint loads cleanly into a runner with
`num_delivery_slots=5`. The remaining gap is a semantic OOD on the
pooled-embedding distribution; see the smoke-test failure-counts
(`timeout=N`) for an empirical measure of that cost. If the gap
becomes load-bearing, retrain at `num_deliveries=5`.

**Fairness framing.** DQN and NNA-Dijkstra / NNA-AStar now have
identical structural block-avoidance: DQN via action-masking,
baselines via fair replan. The residual difference — choice among
passable neighbors — is the learned-vs-classical-greedy comparison
the thesis is making.

### 5.5 DQN — conceptual description (as seen in training)

**What it sees at plan time:** nothing global. It operates step by step.

**What it sees at each step:** a state vector plus a binary action mask.
State includes (from `../RL_framework/rl_routing_wCUDA_wCheckP_latest.py:474–599`):

- One-hot rain-level indicator (`rain_dim = 5`).
- Current node position features.
- Unvisited delivery set (encoded as a masked embedding).
- Local neighborhood features for each candidate next node (up to
  `max_neighbor_slots = 4` from the `_det` config).

Crucially, state includes each neighbor's `flood_hazard` and
`landslide_hazard` scores plus its `blocked` status (the action mask).

**Action space:** discrete, one slot per neighbor (up to
`max_neighbor_slots`). Masked actions have Q-value pinned to `-1e9`, so
`argmax` never picks them.

**Execution loop (per scenario):**

1. Env is reset with `start_node`, `delivery_nodes`, and pre-activated
   graph (blocked edges fixed at episode start, unchanged throughout the
   episode).
2. At each step, compute state → get Q-values → mask blocked actions →
   pick argmax (since eval `epsilon = 0.0`).
3. Transition: travel_time accumulates from the scenario's
   `travel_time_map`; reward is tracked for training but ignored at
   eval.
4. Episode ends when all deliveries visited (success) or step count
   exceeds `max_steps` (timeout) or every neighbor is blocked (trapped).

**Failure modes:** `trapped`, `timeout`. By construction never
`invalid_action`.

**Why this is fair.** By pre-baking `blocked_edges` into the scenario, the
DQN and the NNA-Dijkstra both see the same world. The DQN's action mask
gives it structural block avoidance; the NNA-Dijkstra's fair replan gives
it the same structural capability. The residual difference — what do they
choose when multiple passable neighbors exist? — is the honest policy
comparison.

**Information gradient (who knows what, when):**

```
                       |  Sees base graph  | Sees blocked edges       | Sees activated travel_time |
NNA-*-Blind            |  ✓                |  ✗ (never — fails)       |  ✗                         |
NNA-Dijkstra           |  ✓                |  ✗ (until traverse)      |  ✗                         |
NNA-A*                 |  ✓                |  ✗ (until traverse)      |  ✗                         |
NNA-Dijkstra-HA-Blind  |  ✓                |  ✗ (never — fails)       |  ✓ at plan (full graph)    |
DQN                    |  ✓ (via state)    |  local only (mask)       |  ✗                         |
NNA-Dijkstra-HA        |  ✓                |  ✓ (entire set)          |  ✓ (entire map, passable)  |
```

Note that this is no longer a strict 1D ladder. `HA-Blind` knows the
hazard-weighted travel times (same as the oracle) but is block-blind
(like the plain Blind variants); meanwhile `NNA-Dijkstra` (replan)
knows nothing about hazards but can react to blocks. Neither strictly
dominates the other — their relative performance at a given RI depends
on which capability matters more. The story the thesis tells is now a
2D decomposition (see §7): hazard-awareness × block-foresight.

---

## 6. Evaluation formulas — full derivation

### 6.1 Variable glossary

Every symbol used in this section, in one place.

| Symbol | Meaning |
|---|---|
| `e = (u, v)` | Directed edge in the road graph. |
| `L_e` | Physical length of edge `e` in meters. Pulled from OSM `length` attr. |
| `v_e` | Baseline speed on edge `e` in km/h. Constant 30 km/h for this study (§3.1.1). |
| `base_time_e` | `L_e / (v_e * 1000 / 60)` — baseline travel time in minutes under dry conditions. Pre-computed in `prepare_data.py`. |
| `H_f,e` | Flood hazard score of edge `e`, in [0, 1]. Canonical-3: 0.0 none, 0.2 low, 0.6 moderate, 1.0 high. |
| `H_l,e` | Landslide hazard score of edge `e`, in [0, 1]. Same mapping. |
| `RI` | Rainfall intensity level, integer 1–5. Sampled per episode in training; fixed per scenario in the harness. |
| `θ_f(RI)` | Flood-blocking threshold at RI. An edge with `H_f,e ≥ θ_f(RI)` is eligible for blocking. From Table 4 / `_det` configs: RI1 → 1.1 (never), RI2 → 1.0, RI3 → 0.6, RI4 → 0.6, RI5 → 0.6. |
| `θ_l(RI)` | Landslide-blocking threshold at RI. RI1–3 → 1.1 (never), RI4 → 1.0, RI5 → 0.6. |
| `μ(RI)` | Speed multiplier at RI. RI1 → 0.94, RI2 → 0.90, RI3 → 0.85, RI4 → 0.40, RI5 → 0.20. Multiplies the baseline speed `v_e`. Lower = slower. |
| `α_f` | **Flood travel-time drag weight.** From manuscript §B, empirically calibrated. 0.5 in the `_det` configs (`hazard.flood_time_weight`). Used ONLY in the travel-time formula. |
| `α_l` | **Landslide travel-time drag weight.** 0.5 (`hazard.landslide_time_weight`). |
| `w_f` | **Flood reward/exposure weight.** From manuscript §D (reward penalty) and §6.4.3 (safety metric). 0.6 per `_det` config's `reward.w_flood` and `metrics/hazard_exposure.py`. **Never appears in the travel-time formula — that's what α is for.** |
| `w_l` | **Landslide reward/exposure weight.** 0.4. |
| `λ_hazard(e)` | Hazard-induced travel-time multiplier for edge `e`: `1 + α_f · H_f,e + α_l · H_l,e`. Safe edge → λ = 1; worst-case (H_f = H_l = 1) with α_f = α_l = 0.5 → λ = 2. |
| `T_e(RI)` | Effective travel time of edge `e` at rainfall RI, in minutes. |
| `R` | A route — an ordered sequence of traversed edges. |
| `L(R)` | Total length of route `R` in meters (`Σ_{e ∈ R} L_e`). |
| `T(R, RI)` | Total travel time of route `R` at RI (`Σ_{e ∈ R} T_e(RI)`). |
| `Hazard(R)` | Hazard exposure of route `R` (defined below). |

### 6.2 Edge blocking rule (deterministic_v3)

An edge `e` is blocked in a given RI scenario iff:

```
   H_f,e ≥ θ_f(RI)   OR   H_l,e ≥ θ_l(RI)
```

Implementation: `scenario_generator.py::compute_blocked_edges`. The set is
computed once per RI and persisted in the scenario record, so every
algorithm's run sees the identical blocked set.

### 6.3 Effective travel time (for non-blocked edges)

Per manuscript §3.1.3 B, the effective travel time of a non-blocked edge
under rainfall RI is:

```
   T_e(RI)  =  base_time_e                        baseline minutes
              ÷ μ(RI)                             speed reduction
              × (1 + α_f · H_f,e + α_l · H_l,e)   hazard drag (= λ_hazard)
```

Two factors multiply together:

- **Speed reduction `1/μ(RI)`** captures the whole-system slowdown from
  rainfall. At RI1 the driver is 6% slower (`1/0.94 ≈ 1.064`). At RI5 the
  driver is 5× slower (`1/0.20 = 5`). This is the dominant effect at high
  RI.
- **Hazard drag `λ_hazard`** captures per-edge extra slowness for
  traversing hazardous zones (reduced visibility, cautious driving,
  partial water, debris, etc.). With `α_f = α_l = 0.5` (the current
  `_det` calibration), a worst-case edge (H_f = H_l = 1) has
  `λ = 1 + 0.5 + 0.5 = 2`.

**α vs w — don't conflate them.** The manuscript uses two distinct
symbol sets:

- **α_f, α_l (travel-time drag weights)** appear only here, in
  `λ_hazard`. Sourced from `hazard.flood_time_weight` /
  `hazard.landslide_time_weight` in the `_det` config. Empirically
  calibrated to 0.5/0.5.
- **w_f, w_l (reward / hazard-exposure weights)** appear in the training
  reward penalty (§D) and in the hazard-exposure safety metric
  (§6.4.3). 0.6/0.4 per `_det` config `reward.w_flood` /
  `reward.w_landslide`. **They do NOT appear in the travel-time
  formula.**

An earlier version of `scenario_generator.py` used `w_f, w_l` in the
λ formula (the "_det reward defaults" in the hardcoded constants were
the old 0.6/0.4). That was a code-level bug that silently produced
travel times diverging from what the DQN saw during training. Fixed;
the generator now reads α from the config, falling back to the 0.5/0.5
module defaults.

`T_e(RI)` is computed once per (edge, RI) in `compute_travel_time_map` and
persisted in `scenario.travel_time_map`. Policies read it at
traversal time; they never compute it themselves. This prevents
"algorithm A and B disagree on what an edge actually costs" bugs.

### 6.4 Metrics — definitions and rationale

All metrics are implemented as plugins in `metrics/` — each file exports
`compute(scenario, route) -> float`. The evaluator aggregates by
`(algorithm_id, RI_key)` and `"all"`. `NaN` values are excluded from the
aggregate (used to mark "metric not applicable on failed episodes").

#### 6.4.1 Success rate (`metrics/success.py`)

```
   success(scenario, route) = 1  if route.success else 0
```

Aggregator: mean over scenarios. This is the primary mission-critical
metric (§3.6.1 A). 100% success with slow times is preferable to 95%
success with fast times — delivery-failure cost dwarfs efficiency loss in
disaster contexts.

Because the cohort is feasibility-filtered at Stage 1, the denominator is
the cohort size. An algorithm's success rate therefore directly measures
*policy failures* (trapped, timeout), not *scenario infeasibility*.

#### 6.4.2 Average travel time (`metrics/travel_time.py`)

```
   travel_time(scenario, route) = Σ_{e ∈ route.per_edge} e.travel_time   if success
                                = NaN                                     if failed
```

NaN for failed routes means the aggregator only averages over successful
ones (matches §3.6.1 B: "calculated only for successfully completed
episodes").

This is **secondary** to success rate. Readers must be reminded that a
10% slower travel time with higher success rate is operationally better
than the reverse.

**Caveat on success-conditional averaging.** If Algorithm A succeeds on
95% of scenarios but Algorithm B on only 80%, B's time average is taken
over a potentially easier sub-cohort (it failed on the hard ones). This
can make the weaker algorithm *appear* faster. The evaluator reports
per-RI means to help spot this; consumers should also look at Pareto
plots (planned for `visualization.py`) rather than relying on the single
"avg travel time" number.

#### 6.4.3 Hazard exposure (`metrics/hazard_exposure.py`) — the key safety metric

Manuscript §3.6.1 C defines:

```
   Hazard(R) = Σ_{e ∈ R} w_e · L_e
   where  w_e = w_f · H_f,e + w_l · H_l,e
```

Expanded:

```
   Hazard(R) = Σ_{e ∈ R} (w_f · H_f,e + w_l · H_l,e) · L_e
             = w_f · Σ_{e ∈ R} H_f,e · L_e   +  w_l · Σ_{e ∈ R} H_l,e · L_e
             = w_f · (flood-exposure length) + w_l · (landslide-exposure length)
```

Units: `score × meters`. With `w_f = 0.6`, `w_l = 0.4`, a 100 m edge of
pure flood-high hazard (H_f = 1, H_l = 0) contributes `0.6 × 100 = 60`.
A 100 m fully safe edge contributes `0`. A 100 m worst-case edge
(H_f = H_l = 1) contributes `0.6 × 100 + 0.4 × 100 = 100`.

**Why length-weighted.** Without `L_e`, a short high-hazard shortcut
looks identical to a long high-hazard detour — but the latter keeps the
vehicle in the danger zone for more road. Length weighting respects the
manuscript §3.6.1 C stated rationale: "cumulative hazard-weighted
distance."

**Why w_f = 0.6, w_l = 0.4.** These are the `_det` config reward
weights. Keeping the metric weights aligned with the training reward
weights means the evaluator rewards the DQN on exactly the same axis it
was trained to optimize. Without this alignment, we'd risk measuring the
DQN on something it never saw during training.

**Why unweighted `Σ (H_f + H_l)` in the old `monte_carlo.py` is wrong.**
It lacks `L_e` (can't distinguish short from long exposure), doesn't
split by flood/landslide, and doesn't match the training reward. The new
metric fixes all three.

**NaN on failure** — like travel time, hazard exposure is only meaningful
for completed routes.

#### 6.4.4 Robustness (`metrics/robustness.py`) — **implemented**

Manuscript §3.6.1 E:

```
   Robustness(metric) = 1  -  σ(metric across RI) / μ(metric across RI)
```

A robustness score near 1.0 means the metric is consistent across RI₁ –
RI₅. A low score means the policy degrades sharply at high RI. Curriculum
learning is supposed to improve this; this metric validates that claim.

**Implementation.** Unlike the other metrics, robustness is a *second-
order* statistic — it consumes per-RI means, not per-route values.
`metrics/robustness.py` exports `compute_robustness_ratio(ri_means)`; the
evaluator calls it once per `(algorithm, metric)` pair after the main
aggregation loop and stores the result in
`report["algorithms"][algo_id]["robustness"][metric_name]`.

Because of that shape, robustness is NOT in `metrics.REGISTRY` (which is
the per-route dispatch table). It's populated by the
`_attach_robustness` hook in `evaluator.py` after `_safe_stats` has run.

Guard: returns `None` if fewer than two RI means are available or if
`|μ| < 1e-9` (to avoid divide-by-zero). Null robustness values mean
"metric wasn't applicable on enough RIs to judge" rather than "policy
is not robust."

#### 6.4.5 Hazard score — raw unweighted exposure (`metrics/hazard_score.py`)

```
   hazard_score(R) = Σ_{e ∈ R} (H_f,e + H_l,e) · L_e   if success
                   = NaN                                if failed
```

Companion to `hazard_exposure` (§6.4.3) that **drops the reward weights
w_f, w_l**. Interpretation: "how many meters × score units of hazard did
the policy traverse, treating flood and landslide equally?"

The two metrics answer complementary questions:

- `hazard_exposure` — "policy-weighted safety cost" (aligned with the
  DQN's training reward penalty; use this when comparing policies on the
  axis they were trained on).
- `hazard_score` — "raw physical exposure" (independent of reward
  calibration; use this when debating "what if we retrained with
  different weights?" or for sensitivity analysis).

For a 100 m worst-case edge (H_f = H_l = 1), `hazard_exposure` contributes
`(0.6 + 0.4) · 100 = 100` whereas `hazard_score` contributes
`(1 + 1) · 100 = 200`. So absolute magnitudes differ even though the
per-algorithm *ordering* is usually the same.

#### 6.4.6 Steps — edge-traversal count (`metrics/steps.py`)

```
   steps(R) = len(R.per_edge)   if success
            = NaN                if failed
```

Counts every edge walked, including those added by NNA replans. Smaller
≠ better (the hazard-blind NNA often finds the shortest path in step-
count because it ignores safety detours); use in conjunction with
`distance` and `travel_time` to characterize path shape.

#### 6.4.7 Distance — meters walked (`metrics/distance.py`)

```
   distance(R) = Σ_{e ∈ R} e.length_m   if success
               = NaN                     if failed
```

Physical distance traversed — reflects what the policy *actually* walked,
not what it planned. For the replan-capable NNAs this can exceed the
shortest-path distance by the detours taken during replan.

#### 6.4.8 Runtime — wall-clock ms per episode (`metrics/runtime.py`)

```
   runtime(R) = R.wall_time_ms   (always defined, success or failure)
```

Thin re-export of `Route.wall_time_ms` as a registered metric so it
participates in the standard aggregation + CSV pipelines. Unlike
`travel_time` / `hazard_exposure` / `steps` / `distance`, this is
meaningful even on failed episodes — e.g., "how long did the blind NNA
spend planning + walking before hitting a block?" — which is why it has
no NaN-on-failure guard.

Computational cost per policy is dominated by graph algorithms (Dijkstra /
A*) for NNAs and forward passes through a small MLP for DQN. NNA
runtimes live in the 1–10 ms range per scenario; DQN is closer to 20–80
ms per scenario (model load dominates on the first scenario per RI).

#### 6.4.9 Replan count (currently reported as diagnostic, not a metric)

Not a "metric" in the thesis sense, but the evaluator surfaces
`mean replan_count` per algorithm. High numbers at high RI mean the
NNA's hazard-blind plan is repeatedly in conflict with actual blocks — a
direct measure of how much work the fair-replan protocol is doing on the
NNA's behalf. If this number is close to zero, the fairness fix is
minimally affecting outcomes; if it's large, the fix is load-bearing.

`replan_count` is always 0 for blind NNAs, DQN, and NNA-HA by
construction — those policies don't use the replan mechanism.

---

## 7. Fairness justification, point by point

Going back to the original fairness concerns from the session:

| Concern | Mitigation in this harness |
|---|---|
| Hazard-blind NNAs fail hard on blocked edges while DQN masks them out. | NNAs now replan from current node on block encounter (`runners/base.py::run_nna_with_fair_replan`). Structural capability is equalized. |
| Retry-sampling in old MC drops the hardest episodes only for the baselines. | Retries moved to Stage 1 (scenario generation). `scenarios.jsonl` contains only feasible scenarios; every algorithm sees the identical set. |
| NNA-HA oracle planning over the activated graph gave it perfect realization foresight. | Kept as the "upper bound" per manuscript §3.5.2 — labeled explicitly as oracle, not a peer competitor. Fair head-to-head is DQN vs hazard-blind NNAs. |
| Success-conditional averaging of travel time biases toward whichever algorithm fails on harder episodes. | Per-RI breakdown reported. Remediation path: Pareto plots in `visualization.py` (deferred). |
| Probabilistic activation in old code vs. deterministic in the manuscript. | Harness uses `activation_mode = "deterministic_v3"` by default. Probabilistic is retained as legacy mode only. |
| Hazard-exposure formula in old code didn't match §3.6.1 C. | Reimplemented correctly in `metrics/hazard_exposure.py`. Hand-calc regression test planned. |
| Different RNG paths between DQN eval and baseline MC. | Single `master_seed` in the cohort; scenarios are persisted; no stochasticity at policy-run time (DQN eval `epsilon=0`). |

Residual asymmetries we deliberately accept:

- **NNA-HA sees the full blocked set at plan time.** This is the point of
  an oracle; it is clearly framed as an upper bound, not a competitor.
- **DQN sees rain level explicitly in its state.** NNA-Dijkstra and
  NNA-A\* do not (they are hazard-blind by definition). This is a
  capability difference the thesis is specifically trying to measure.
- **DQN's local neighborhood embedding exposes per-neighbor hazard
  scores.** Same story — the thesis is about whether a policy can learn
  to use hazard scores locally. NNA-Dijkstra being blind to them is the
  baseline being compared against, not a bug.

The harness eliminates unintentional asymmetries. The remaining ones are
the intentional experimental variables.

### 7.1 The 2D capability matrix (post-HA-Blind)

Prior to §5.3b, the classical NNA runners formed a 1D ladder: Blind <
Replan < HA oracle. Adding `NNA-Dijkstra-HA-Blind` turns that ladder
into a 2D matrix along two orthogonal capabilities:

|                        | **Block-blind** (fail on block) | **Block-aware** (replan or exclude) |
|---|---|---|
| **Hazard-blind** (`base_time` weights) | `NNA-Dijkstra-Blind` | `NNA-Dijkstra` (fair replan) |
| **Hazard-aware** (`travel_time` weights) | `NNA-Dijkstra-HA-Blind` | `NNA-Dijkstra-HA` (oracle) |

Reading the matrix gives two direct ablation claims:

- **Value of hazard-aware weighting** (in isolation from block
  foresight) = `HA-Blind` − `Blind`. Isolates the effect of the λ-drag
  reward shaping on planner behavior when the policy still commits to
  its plan and fails on blocks.
- **Value of block foresight** (on top of hazard-aware weights) =
  `HA` − `HA-Blind`. Isolates the effect of pre-excluding blocked
  edges from the planning substrate.

These two gaps sum to the original `Blind → HA` jump but are
individually interpretable. The matrix also shows why `NNA-Dijkstra`
(fair replan) and `NNA-Dijkstra-HA-Blind` cannot be ranked globally:
they trade block foresight for hazard-aware weighting, and which one
wins depends on the RI regime and graph topology.

The DQN's position in the matrix is **hazard-aware (via state
features) × partial block-foresight (via the action mask at each
step)**. It sits between Replan and HA on the block-foresight axis
(per-step local awareness rather than scenario-wide pre-exclusion),
and is hazard-aware in its policy (learned from state features plus
training reward). Its fair comparators are `NNA-Dijkstra` and
`NNA-Dijkstra-HA-Blind` — both have one of the two capabilities the
DQN combines.

---

## 8. Sample end-to-end run (traceable output — legacy, pre-α-fix)

The following was produced by the initial smoke test on 2026-04-18 on the
`selected_subgraph_n200` graph, **before** the `α_f / α_l` travel-time
weight fix (see §6.3). The blocked-edge counts and SCC sizes are
α-independent (blocking is a threshold test on raw hazard scores), so
those numbers are still exact. The per-edge `travel_time` and aggregate
`travel_time`/`hazard_exposure` means are off by ~6% from post-fix
runs — read the structural ratios, not the absolute magnitudes. The
most recent six-method smoke-test numbers live in
[`experimental_setup_blueprint_e2e.md`](./experimental_setup_blueprint_e2e.md) §5.

### 8.1 Stage 1 run

```
$ python -m src.evaluation.scenario_generator \
    --graph data/staged_subgraphs/selected_subgraph_n200.graphml \
    --graph-id la_trinidad_subgraph_n200 \
    --config src/evaluation/configs/hazard_training_final/balanced_HF/stage_200_balanced_HF_RI3_det.json \
    --cohort-id la_trinidad_mini \
    --num-scenarios 100 \
    --num-deliveries 5 \
    --master-seed 42 \
    --max-steps 220

17:08:24  INFO     Loading graph: data/staged_subgraphs/selected_subgraph_n200.graphml
17:08:24  INFO       nodes=200 edges=417
17:08:24  INFO       RI1: blocked 0/417 edges (0.0%);  largest SCC = 200/200 nodes
17:08:24  INFO       RI2: blocked 18/417 edges (4.3%); largest SCC = 189/200 nodes
17:08:24  INFO       RI3: blocked 54/417 edges (12.9%);largest SCC = 96/200 nodes
17:08:24  INFO       RI4: blocked 159/417 edges (38.1%);largest SCC = 68/200 nodes
17:08:24  INFO       RI5: blocked 277/417 edges (66.4%);largest SCC = 10/200 nodes
17:08:25  INFO       wrote 100 scenarios to ...\cohorts\la_trinidad_mini\scenarios.jsonl
17:08:25  INFO     Cohort generation took 0.3s
```

Expected numbers:

- Edges should roughly double after undirected → directed symmetrization
  (209 in the file → 417 in the loaded DiGraph).
- Blocked count should increase monotonically with RI.
- SCC size should decrease monotonically with RI. RI5 ≈ 10 nodes on this
  subgraph — the cohort-composition constraint discussed in the
  handoff.

### 8.2 `cohort.json` (abbreviated)

```json
{
  "cohort_id": "la_trinidad_mini",
  "generated_at": "2026-04-18T17:08:25+08:00",
  "master_seed": 42,
  "graph_id": "la_trinidad_subgraph_n200",
  "num_scenarios": 100,
  "sampling_policy": "stratified_by_RI",
  "ri_distribution": {"RI1": 20, "RI2": 20, "RI3": 20, "RI4": 20, "RI5": 20},
  "num_deliveries": 5,
  "activation_mode": "deterministic_v3",
  "feasibility_filtered": true,
  "scenarios_path": "scenarios.jsonl"
}
```

### 8.3 One scenario (abbreviated)

```json
{
  "scenario_id": "la_trinidad_mini_000000",
  "graph_id": "la_trinidad_subgraph_n200",
  "rain_level": 1,
  "activation_mode": "deterministic_v3",
  "activation_seed": 43,
  "start_node": "6059062479",
  "delivery_nodes": ["9109808232","3804492302","7958998574","3555331362","7958998581"],
  "blocked_edges": [],
  "travel_time_map": {"1118275113|3802985218": 1.0436, "1118275113|3804477878": 0.2134, ...},
  "max_steps": 220,
  "num_deliveries": 5,
  "metadata": {"master_seed": 42, "ri_key": "RI1", "sample_attempts_for_this_ri": 1}
}
```

### 8.4 Stage 2 run

```
$ python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra

17:08:37  INFO     Cohort: la_trinidad_mini  (100 scenarios)
17:08:37  INFO     Running NNA-Dijkstra (sha256:7e58c93e37baa189)
17:08:38  INFO       [NNA-Dijkstra] 100/100
17:08:38  INFO       NNA-Dijkstra: 100 routes in 0.69s
```

### 8.5 One route (abbreviated)

```json
{
  "scenario_id": "la_trinidad_mini_000000",
  "algorithm_id": "NNA-Dijkstra",
  "algorithm_config_hash": "sha256:7e58c93e37baa189",
  "visit_order": ["3555331362","7958998574","7958998581","3804492302","9109808232"],
  "per_edge": [
    {"u":"6059062479","v":"7673433614","step":0,"was_replan":false,
     "travel_time":0.1432,"hazard_flood":0.0,"hazard_landslide":1.0,"length_m":48.06},
    {"u":"7673433614","v":"7673433628","step":1,"was_replan":false,
     "travel_time":0.1407,"hazard_flood":0.0,"hazard_landslide":1.0,"length_m":47.22},
    ...
  ],
  "success": true,
  "failure_reason": null,
  "replan_count": 0,
  "wall_time_ms": 6.34,
  "policy_metadata": {"plan_weight":"base_time","variant":"dijkstra_hazard_blind_with_fair_replan"}
}
```

Note that the first two edges are `hazard_landslide = 1.0` — the policy is
hazard-blind, so it happily chose the shortest base-time path even
through landslide-rated road. This is expected NNA behavior. The DQN
should avoid these edges when they are traversable, and the hazard
exposure metric will reflect that.

### 8.6 Stage 3 run

```
$ python -m src.evaluation.evaluator --cohort-dir src/evaluation/cohorts/la_trinidad_mini

========================================================================
Cohort: la_trinidad_mini  (100 scenarios)
Graph:  la_trinidad_subgraph_n200
Mode:   deterministic_v3
========================================================================

  NNA-Dijkstra  (scored 100)
    success_rate     = 100.00%  (100 episodes)
    travel_time(min) = mean=19.7  std=9.5  over 100 successful
    hazard_exposure  = mean=1730.13  std=1385.94  over 100 successful
    replan_count     = mean=0.14  max=2
    by RI:   RI1=100.0%  RI2=100.0%  RI3=100.0%  RI4=100.0%  RI5=100.0%
```

### 8.7 Sanity-check expectations

From the smoke test, these patterns should hold on any reasonable cohort:

| Pattern | Rationale |
|---|---|
| Success = 100% at all RI for the feasibility-filtered NNA-Dijkstra | Stage 1 only admits feasible scenarios; fair replan handles any local block. |
| Hazard-exposure mean *decreases* at high RI | At RI5, so many edges are blocked that the only passable paths are the unblocked (therefore safer) ones — artifact of the SCC collapse. On the full graph this trend is subtler. |
| Travel time mean is *lower* at RI5 in the small cohort | Same reason — only short paths survive in the 10-node RI5 SCC. On the full graph, high-RI travel times rise due to speed_mult. |
| `replan_count` rises with RI | More blocked edges → more conflicts with hazard-blind plans. |

If any of those trends is inverted, it signals either a bug in the
harness or an unexpected graph property — investigate.

---

## 9. Adding new things

### 9.1 Add a new baseline algorithm

1. Create `runners/<name>.py` with a dataclass implementing the `Policy`
   protocol from `runners/base.py`. Typically you just wrap one of the
   shared execution helpers with a different `path_fn`:
   - `run_nna_with_fair_replan` — plan-blind, repair on block (see
     `runners/nna.py` and `runners/nna_astar.py` for examples).
   - `run_nna_blind` — plan once, fail on block (see `runners/nna_blind.py`
     and `runners/nna_astar_blind.py` for the hazard-blind flavor, and
     `runners/nna_ha_blind.py` for a hazard-aware-weight-but-block-blind
     flavor). For hazard-aware-weight variants, pass
     `plan_graph=view.hazard_aware_full_graph` and `plan_on="travel_time"`
     — the same helper handles the blocked-set check at traversal time.
   For a block-aware hazard-aware variant (true oracle), follow
   `runners/nna_ha.py` which operates directly on `view.activated_graph`
   without a helper.
2. Register in `POLICY_FACTORIES` in `run_policies.py`.
3. Invoke via `--algorithms <name>`.

### 9.2 Add a new DQN variant

`runners/dqn.py` already implements one `DQNRunner` per reward-profile
with lazy per-RI checkpoint dispatch. To add a fourth profile
(say, `aggressive_HF`):

1. Ensure a directory tree exists under
   `models/rl_checkpoints/aggressive_HF/stage_200_aggressive_HF_RI{1..5}_det/best_model.pt`.
2. Copy a representative `_det.json` config to
   `src/evaluation/configs/hazard_training_final/aggressive_HF/stage_200_aggressive_HF_RI3_det.json`
   (RI3 is conventional — architecture is RI-invariant per profile).
3. In `run_policies.py` extend `POLICY_FACTORIES`:

   ```python
   "DQN@aggressive_HF": lambda: _make_dqn_runner("aggressive_HF"),
   ```

   The existing `_make_dqn_runner(profile)` helper handles everything
   else (checkpoint root, config path resolution, lazy loading).

4. Invoke via `--algorithms DQN@aggressive_HF`. The runner picks
   `RI{scenario.rain_level}` automatically; no per-RI flag.

To change the *architecture* (e.g., larger hidden dims, different
`node_embedding_dim`), edit the representative config for that profile
and retrain the checkpoint set — then step 3 is unchanged because the
runner just swaps `state_dict`s into whatever architecture the config
specifies.

> **Standalone note:** the DQN runner depends only on the vendored RL
> backend at `src/evaluation/rl_backend/` and the bundled checkpoints
> at `models/rl_checkpoints/`. Both ship with this project — no
> external repo is needed. See §11 and the blueprint for the
> self-contained install flow.

### 9.3 Add a new metric

1. Create `metrics/<name>.py` with `def compute(scenario, route) -> float`.
2. Register it in `metrics/__init__.py::REGISTRY`.
3. Rerun `evaluator.py` — no need to regenerate scenarios or re-run
   policies. This is the whole point of the three-stage split.

### 9.4 Generate a new cohort (different graph, RI mix, or deliveries)

Just change the CLI arguments. The harness supports:

- `--graph` — any graphml with the expected edge attributes.
- `--ri-keys RI1 RI2 RI3` — restrict to a subset of RI levels.
- `--num-deliveries N` — any positive integer.
- `--activation-mode probabilistic_v1` — for legacy comparison.
- `--cohort-id <new_id>` — outputs to a separate directory.

The canonical thesis cohort (recommended in the plan) is
`la_trinidad_v1` on the full `la_trinidad_hazard_graph.graphml` with
num_scenarios=2500, num_deliveries=5, master_seed=42, deterministic_v3.

---

## 10. What's intentionally NOT in v1

**Implemented since the initial v1**: NNA-A\*, NNA-Dijkstra-HA, Robustness
metric (§5.2, §5.3, §6.4.4), and the DQN runner with per-RI dispatch
for the three profiles (§5.4). Still deferred:

| Deferred | Reason |
|---|---|
| Retraining DQN at `num_deliveries=5` | Current checkpoints were trained at 2; the runner loads them into 5-slot inference (count-invariant pool), producing a ~10–20% timeout tax. Retraining closes the OOD gap; out of scope for the runner session. |
| Folium HTML replays and aggregate PNG plots (`visualization.py`) | Nice-to-have for qualitative inspection; not on the critical path for producing numbers. |
| Probabilistic-mode cohort | Deterministic_v3 is the locked direction; probabilistic is legacy only. |
| pytest suite | The verification-plan section of the plan file lists six tests to implement. |

> **Running experiments today:** consult
> [`experimental_setup_blueprint_e2e.md`](./experimental_setup_blueprint_e2e.md)
> for the full suite of validated command recipes (smoke test, canonical
> thesis cohort, per-profile ablations, probabilistic legacy mode, α
> overrides), expected outputs, per-stage manual verification
> checklists, and a "Scaling & Code-Cleaning Notes" section that
> documents the current hardcoded cross-repo paths.

---

## 11. References

- [`experimental_setup_blueprint_e2e.md`](./experimental_setup_blueprint_e2e.md)
  — operational companion: commands, expected outputs, manual
  verification checklists, data-gathering workflow, scaling notes.
- `HANDOFF_2026-04-18.md` — session-level handoff at the parent repo root.
- `CLAUDE.md` — parent-repo orientation for future Claude Code sessions.
- Manuscript `thesis_manuscript.docx` (or `thesis_manuscript.md` after
  pandoc conversion) — §3.1.2, §3.1.3, §3.5, §3.6 are the methodology
  this harness implements. §B (α_f, α_l travel-time drag) and §D (w_f,
  w_l reward-penalty) are the weight schemas referenced in §6.1 / §6.3.
- `src/evaluation/rl_backend/rl_routing_wCUDA_wCheckP.py` — vendored
  training/inference module. The DQN runner imports `HazardRoutingEnv`,
  `DQN`, `select_action`, `load_config`, and `apply_runtime_config`
  from it via the re-export adapter `runners/_rl_backend.py`. No
  `sys.path` manipulation. Load-bearing only for DQN evaluation; NNA
  runners don't depend on this module or on torch.
- `models/rl_checkpoints/{profile}/stage_200_{profile}_RI{1..5}_det/best_model.pt`
  — 15 bundled checkpoints (3 profiles × 5 RIs) the DQNRunner
  dispatches to. ~7.5 MB total; committed directly.
- `_det` activation configs under
  `src/evaluation/configs/hazard_training_final/*/*_det.json` — source of
  truth for rain-level thresholds, speed multipliers, and (post-α-fix)
  `hazard.flood_time_weight` / `hazard.landslide_time_weight`. Copied
  verbatim from `RL_framework/configs/hazard_training_final/` so Benguet
  can regenerate cohorts standalone. Any `_det.json` can be passed to
  `scenario_generator.py --config`.
