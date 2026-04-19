# RL Backend (vendored)

Self-contained DQN training / inference module used by the Fair Evaluation
Harness. This directory is a **vendored snapshot** of the research code,
not a git submodule — intentional, because:

- We only need a stable copy at a known architecture. Submodule
  versioning is overkill for a two-file dependency.
- Shipping the module inside `src/evaluation/` makes the evaluation
  pipeline `pip install .`-able in one step.
- Downstream code imports through the `runners/_rl_backend.py` adapter,
  not from files in this directory directly, so downstream call sites
  don't have to mention the vendored filename.

## Files

| File | Purpose |
|---|---|
| `rl_routing_wCUDA_wCheckP.py` | Environment (`HazardRoutingEnv`), network (`DQN`), action selection, config loader, hazard activation. Both training and inference live in one module. |
| `utils/graph_utils.py` | Graph helpers used by the module above (`get_raw_osm_graph`, `to_training_graph`). |

## Why `rl_routing_wCUDA_wCheckP.py` isn't renamed

The filename is ugly, but renaming would force us to verify every
`torch.load` call path and every debug log line that mentions it. The
file is never displayed to users — code only references it through
`runners/_rl_backend.py`. Keep the name stable; that's the lowest-risk
option.

## Key invariants the harness depends on

- `DQN` constructor signature: `(state_dim, action_dim, num_nodes, num_delivery_slots, hidden_sizes, node_embedding_dim)`.
- `HazardRoutingEnv.state_dim` == 40 for the canonical config (7 + 4·7 + 5).
- `HazardRoutingEnv.action_dim` == 4 (`max_neighbor_slots`).
- Checkpoints serialize plain Python types + tensor state only (no
  pickled class instances). Safe to relocate the module without
  breaking `torch.load`.

## Upstream-merge policy

If the source training module evolves, re-copy the two files (and
update this README with the new snapshot date). Do **not** edit the
vendored files in place — any fix we want on our side should flow
through the re-export adapter at `runners/_rl_backend.py`.

## Dependency

Requires torch. Install with `uv sync --extra dqn` from the project
root — the `dqn` extra in `pyproject.toml` pulls `torch>=2.10.0`. NNA
runners don't need this; only the DQN runner imports torch.
