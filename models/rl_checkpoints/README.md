# Bundled DQN Checkpoints

15 trained DQN checkpoints that the Fair Evaluation Harness loads at
inference time. Organized by reward profile, one checkpoint per
rainfall intensity (RI).

## Layout

```
models/rl_checkpoints/
├── balanced_HF/
│   ├── stage_200_balanced_HF_RI1_det/best_model.pt
│   ├── stage_200_balanced_HF_RI2_det/best_model.pt
│   ├── stage_200_balanced_HF_RI3_det/best_model.pt
│   ├── stage_200_balanced_HF_RI4_det/best_model.pt
│   └── stage_200_balanced_HF_RI5_det/best_model.pt
├── fast_HF/   (5 checkpoints, same pattern)
└── safe_HF/   (5 checkpoints, same pattern)
```

## Provenance

| Field | Value |
|---|---|
| Training graph | La Trinidad 200-node subgraph (`data/staged_subgraphs/selected_subgraph_n200.graphml`) |
| Training activation mode | `deterministic_v3` (shifted thresholds, no per-edge probabilistic blocking) |
| Training `num_deliveries` | **2** (harness cohorts run at 5 — documented OOD; see `src/evaluation/README.md` §5) |
| Hazard time weights | α_f = α_l = 0.5 |
| Reward weights | w_f = 0.6, w_l = 0.4 (per profile; fast/safe override) |
| Node embedding dim | 16 |
| Hidden sizes | `[128, 64]` |
| State dim | 40 (7 target + 4·7 neighbor + 5 rain one-hot) |
| Action dim | 4 (`max_neighbor_slots`) |

## Loading path (runtime)

- `src/evaluation/run_policies.py::_DQN_CHECKPOINT_ROOT` points at this
  directory.
- `src/evaluation/runners/dqn.py::_checkpoint_path(ri)` resolves one
  file as `<checkpoint_root>/<profile>/stage_200_<profile>_RI<ri>_det/best_model.pt`.

## Per-RI dispatch

At evaluation, each DQN runner (`DQN@balanced_HF`, `DQN@fast_HF`,
`DQN@safe_HF`) owns all five RI checkpoints and picks the one matching
`scenario.rain_level` (1–5) at inference. This mirrors how the
checkpoints were trained — as parallel specialists, not a sequential
curriculum — and is what the reportable cross-method comparison
assumes.

## Architecture invariants (must not drift)

The harness constructs `DQN(state_dim, action_dim, num_nodes,
num_delivery_slots, hidden_sizes, node_embedding_dim)` using
architecture values from the matching config file at
`src/evaluation/configs/hazard_training_final/<profile>/stage_200_<profile>_RI3_det.json`.
If any of `hidden_sizes`, `node_embedding_dim`, or the state-dim
derivation changes, the corresponding checkpoint must be retrained and
this README updated.

## Checkpoint payload shape

Each `best_model.pt` is a dict with (at minimum) these keys:

- `model_state_dict` — DQN policy network weights (tensor state only).
- `target_state_dict` — target network weights.
- `optimizer_state_dict` — Adam state.
- `base_graph_node_link` — `nx.node_link_data(base_graph)` snapshot,
  used by the runner to derive the int-indexed graph the DQN was
  trained on.
- `graph_num_nodes`, `action_dim`, `num_deliveries`, `seed`,
  `model_variant` — metadata scalars.

No pickled class instances — all plain Python types + tensors — so
`torch.load(..., weights_only=False, map_location="cpu")` is safe
and portable across machines.

## Regeneration

If you retrain, run the training pipeline and drop the resulting
`best_model.pt` files into the matching subfolder here. No changes to
the evaluation code are needed as long as the architecture invariants
above are preserved.

## Size

- 15 files × ~514 KB = **~7.5 MB total**. Below any sane git-lfs
  cutoff; committed directly to the repo so the evaluation harness is
  `git clone`-runnable without external steps.
