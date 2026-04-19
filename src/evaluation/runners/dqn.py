"""DQN runner with per-RI checkpoint dispatch.

Three instances are registered in ``run_policies.POLICY_FACTORIES`` — one
per profile (``balanced_HF``, ``fast_HF``, ``safe_HF``). Each instance
owns five checkpoints (RI1..RI5) and selects the right one per scenario
by reading ``scenario.rain_level``. That matches how the bundled
checkpoints were trained — as parallel specialists, not a sequential
curriculum: each ``stage_200_{profile}_RI{N}_det/best_model.pt`` is
fine-tuned from the profile's warm pretrain for a specific rain level.

Architectural note on num_delivery_slots
----------------------------------------
The bundled checkpoints were trained with ``num_deliveries = 2``. Our
canonical harness cohort uses ``num_deliveries = 5``. The DQN's
``num_delivery_slots`` constructor argument controls the *input tensor
shape* for ``unvisited_idx`` / ``unvisited_mask``, but the MLP's
learned weights are count-invariant (pooled_unvisited is a masked
mean → shape ``(emb_dim,)`` regardless of slots). Loading a 2-trained
checkpoint into a model constructed with 5 slots is architecturally
safe; the 2→5 gap is a mild semantic OOD on the pooled-embedding
distribution. If smoke tests show this hurts, retrain with matching
num_deliveries.

Node-id mapping
---------------
The vendored ``HazardRoutingEnv`` uses integer node indices (0..N-1);
``to_training_graph`` relabels OSM IDs via
``nx.convert_node_labels_to_integers``. Our harness persists scenarios
with the original OSM string IDs. We bridge the two at runner init by
matching node ``(x, y)`` positions between the env's graph (loaded from
the checkpoint's ``base_graph_node_link`` snapshot) and our cohort
graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch

from ..schemas import EdgeStep, Route, Scenario, edge_key
from .base import GraphView, config_hash
from ._rl_backend import (
    DQN,
    HazardRoutingEnv,
    RAIN_KEYS,
    activate_hazards,
    apply_runtime_config,
    load_config,
    select_action,
    set_seed,
)


# Precision for position-based node matching. Graphml files round x, y to
# ~7 decimal places; 6 is safely inside that margin.
_POS_PRECISION = 6


def _position_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), _POS_PRECISION), round(float(y), _POS_PRECISION))


@dataclass
class DQNRunner:
    algorithm_id: str
    profile: str                         # balanced_HF | fast_HF | safe_HF
    checkpoint_root: Path                # .../models/rl_checkpoints
    config_path: Path                    # one representative config (arch is same across RIs)
    device: str = "cpu"
    policy_metadata: dict = field(default_factory=dict)

    # Lazy state — populated on first `run()`.
    _cfg: Optional[dict] = field(default=None, init=False, repr=False)
    _env: Optional[HazardRoutingEnv] = field(default=None, init=False, repr=False)
    _models: dict[int, torch.nn.Module] = field(default_factory=dict, init=False, repr=False)
    _osm_to_int: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _int_to_osm: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.policy_metadata = dict(self.policy_metadata)
        self.policy_metadata.setdefault("variant", "dqn_greedy_actionmask_per_ri_dispatch")
        self.policy_metadata.setdefault("profile", self.profile)
        self.policy_metadata.setdefault("checkpoint_root", str(self.checkpoint_root))
        self.policy_metadata.setdefault("config", str(self.config_path))
        self.policy_metadata.setdefault("eval_epsilon", 0.0)
        self.policy_metadata.setdefault("runner_version", "v1.0")
        self.algorithm_config_hash = config_hash(
            {"algorithm_id": self.algorithm_id, **self.policy_metadata}
        )

    # ------------------------------------------------------------------
    # Lazy construction
    # ------------------------------------------------------------------

    def _ensure_env(self, cohort_base_graph: nx.DiGraph) -> None:
        if self._env is not None:
            return

        self._cfg = load_config(str(self.config_path))
        set_seed(int(self._cfg.get("seed", 0)))
        apply_runtime_config(self._cfg)

        # Load an anchor checkpoint to extract the base-graph snapshot.
        # All checkpoints in a profile share the same base graph (they're
        # fine-tunes from the same profile pretrain).
        anchor_path = self._checkpoint_path(3)
        if not anchor_path.is_file():
            # Fall back to any available RI
            for ri in (1, 2, 4, 5):
                candidate = self._checkpoint_path(ri)
                if candidate.is_file():
                    anchor_path = candidate
                    break
            else:
                raise RuntimeError(
                    f"No checkpoints found for profile {self.profile!r} under "
                    f"{self.checkpoint_root / self.profile}"
                )
        anchor_ckpt = torch.load(
            str(anchor_path), map_location=self.device, weights_only=False
        )
        if "base_graph_node_link" not in anchor_ckpt:
            raise RuntimeError(
                f"Checkpoint {anchor_path} lacks base_graph_node_link; cannot "
                "derive the backend's int-indexed graph. Retrain with "
                "graph serialization enabled, or point the runner at a "
                "checkpoint produced by the current training script."
            )
        env_base_graph = nx.node_link_graph(
            anchor_ckpt["base_graph_node_link"], edges="edges"
        )

        env_cfg = self._cfg["environment"]
        reward_cfg = self._cfg["reward"]
        self._env = HazardRoutingEnv(
            env_base_graph,
            num_deliveries=int(env_cfg["num_deliveries"]),
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
        )

        self._build_id_maps(cohort_base_graph)

    def _build_id_maps(self, cohort_base_graph: nx.DiGraph) -> None:
        env = self._env
        assert env is not None
        env_pos_to_int: dict[tuple[float, float], int] = {}
        for int_id, pos in env.node_pos.items():
            key = _position_key(pos[0], pos[1])
            env_pos_to_int[key] = int_id

        for osm_str_id, data in cohort_base_graph.nodes(data=True):
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            key = _position_key(x, y)
            int_id = env_pos_to_int.get(key)
            if int_id is None:
                raise RuntimeError(
                    f"OSM node {osm_str_id!r} at ({x}, {y}) has no matching "
                    f"position in the checkpoint's graph. Node count: "
                    f"cohort={cohort_base_graph.number_of_nodes()}, "
                    f"env={env.num_nodes}. The cohort graph and the "
                    "checkpoint-trained graph must describe the same subgraph."
                )
            self._osm_to_int[osm_str_id] = int_id
            self._int_to_osm[int_id] = osm_str_id

    def _checkpoint_path(self, ri: int) -> Path:
        return (
            self.checkpoint_root
            / self.profile
            / f"stage_200_{self.profile}_RI{ri}_det"
            / "best_model.pt"
        )

    def _get_model(self, ri: int) -> torch.nn.Module:
        if ri not in self._models:
            assert self._env is not None and self._cfg is not None
            path = self._checkpoint_path(ri)
            if not path.is_file():
                raise RuntimeError(
                    f"Checkpoint for {self.profile} RI{ri} missing at {path}"
                )
            model_cfg = self._cfg["model"]
            model = DQN(
                self._env.state_dim,
                self._env.action_dim,
                num_nodes=self._env.num_nodes,
                num_delivery_slots=self._env.num_deliveries,
                hidden_sizes=tuple(model_cfg["hidden_sizes"]),
                node_embedding_dim=int(model_cfg.get("node_embedding_dim", 16)),
            )
            ckpt = torch.load(
                str(path), map_location=self.device, weights_only=False
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model.to(self.device)
            self._models[ri] = model
        return self._models[ri]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, scenario: Scenario, view: GraphView) -> Route:
        self._ensure_env(view.base_graph)
        env = self._env
        assert env is not None

        # Override env to match our scenario's rain level + blocked edges +
        # travel_time_map. Start from activate_hazards so edge attrs
        # (flood_score, landslide_score, edge_state) are populated, then
        # overwrite blocked + travel_time from our pre-baked scenario
        # state so the DQN's world matches exactly what the baselines see.
        ri_key = f"RI{scenario.rain_level}"
        env.G = activate_hazards(env.base_graph, ri_key)

        blocked_osm: set[tuple[str, str]] = scenario.blocked_set()

        for u_int, v_int, data in env.G.edges(data=True):
            u_str = self._int_to_osm.get(u_int)
            v_str = self._int_to_osm.get(v_int)
            if u_str is None or v_str is None:
                # Defensive — should not happen if _build_id_maps succeeded.
                continue
            # env.G is undirected; our scenario's blocked set is directed.
            # Physically a blocked road is blocked both ways, so either
            # direction appearing in our set blocks the undirected edge.
            is_blocked = (u_str, v_str) in blocked_osm or (v_str, u_str) in blocked_osm
            data["blocked"] = bool(is_blocked)
            if is_blocked:
                data["travel_time"] = None
            else:
                tt = scenario.travel_time_map.get(edge_key(u_str, v_str))
                if tt is None:
                    tt = scenario.travel_time_map.get(edge_key(v_str, u_str))
                if tt is not None:
                    data["travel_time"] = float(tt)

        # Episode state.
        start_int = self._osm_to_int[scenario.start_node]
        deliveries_int = [self._osm_to_int[d] for d in scenario.delivery_nodes]
        env.num_deliveries = len(deliveries_int)  # rebind — see module docstring
        env.current_node = start_int
        env.delivery_nodes = set(deliveries_int)
        env.completed = set()
        env.total_time = 0.0
        env.total_hazard = 0.0
        env.steps = 0
        env.rain_onehot = np.zeros(env.rain_dim, dtype=float)
        env.rain_onehot[RAIN_KEYS.index(ri_key)] = 1.0

        model = self._get_model(scenario.rain_level)

        # Inference loop.
        t0 = time.perf_counter()
        state = env._get_state()
        visit_order: list[str] = []
        visited_set: set[int] = set()
        edge_sequence: list[list[str]] = []
        per_edge: list[dict] = []
        step_idx = 0
        failure_reason: Optional[str] = None
        max_steps = scenario.max_steps

        while True:
            mask = env.get_action_mask()
            action = select_action(model, state, mask, epsilon=0.0)
            if action is None:
                failure_reason = "trapped"
                break

            prev_int = env.current_node
            next_state, _reward, done, info = env.step(action)
            next_int = env.current_node

            u_str = self._int_to_osm[prev_int]
            v_str = self._int_to_osm[next_int]
            base_data = (
                view.base_graph[u_str][v_str]
                if view.base_graph.has_edge(u_str, v_str)
                else {}
            )
            step_record = EdgeStep(
                u=u_str,
                v=v_str,
                step=step_idx,
                was_replan=False,
                travel_time=float(scenario.travel_time(u_str, v_str)),
                hazard_flood=float(base_data.get("flood_hazard", 0.0)),
                hazard_landslide=float(base_data.get("landslide_hazard", 0.0)),
                length_m=float(base_data.get("length", 0.0)),
            )
            per_edge.append(step_record.to_dict())
            edge_sequence.append([u_str, v_str])
            step_idx += 1
            state = next_state

            if next_int in env.delivery_nodes and next_int not in visited_set:
                visited_set.add(next_int)
                visit_order.append(v_str)

            if step_idx > max_steps:
                failure_reason = "timeout"
                break

            if done:
                reason = info.get("termination_reason")
                if reason == "success":
                    pass
                elif reason in ("timeout", "step_guard_timeout"):
                    failure_reason = "timeout"
                elif reason == "invalid_action":
                    failure_reason = "invalid_action"
                else:
                    failure_reason = "trapped"
                break

        wall_ms = (time.perf_counter() - t0) * 1000.0
        success = failure_reason is None
        return Route(
            scenario_id=scenario.scenario_id,
            algorithm_id=self.algorithm_id,
            algorithm_config_hash=self.algorithm_config_hash,
            visit_order=visit_order,
            edge_sequence=edge_sequence,
            per_edge=per_edge,
            success=success,
            failure_reason=failure_reason,
            replan_count=0,
            wall_time_ms=wall_ms,
            policy_metadata={
                **self.policy_metadata,
                "checkpoint_used": f"RI{scenario.rain_level}",
            },
        )
