"""Dataclasses for benchmark / scenario / route plus JSONL I/O helpers.

The harness persists state as newline-delimited JSON so each stage can
stream input without loading the whole file, adding a new scenario or
route never rewrites existing ones, and artifacts are peer-reviewable
text.

Edge identifiers are encoded ``"u|v"`` because JSON object keys must be
strings. The pipe character never appears in OSM node ids.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional


EDGE_SEP = "|"

BENCHMARK_SCHEMA_VERSION = 2


def edge_key(u: str, v: str) -> str:
    return f"{u}{EDGE_SEP}{v}"


def parse_edge_key(k: str) -> tuple[str, str]:
    u, v = k.split(EDGE_SEP, 1)
    return u, v


# ---------------------------------------------------------------------------
# Benchmark metadata (schema v2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    """Top-level manifest for a generated benchmark.

    Persisted as ``benchmark.json`` at the root of a benchmark directory
    alongside ``scenarios.jsonl``, ``runs/<algorithm>.jsonl`` and
    ``report/metrics.json``.
    """

    benchmark_id: str
    schema_version: int
    generated_at: str
    master_seed: int
    graph_id: str
    graph_path: str
    n_scenarios: int
    n_evaluations: int
    k_deliveries: int
    rain_intensities: list[str]
    activation_strategy: str
    sampler: str
    longitudinal: bool
    ri_distribution: dict[str, int]
    feasibility_filtered: bool
    scenarios_path: str = "scenarios.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    graph_id: str
    rain_level: int
    activation_mode: str
    activation_seed: int
    start_node: str
    delivery_nodes: list[str]
    blocked_edges: list[list[str]]  # [[u, v], ...]
    travel_time_map: dict[str, float]  # "u|v" -> minutes
    max_steps: int
    num_deliveries: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def blocked_set(self) -> set[tuple[str, str]]:
        return {(u, v) for u, v in self.blocked_edges}

    def travel_time(self, u: str, v: str) -> float:
        return self.travel_time_map[edge_key(u, v)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=d["scenario_id"],
            graph_id=d["graph_id"],
            rain_level=int(d["rain_level"]),
            activation_mode=d["activation_mode"],
            activation_seed=int(d["activation_seed"]),
            start_node=d["start_node"],
            delivery_nodes=list(d["delivery_nodes"]),
            blocked_edges=[list(e) for e in d["blocked_edges"]],
            travel_time_map={k: float(v) for k, v in d["travel_time_map"].items()},
            max_steps=int(d["max_steps"]),
            num_deliveries=int(d["num_deliveries"]),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeStep:
    u: str
    v: str
    step: int
    was_replan: bool
    travel_time: float
    hazard_flood: float
    hazard_landslide: float
    length_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Failure reason vocabulary shared by all policies (see plan §Locked design
# decisions, item 5). ``invalid_action`` is a bug signal — a well-behaved
# policy should never produce it because blocked edges are filtered from the
# valid action set. ``blocked`` is specific to the blind-NNA variants which
# commit to a planned path and fail on first blocked-edge encounter.
FAILURE_REASONS = frozenset(
    {"trapped", "timeout", "invalid_action", "no_route", "blocked"}
)


@dataclass(frozen=True)
class Route:
    scenario_id: str
    algorithm_id: str
    algorithm_config_hash: str
    visit_order: list[str]
    edge_sequence: list[list[str]]
    per_edge: list[dict[str, Any]]  # list of EdgeStep.to_dict()
    success: bool
    failure_reason: Optional[str]
    replan_count: int
    wall_time_ms: float
    policy_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Route":
        return cls(
            scenario_id=d["scenario_id"],
            algorithm_id=d["algorithm_id"],
            algorithm_config_hash=d["algorithm_config_hash"],
            visit_order=list(d["visit_order"]),
            edge_sequence=[list(e) for e in d["edge_sequence"]],
            per_edge=[dict(step) for step in d["per_edge"]],
            success=bool(d["success"]),
            failure_reason=d.get("failure_reason"),
            replan_count=int(d.get("replan_count", 0)),
            wall_time_ms=float(d.get("wall_time_ms", 0.0)),
            policy_metadata=dict(d.get("policy_metadata", {})),
        )


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":"), sort_keys=True))
            f.write("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_benchmark(benchmark_dir: Path) -> Benchmark:
    bm_path = benchmark_dir / "benchmark.json"
    with bm_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return Benchmark(
        benchmark_id=d["benchmark_id"],
        schema_version=int(d.get("schema_version", BENCHMARK_SCHEMA_VERSION)),
        generated_at=d["generated_at"],
        master_seed=int(d["master_seed"]),
        graph_id=d["graph_id"],
        graph_path=d["graph_path"],
        n_scenarios=int(d["n_scenarios"]),
        n_evaluations=int(d["n_evaluations"]),
        k_deliveries=int(d["k_deliveries"]),
        rain_intensities=list(d["rain_intensities"]),
        activation_strategy=d["activation_strategy"],
        sampler=d["sampler"],
        longitudinal=bool(d["longitudinal"]),
        ri_distribution={k: int(v) for k, v in d["ri_distribution"].items()},
        feasibility_filtered=bool(d["feasibility_filtered"]),
        scenarios_path=d.get("scenarios_path", "scenarios.jsonl"),
    )


def write_benchmark(benchmark_dir: Path, benchmark: Benchmark) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    with (benchmark_dir / "benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(benchmark.to_dict(), f, indent=2, sort_keys=True)


def read_scenarios(benchmark_dir: Path) -> Iterator[Scenario]:
    benchmark = read_benchmark(benchmark_dir)
    for rec in read_jsonl(benchmark_dir / benchmark.scenarios_path):
        yield Scenario.from_dict(rec)


def read_routes(path: Path) -> Iterator[Route]:
    for rec in read_jsonl(path):
        yield Route.from_dict(rec)
