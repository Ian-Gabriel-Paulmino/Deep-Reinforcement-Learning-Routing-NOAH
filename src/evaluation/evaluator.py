"""Stage 3: aggregate metrics from saved runs and emit a report.

Reads ``benchmark.json`` + every ``runs/*.jsonl`` in the benchmark dir,
runs each metric in ``metrics.REGISTRY`` over every ``(scenario, route)``
pair, and aggregates by ``(algorithm_id, rain_level)``. Writes
``report/metrics.json``.

Usage (run from the Benguet project root):
    python -m src.evaluation.evaluator \\
        --benchmark-dir src/evaluation/benchmarks/la_trinidad_mini
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .metrics import REGISTRY
from .metrics.robustness import compute_robustness_ratio
from .schemas import Route, Scenario, read_benchmark, read_jsonl, read_scenarios


logger = logging.getLogger("evaluation.evaluator")


def _scenarios_by_id(benchmark_dir: Path) -> dict[str, Scenario]:
    return {s.scenario_id: s for s in read_scenarios(benchmark_dir)}


def _safe_stats(values: list[float]) -> dict[str, float]:
    filtered = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not filtered:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(filtered),
        "mean": float(statistics.fmean(filtered)),
        "stdev": float(statistics.pstdev(filtered)) if len(filtered) > 1 else 0.0,
        "min": float(min(filtered)),
        "max": float(max(filtered)),
    }


def _attach_robustness(report: dict) -> None:
    """Populate ``report["algorithms"][algo_id]["robustness"]`` in place.

    Robustness = 1 - σ/μ across per-RI means, per (algorithm, metric).
    See `metrics/robustness.py` for the definition and manuscript §3.6.1 E.
    """
    for algo_id, info in report["algorithms"].items():
        robustness: dict[str, Optional[float]] = {}
        for metric_name, buckets in info["metrics"].items():
            ri_means = [
                buckets[ri]["mean"]
                for ri in buckets
                if ri != "all" and buckets[ri].get("mean") is not None
            ]
            robustness[metric_name] = compute_robustness_ratio(ri_means)
        info["robustness"] = robustness


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


_RAW_CSV_FIXED_COLS = (
    "scenario_id",
    "RI",
    "algorithm_id",
    "failure_reason",
    "replan_count",
)


def _write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Per-episode raw metric values — one row per (scenario, algorithm).

    Column order: fixed prefix (``scenario_id``, ``RI``, ``algorithm_id``,
    ``failure_reason``, ``replan_count``) followed by every metric in the
    :data:`~src.evaluation.metrics.REGISTRY` order. ``NaN`` values are written
    as empty strings so Excel / pandas ingest cleanly.
    """
    fieldnames = list(_RAW_CSV_FIXED_COLS) + list(REGISTRY.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {k: row.get(k, "") for k in fieldnames}
            for metric_name in REGISTRY:
                v = clean.get(metric_name)
                if isinstance(v, float) and math.isnan(v):
                    clean[metric_name] = ""
            writer.writerow(clean)
    logger.info(f"Wrote {path} ({len(rows)} rows)")


_OVERALL_CSV_FIXED_COLS = ("algorithm_id", "bucket", "n")


def _write_overall_csv(path: Path, report: dict) -> None:
    """Aggregated stats per (algorithm, bucket) — wide format.

    Buckets: ``RI1`` .. ``RI5`` plus ``"all"``. Per-metric columns: ``_mean``,
    ``_stdev``, ``_min``, ``_max``. Robustness columns are populated only on
    the ``bucket == "all"`` row (blank elsewhere).
    """
    stat_suffixes = ("mean", "stdev", "min", "max")
    metric_cols: list[str] = []
    for metric_name in REGISTRY:
        for suffix in stat_suffixes:
            metric_cols.append(f"{metric_name}_{suffix}")

    robust_cols = [f"robustness_{m}" for m in REGISTRY]

    fieldnames = (
        list(_OVERALL_CSV_FIXED_COLS)
        + metric_cols
        + robust_cols
        + ["failure_counts"]
    )

    rows: list[dict[str, Any]] = []
    for algo_id, info in report["algorithms"].items():
        # Discover buckets; ensure deterministic ordering RI1..RI5 then "all".
        all_buckets: set[str] = set()
        for buckets in info["metrics"].values():
            all_buckets.update(buckets.keys())
        ri_buckets = sorted(b for b in all_buckets if b != "all")
        ordered_buckets = ri_buckets + (["all"] if "all" in all_buckets else [])

        for bucket in ordered_buckets:
            row: dict[str, Any] = {
                "algorithm_id": algo_id,
                "bucket": bucket,
            }
            n_values: list[int] = []
            for metric_name in REGISTRY:
                stats = info["metrics"].get(metric_name, {}).get(bucket, {})
                for suffix in stat_suffixes:
                    v = stats.get(suffix)
                    row[f"{metric_name}_{suffix}"] = "" if v is None else v
                n = stats.get("n")
                if isinstance(n, int):
                    n_values.append(n)
            row["n"] = max(n_values) if n_values else 0

            if bucket == "all":
                for metric_name in REGISTRY:
                    v = info.get("robustness", {}).get(metric_name)
                    row[f"robustness_{metric_name}"] = "" if v is None else v
                fc = info.get("failure_counts", {})
                if fc:
                    row["failure_counts"] = ";".join(
                        f"{k}={v}" for k, v in sorted(fc.items())
                    )
                else:
                    row["failure_counts"] = ""
            else:
                for metric_name in REGISTRY:
                    row[f"robustness_{metric_name}"] = ""
                row["failure_counts"] = ""

            rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info(f"Wrote {path} ({len(rows)} rows)")


def evaluate(benchmark_dir: Path, out_dir: Optional[Path] = None) -> dict:
    benchmark = read_benchmark(benchmark_dir)
    scenarios = _scenarios_by_id(benchmark_dir)

    out_dir = out_dir or (benchmark_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = benchmark_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs/ directory at {runs_dir}")

    report: dict = {
        "benchmark_id": benchmark.benchmark_id,
        "schema_version": benchmark.schema_version,
        "n_evaluations": benchmark.n_evaluations,
        "n_scenarios": benchmark.n_scenarios,
        "graph_id": benchmark.graph_id,
        "activation_strategy": benchmark.activation_strategy,
        "sampler": benchmark.sampler,
        "longitudinal": benchmark.longitudinal,
        "algorithms": {},
    }

    raw_rows: list[dict[str, Any]] = []

    for runs_file in sorted(runs_dir.glob("*.jsonl")):
        algorithm_id = runs_file.stem
        logger.info(f"Scoring {algorithm_id}")

        # {metric_name: {ri_key: [values]}}  plus an "all" bucket
        per_metric: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # diagnostic counters
        failure_counts: dict[str, int] = defaultdict(int)
        replan_counts: list[int] = []
        wall_times: list[float] = []
        scored = 0

        for rec in read_jsonl(runs_file):
            route = Route.from_dict(rec)
            scenario = scenarios.get(route.scenario_id)
            if scenario is None:
                logger.warning(f"  route references unknown scenario {route.scenario_id}")
                continue
            ri_key = f"RI{scenario.rain_level}"
            scored += 1
            if not route.success:
                failure_counts[route.failure_reason or "unknown"] += 1
            replan_counts.append(int(route.replan_count))
            wall_times.append(float(route.wall_time_ms))
            row_metrics: dict[str, float] = {}
            for metric_name, metric_fn in REGISTRY.items():
                val = metric_fn(scenario, route)
                per_metric[metric_name]["all"].append(val)
                per_metric[metric_name][ri_key].append(val)
                row_metrics[metric_name] = val

            raw_rows.append(
                {
                    "scenario_id": route.scenario_id,
                    "RI": ri_key,
                    "algorithm_id": algorithm_id,
                    "failure_reason": route.failure_reason or "",
                    "replan_count": int(route.replan_count),
                    **row_metrics,
                }
            )

        per_metric_stats: dict[str, dict[str, dict]] = {}
        for metric_name, buckets in per_metric.items():
            per_metric_stats[metric_name] = {
                bucket: _safe_stats(values) for bucket, values in buckets.items()
            }

        report["algorithms"][algorithm_id] = {
            "runs_file": str(runs_file.relative_to(benchmark_dir)),
            "routes_scored": scored,
            "failure_counts": dict(failure_counts),
            "replan_count_stats": _safe_stats([float(r) for r in replan_counts]),
            "wall_time_ms_stats": _safe_stats(wall_times),
            "metrics": per_metric_stats,
        }

    _attach_robustness(report)

    report_path = out_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {report_path}")

    _write_raw_csv(out_dir / "raw_metrics.csv", raw_rows)
    _write_overall_csv(out_dir / "overall_metrics.csv", report)

    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    print()
    print("=" * 72)
    print(
        f"Benchmark: {report['benchmark_id']}  "
        f"({report.get('n_evaluations', report.get('n_scenarios', 0))} scenarios)"
    )
    print(f"Graph:     {report['graph_id']}")
    print(f"Strategy:  {report['activation_strategy']}")
    print(f"Sampler:   {report.get('sampler', 'scc_restricted')}")
    print("=" * 72)
    for algo_id, info in report["algorithms"].items():
        print(f"\n  {algo_id}  (scored {info['routes_scored']})")
        m = info["metrics"]
        sr = m.get("success", {}).get("all", {})
        tt = m.get("travel_time", {}).get("all", {})
        hz = m.get("hazard_exposure", {}).get("all", {})
        print(
            f"    success_rate     = {(sr.get('mean') or 0.0) * 100:6.2f}%  "
            f"({sr.get('n', 0)} episodes)"
        )
        if tt.get("n"):
            print(
                f"    travel_time(min) = mean={tt['mean']:.1f}  "
                f"std={tt['stdev']:.1f}  over {tt['n']} successful"
            )
        if hz.get("n"):
            print(
                f"    hazard_exposure  = mean={hz['mean']:.2f}  "
                f"std={hz['stdev']:.2f}  over {hz['n']} successful"
            )
        replan = info["replan_count_stats"]
        if replan.get("n"):
            print(
                f"    replan_count     = mean={replan['mean']:.2f}  "
                f"max={replan['max']:.0f}"
            )
        if info["failure_counts"]:
            reasons = ", ".join(
                f"{k}={v}" for k, v in sorted(info["failure_counts"].items())
            )
            print(f"    failures: {reasons}")

        # Per-RI breakdown for success rate
        sr_per_ri = m.get("success", {})
        ri_keys = sorted(k for k in sr_per_ri if k != "all")
        if ri_keys:
            ri_line = "    by RI:   " + "  ".join(
                f"{ri}={(sr_per_ri[ri].get('mean') or 0.0) * 100:5.1f}%"
                for ri in ri_keys
            )
            print(ri_line)

        # Robustness (1 - σ/μ across RI means) per metric
        robustness = info.get("robustness", {})
        rob_parts = []
        for metric_name in ("success", "travel_time", "hazard_exposure"):
            val = robustness.get(metric_name)
            if val is not None:
                rob_parts.append(f"{metric_name}={val:.3f}")
        if rob_parts:
            print("    robustness:  " + "  ".join(rob_parts))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stage 3: evaluate runs and emit report.")
    p.add_argument("--benchmark-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    evaluate(benchmark_dir=args.benchmark_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
