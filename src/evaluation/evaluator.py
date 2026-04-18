"""Stage 3: aggregate metrics from saved routes and emit a report.

Reads ``cohort.json`` + every ``routes/*.jsonl`` in the cohort dir, runs each
metric in ``metrics.REGISTRY`` over every ``(scenario, route)`` pair, and
aggregates by ``(algorithm_id, rain_level)``. Writes ``report/metrics.json``.

Usage (run from the Benguet project root):
    python -m src.evaluation.evaluator \\
        --cohort-dir src/evaluation/cohorts/la_trinidad_mini
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .metrics import REGISTRY
from .schemas import Route, Scenario, read_cohort, read_jsonl, read_scenarios


logger = logging.getLogger("evaluation.evaluator")


def _scenarios_by_id(cohort_dir: Path) -> dict[str, Scenario]:
    return {s.scenario_id: s for s in read_scenarios(cohort_dir)}


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


def evaluate(cohort_dir: Path, out_dir: Optional[Path] = None) -> dict:
    cohort = read_cohort(cohort_dir)
    scenarios = _scenarios_by_id(cohort_dir)

    out_dir = out_dir or (cohort_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    route_dir = cohort_dir / "routes"
    if not route_dir.exists():
        raise FileNotFoundError(f"No routes/ directory at {route_dir}")

    report: dict = {
        "cohort_id": cohort.cohort_id,
        "num_scenarios": cohort.num_scenarios,
        "graph_id": cohort.graph_id,
        "activation_mode": cohort.activation_mode,
        "algorithms": {},
    }

    for routes_file in sorted(route_dir.glob("*.jsonl")):
        algorithm_id = routes_file.stem
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

        for rec in read_jsonl(routes_file):
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
            for metric_name, metric_fn in REGISTRY.items():
                val = metric_fn(scenario, route)
                per_metric[metric_name]["all"].append(val)
                per_metric[metric_name][ri_key].append(val)

        per_metric_stats: dict[str, dict[str, dict]] = {}
        for metric_name, buckets in per_metric.items():
            per_metric_stats[metric_name] = {
                bucket: _safe_stats(values) for bucket, values in buckets.items()
            }

        report["algorithms"][algorithm_id] = {
            "routes_file": str(routes_file.relative_to(cohort_dir)),
            "routes_scored": scored,
            "failure_counts": dict(failure_counts),
            "replan_count_stats": _safe_stats([float(r) for r in replan_counts]),
            "wall_time_ms_stats": _safe_stats(wall_times),
            "metrics": per_metric_stats,
        }

    report_path = out_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {report_path}")

    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    print()
    print("=" * 72)
    print(f"Cohort: {report['cohort_id']}  ({report['num_scenarios']} scenarios)")
    print(f"Graph:  {report['graph_id']}")
    print(f"Mode:   {report['activation_mode']}")
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


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stage 3: evaluate routes and emit report.")
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    evaluate(cohort_dir=args.cohort_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
