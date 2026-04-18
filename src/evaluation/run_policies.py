"""Stage 2: run policies against a committed cohort.

Reads ``cohorts/<cohort_id>/scenarios.jsonl``, runs every configured policy,
and writes one ``routes/<algorithm_id>.jsonl`` per policy. The base graph is
loaded once and reused across scenarios.

Usage (run from the Benguet project root):
    python -m src.evaluation.run_policies \\
        --cohort-dir src/evaluation/cohorts/la_trinidad_mini \\
        --algorithms NNA-Dijkstra
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from .runners.base import build_graph_view
from .runners.nna import NNADijkstra
from .scenario_generator import load_graph
from .schemas import read_cohort, read_scenarios, write_jsonl


logger = logging.getLogger("evaluation.run_policies")


POLICY_FACTORIES = {
    "NNA-Dijkstra": lambda: NNADijkstra(),
}


def run_policies(
    cohort_dir: Path,
    algorithm_ids: list[str],
    out_dir: Optional[Path] = None,
) -> None:
    cohort = read_cohort(cohort_dir)
    out_dir = out_dir or (cohort_dir / "routes")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cohort: {cohort.cohort_id}  ({cohort.num_scenarios} scenarios)")
    logger.info(f"Graph:  {cohort.graph_path}")

    base_graph = load_graph(Path(cohort.graph_path))

    scenarios = list(read_scenarios(cohort_dir))

    for algorithm_id in algorithm_ids:
        if algorithm_id not in POLICY_FACTORIES:
            raise ValueError(
                f"Unknown algorithm {algorithm_id!r}. "
                f"Available: {sorted(POLICY_FACTORIES)}"
            )
        policy = POLICY_FACTORIES[algorithm_id]()
        logger.info(f"\nRunning {algorithm_id} ({policy.algorithm_config_hash})")

        routes = []
        t0 = time.perf_counter()
        for i, scenario in enumerate(scenarios):
            view = build_graph_view(base_graph, scenario)
            route = policy.run(scenario, view)
            routes.append(route.to_dict())
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"  [{algorithm_id}] {i+1}/{len(scenarios)}")
        elapsed = time.perf_counter() - t0

        out_path = out_dir / f"{algorithm_id}.jsonl"
        write_jsonl(out_path, routes)
        logger.info(
            f"  {algorithm_id}: {len(routes)} routes in {elapsed:.2f}s "
            f"-> {out_path}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stage 2: run policies against a cohort.")
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["NNA-Dijkstra"],
        help=f"Available: {sorted(POLICY_FACTORIES)}",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    run_policies(
        cohort_dir=args.cohort_dir,
        algorithm_ids=args.algorithms,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
