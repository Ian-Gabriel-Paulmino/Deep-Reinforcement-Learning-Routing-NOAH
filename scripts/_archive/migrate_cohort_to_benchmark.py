"""One-shot migration: cohorts/ -> benchmarks/, cohort.json -> benchmark.json,
routes/ -> runs/.

Idempotent. Re-running on an already-migrated tree is a no-op (it logs that
each target already exists and skips).

Usage (run from the Benguet project root):
    python scripts/migrate_cohort_to_benchmark.py
    python scripts/migrate_cohort_to_benchmark.py --dry-run
    python scripts/migrate_cohort_to_benchmark.py --delete-source

By default the source ``cohorts/<id>/`` tree is preserved alongside the new
``benchmarks/<id>/`` tree for one cycle (per the rename plan §0.2). Pass
``--delete-source`` to remove ``cohorts/<id>/`` after successful migration.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger("migrate_cohort_to_benchmark")


SCHEMA_VERSION = 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cohorts_root() -> Path:
    return _project_root() / "src" / "evaluation" / "cohorts"


def _benchmarks_root() -> Path:
    return _project_root() / "src" / "evaluation" / "benchmarks"


def _translate_manifest(cohort_data: dict[str, Any], sampler_name: str) -> dict[str, Any]:
    """Translate a v1 cohort.json dict into a v2 benchmark.json dict.

    Field map (v1 -> v2):
      cohort_id           -> benchmark_id
      num_scenarios       -> n_scenarios
      num_deliveries      -> k_deliveries
      activation_mode     -> activation_strategy
      sampling_policy     -> (dropped; replaced by sampler + activation_strategy)
    Added in v2: schema_version, n_evaluations, rain_intensities, sampler,
    longitudinal.
    """
    ri_distribution = dict(cohort_data.get("ri_distribution") or {})
    ri_keys = sorted(ri_distribution.keys())
    n_scenarios = int(cohort_data.get("num_scenarios", 0))

    return {
        "benchmark_id": cohort_data.get("cohort_id"),
        "schema_version": SCHEMA_VERSION,
        "generated_at": cohort_data.get("generated_at"),
        "master_seed": int(cohort_data.get("master_seed", 0)),
        "graph_id": cohort_data.get("graph_id", ""),
        "graph_path": cohort_data.get("graph_path", ""),
        "n_scenarios": n_scenarios,
        # Pre-overhaul cohorts use stratified-by-RI sampling: total scenario
        # rows == n_scenarios (each row corresponds to one (depot, stops, RI)
        # tuple, no longitudinal duplication).
        "n_evaluations": n_scenarios,
        "k_deliveries": int(cohort_data.get("num_deliveries", 0)),
        "rain_intensities": ri_keys,
        "activation_strategy": cohort_data.get("activation_mode", "deterministic_v3"),
        "sampler": sampler_name,
        "longitudinal": False,
        "ri_distribution": ri_distribution,
        "feasibility_filtered": bool(cohort_data.get("feasibility_filtered", True)),
        "scenarios_path": cohort_data.get("scenarios_path", "scenarios.jsonl"),
    }


def _migrate_one(src: Path, dst: Path, *, dry_run: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "src": str(src),
        "dst": str(dst),
        "actions": [],
    }

    if not src.exists():
        summary["actions"].append("skip:source_missing")
        return summary

    if dst.exists():
        if (dst / "benchmark.json").exists():
            summary["actions"].append("skip:already_migrated")
            return summary
        # Partial migration -- destination exists but no manifest yet. Continue
        # so the manifest gets written; copies below check existence per file.
        summary["actions"].append("resume:partial_destination")
    else:
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
        summary["actions"].append("mkdir:dst")

    # 1. Translate cohort.json -> benchmark.json
    cohort_json = src / "cohort.json"
    benchmark_json = dst / "benchmark.json"
    if cohort_json.exists() and not benchmark_json.exists():
        with cohort_json.open("r", encoding="utf-8") as f:
            cohort_data = json.load(f)
        # Pre-overhaul cohorts always used SCC-restricted stratified sampling.
        bm_data = _translate_manifest(cohort_data, sampler_name="scc_restricted")
        if not dry_run:
            with benchmark_json.open("w", encoding="utf-8") as f:
                json.dump(bm_data, f, indent=2, sort_keys=True)
        summary["actions"].append(f"translate:{cohort_json.name}->{benchmark_json.name}")

    # 2. Copy scenarios.jsonl verbatim
    src_scenarios = src / "scenarios.jsonl"
    dst_scenarios = dst / "scenarios.jsonl"
    if src_scenarios.exists() and not dst_scenarios.exists():
        if not dry_run:
            shutil.copy2(src_scenarios, dst_scenarios)
        summary["actions"].append("copy:scenarios.jsonl")

    # 3. Rename routes/ -> runs/ (preserves contents, no field translation)
    src_routes = src / "routes"
    dst_runs = dst / "runs"
    if src_routes.exists() and not dst_runs.exists():
        if not dry_run:
            shutil.copytree(src_routes, dst_runs)
        summary["actions"].append("copy:routes/->runs/")

    # 4. Copy report/ verbatim. The legacy report uses v1 field names
    #    (cohort_id, num_scenarios, activation_mode); the verification
    #    script applies a field-rename when comparing against newly
    #    generated v2 reports.
    src_report = src / "report"
    dst_report = dst / "report"
    if src_report.exists() and not dst_report.exists():
        if not dry_run:
            shutil.copytree(src_report, dst_report)
        summary["actions"].append("copy:report/")

    # 5. Copy any auxiliary report.* snapshots (e.g. report.pre_blind/) verbatim.
    for aux in src.glob("report.*"):
        if not aux.is_dir():
            continue
        dst_aux = dst / aux.name
        if not dst_aux.exists():
            if not dry_run:
                shutil.copytree(aux, dst_aux)
            summary["actions"].append(f"copy:{aux.name}/")

    # 6. Copy .cache/ if present (the API sometimes warms a cache here)
    src_cache = src / ".cache"
    dst_cache = dst / ".cache"
    if src_cache.exists() and not dst_cache.exists():
        if not dry_run:
            shutil.copytree(src_cache, dst_cache)
        summary["actions"].append("copy:.cache/")

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying disk.",
    )
    p.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete cohorts/<id>/ after successful migration. Default is to "
        "keep the source for one cycle.",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)-7s  %(message)s",
        stream=sys.stdout,
    )

    cohorts = _cohorts_root()
    benchmarks = _benchmarks_root()
    if not cohorts.exists():
        logger.info("No cohorts/ directory at %s; nothing to migrate.", cohorts)
        return 0

    if not args.dry_run:
        benchmarks.mkdir(parents=True, exist_ok=True)

    cohort_dirs = sorted(p for p in cohorts.iterdir() if p.is_dir())
    if not cohort_dirs:
        logger.info("cohorts/ is empty; nothing to migrate.")
        return 0

    logger.info(
        "Migrating %d cohort(s) from %s -> %s%s",
        len(cohort_dirs),
        cohorts,
        benchmarks,
        " (dry-run)" if args.dry_run else "",
    )

    for src in cohort_dirs:
        dst = benchmarks / src.name
        summary = _migrate_one(src, dst, dry_run=args.dry_run)
        logger.info("  %s: %s", src.name, ", ".join(summary["actions"]) or "(no-op)")

        if (
            args.delete_source
            and not args.dry_run
            and "skip:already_migrated" not in summary["actions"]
            and (dst / "benchmark.json").exists()
        ):
            shutil.rmtree(src)
            logger.info("  %s: deleted source %s", src.name, src)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
