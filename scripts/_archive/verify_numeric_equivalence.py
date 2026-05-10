"""Compare per-(algorithm, RI) metric values between a v1 cohort report and a
v2 benchmark report. Asserts that the rename in Stage 1 did not change any
numeric outputs (within float tolerance).

The v1 report has top-level keys ``cohort_id``, ``num_scenarios``,
``activation_mode``; the v2 report has ``benchmark_id``, ``n_evaluations``,
``activation_strategy``. We translate the v1 -> v2 names and then walk the
``algorithms[<algo>][metrics][<metric>][<bucket>]`` tree comparing
``mean``, ``stdev``, ``min``, ``max``, ``n``.

Usage:
    python scripts/verify_numeric_equivalence.py \\
        --v1 src/evaluation/cohorts/la_trinidad_mini/report/metrics.json \\
        --v2 src/evaluation/benchmarks/la_trinidad_mini/report/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Tolerances are tight: we should be running the SAME computation on the SAME
# inputs. Any nonzero diff above floating-point noise is a regression.
ABS_TOL = 1e-9
REL_TOL = 1e-9


def _almost_equal(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    diff = abs(a - b)
    if diff <= ABS_TOL:
        return True
    scale = max(abs(a), abs(b))
    return diff <= REL_TOL * scale


def _diff_bucket(
    algo: str, metric: str, bucket: str, b1: dict[str, Any], b2: dict[str, Any]
) -> list[str]:
    diffs: list[str] = []
    for key in ("n", "mean", "stdev", "min", "max"):
        v1 = b1.get(key)
        v2 = b2.get(key)
        if key == "n":
            if v1 != v2:
                diffs.append(f"{algo}.{metric}.{bucket}.n: v1={v1} v2={v2}")
            continue
        if not _almost_equal(v1, v2):
            diffs.append(f"{algo}.{metric}.{bucket}.{key}: v1={v1} v2={v2}")
    return diffs


def compare(v1_path: Path, v2_path: Path) -> int:
    with v1_path.open("r", encoding="utf-8") as f:
        v1 = json.load(f)
    with v2_path.open("r", encoding="utf-8") as f:
        v2 = json.load(f)

    diffs: list[str] = []

    # Top-level structural checks (renamed fields).
    v1_id = v1.get("cohort_id") or v1.get("benchmark_id")
    v2_id = v2.get("benchmark_id") or v2.get("cohort_id")
    if v1_id != v2_id:
        diffs.append(f"top.id: v1={v1_id} v2={v2_id}")

    v1_strategy = v1.get("activation_mode") or v1.get("activation_strategy")
    v2_strategy = v2.get("activation_strategy") or v2.get("activation_mode")
    if v1_strategy != v2_strategy:
        diffs.append(f"top.activation: v1={v1_strategy} v2={v2_strategy}")

    v1_n = v1.get("num_scenarios") or v1.get("n_evaluations")
    v2_n = v2.get("n_evaluations") or v2.get("num_scenarios")
    if v1_n != v2_n:
        diffs.append(f"top.n_scenarios: v1={v1_n} v2={v2_n}")

    if v1.get("graph_id") != v2.get("graph_id"):
        diffs.append(f"top.graph_id: v1={v1.get('graph_id')} v2={v2.get('graph_id')}")

    # Per-algorithm metric comparison.
    v1_algos = v1.get("algorithms") or {}
    v2_algos = v2.get("algorithms") or {}

    only_v1 = sorted(set(v1_algos) - set(v2_algos))
    only_v2 = sorted(set(v2_algos) - set(v1_algos))
    common = sorted(set(v1_algos) & set(v2_algos))

    if only_v1:
        diffs.append(f"algorithms.only_in_v1: {only_v1}")
    if only_v2:
        diffs.append(f"algorithms.only_in_v2: {only_v2}")

    for algo in common:
        m1 = (v1_algos[algo] or {}).get("metrics") or {}
        m2 = (v2_algos[algo] or {}).get("metrics") or {}
        metric_keys = sorted(set(m1) | set(m2))
        for metric in metric_keys:
            buckets1 = m1.get(metric) or {}
            buckets2 = m2.get(metric) or {}
            bucket_keys = sorted(set(buckets1) | set(buckets2))
            for bucket in bucket_keys:
                b1 = buckets1.get(bucket) or {}
                b2 = buckets2.get(bucket) or {}
                diffs.extend(_diff_bucket(algo, metric, bucket, b1, b2))

    if diffs:
        print(f"FAIL: {len(diffs)} differences")
        for d in diffs[:50]:
            print(f"  {d}")
        if len(diffs) > 50:
            print(f"  ... ({len(diffs) - 50} more)")
        return 1

    print(
        f"PASS: {len(common)} algorithm(s) match across {sum(1 for _ in v1_algos)} "
        f"v1 vs {sum(1 for _ in v2_algos)} v2 entries; all per-(algo, RI) "
        f"metric values agree within tol={ABS_TOL} (abs) / {REL_TOL} (rel)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1", required=True, type=Path, help="legacy cohort metrics.json")
    p.add_argument("--v2", required=True, type=Path, help="new benchmark metrics.json")
    args = p.parse_args(argv)

    if not args.v1.exists():
        print(f"ERROR: v1 file not found: {args.v1}", file=sys.stderr)
        return 2
    if not args.v2.exists():
        print(f"ERROR: v2 file not found: {args.v2}", file=sys.stderr)
        return 2

    return compare(args.v1, args.v2)


if __name__ == "__main__":
    raise SystemExit(main())
