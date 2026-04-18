"""Fair evaluation harness for hazard-aware routing.

Three-stage pipeline:
  Stage 1  scenario_generator.py  -> cohorts/<id>/scenarios.jsonl
  Stage 2  run_policies.py        -> cohorts/<id>/routes/<algorithm_id>.jsonl
  Stage 3  evaluator.py           -> cohorts/<id>/report/metrics.json

See ``.claude/plans/do-a-deep-dive-generic-simon.md`` for the full design.
"""
