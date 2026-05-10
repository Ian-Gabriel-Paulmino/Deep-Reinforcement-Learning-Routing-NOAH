"""Fair evaluation harness for hazard-aware routing.

Three-stage pipeline:
  scenario_generator.py  -> benchmarks/<id>/scenarios.jsonl
  run_policies.py        -> benchmarks/<id>/runs/<algorithm_id>.jsonl
  evaluator.py           -> benchmarks/<id>/report/metrics.json

See ``docs/EVALUATION_FRAMEWORK.md`` for the full design.
"""
