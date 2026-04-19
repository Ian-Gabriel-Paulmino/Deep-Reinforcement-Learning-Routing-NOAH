"""Robustness metric (manuscript §3.6.1 E).

    Robustness(metric) = 1 - σ(metric across RI) / μ(metric across RI)

Unlike ``success``, ``travel_time``, ``hazard_exposure``, this is a
**second-order** metric: it consumes per-RI means already computed by
the evaluator rather than per-route values. So it is NOT registered in
``metrics.REGISTRY`` (which is the per-route dispatch table); instead
the evaluator calls :func:`compute_robustness_ratio` once per
``(algorithm, metric)`` pair after the per-route loop finishes.

Interpretation: a score near 1.0 means the metric's mean is consistent
across RI1..RI5 (the policy degrades gracefully); a low or negative
score means the policy's performance varies sharply with rainfall.
Curriculum-trained DQNs are expected to have higher robustness than
hazard-blind baselines at high RI — this metric validates that claim.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Optional


def compute_robustness_ratio(
    ri_means: Iterable[float],
    *,
    eps: float = 1e-9,
) -> Optional[float]:
    """Return ``1 - σ/μ`` across the provided per-RI means.

    Returns ``None`` when:
    - fewer than 2 RI means are provided (can't compute variance),
    - any mean is NaN (metric was not applicable — e.g., all episodes
      failed at that RI),
    - ``|μ| < eps`` (would divide by zero).

    ``σ`` is the population standard deviation over the supplied means
    — matching ``_safe_stats`` in ``evaluator.py``.
    """
    cleaned = [
        float(m) for m in ri_means
        if m is not None and not (isinstance(m, float) and math.isnan(m))
    ]
    if len(cleaned) < 2:
        return None
    mu = statistics.fmean(cleaned)
    if abs(mu) < eps:
        return None
    sd = statistics.pstdev(cleaned)
    return 1.0 - (sd / mu)
