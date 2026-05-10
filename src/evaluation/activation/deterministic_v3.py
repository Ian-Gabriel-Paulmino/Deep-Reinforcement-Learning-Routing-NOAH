"""DeterministicV3 activation strategy.

Refactored from the inline ``compute_blocked_edges`` /
``compute_travel_time_map`` functions in ``scenario_generator.py``. Behaviour
is preserved byte-for-byte: this strategy is the only legal activation
under the locked decision in ``CLAUDE.md`` (deterministic with shifted
thresholds; not probabilistic).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx

from ..schemas import edge_key
from ..scenario_generator import (
    DEFAULT_ALPHA_FLOOD,
    DEFAULT_ALPHA_LANDSLIDE,
    compute_blocked_edges as _compute_blocked_edges_fn,
    compute_travel_time_map as _compute_travel_time_map_fn,
    load_rain_levels_from_config,
    load_time_weights_from_config,
)


@dataclass
class DeterministicV3Strategy:
    """Threshold-based deterministic activation.

    Reads ``hazard.rain_levels`` from a ``_det`` training config and uses the
    flood/landslide block thresholds + per-edge hazard scores to produce a
    deterministic ``blocked_edges`` set. ``compute_travel_time_map`` applies
    the manuscript §B drag formula
    ``T_e = base_time / speed_mult * (1 + alpha_f * H_f + alpha_l * H_l)``.
    """

    config_path: Path
    alpha_flood: Optional[float] = None
    alpha_landslide: Optional[float] = None
    name: str = "deterministic_v3"

    def __post_init__(self) -> None:
        self._rain_levels = load_rain_levels_from_config(self.config_path)
        cfg_af, cfg_al = load_time_weights_from_config(self.config_path)
        self._alpha_flood = (
            self.alpha_flood
            if self.alpha_flood is not None
            else (cfg_af if cfg_af is not None else DEFAULT_ALPHA_FLOOD)
        )
        self._alpha_landslide = (
            self.alpha_landslide
            if self.alpha_landslide is not None
            else (cfg_al if cfg_al is not None else DEFAULT_ALPHA_LANDSLIDE)
        )

    def rain_keys(self) -> list[str]:
        return sorted(self._rain_levels.keys())

    def activation_seed(self, master_seed: int, ri_key: str) -> int:
        return master_seed + int(ri_key.replace("RI", ""))

    def alpha_flood_resolved(self) -> float:
        return self._alpha_flood

    def alpha_landslide_resolved(self) -> float:
        return self._alpha_landslide

    def compute_blocked_edges(
        self,
        graph: nx.DiGraph,
        ri_key: str,
        scenario_seed: int | None = None,
    ) -> set[tuple[str, str]]:
        rain_cfg = self._rain_levels[ri_key]
        # In deterministic mode the activation_seed is unused; we still pass
        # something to preserve the legacy function signature.
        seed = scenario_seed if scenario_seed is not None else 0
        return _compute_blocked_edges_fn(graph, rain_cfg, self.name, seed)

    def compute_travel_time_map(
        self,
        graph: nx.DiGraph,
        ri_key: str,
    ) -> dict[str, float]:
        rain_cfg = self._rain_levels[ri_key]
        return _compute_travel_time_map_fn(
            graph, rain_cfg, self._alpha_flood, self._alpha_landslide
        )


__all__ = ["DeterministicV3Strategy", "edge_key"]
