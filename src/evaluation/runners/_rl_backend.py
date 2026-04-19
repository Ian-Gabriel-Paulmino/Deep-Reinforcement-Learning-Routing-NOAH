"""Stable re-exports of the vendored RL backend.

Downstream code (``runners/dqn.py``) imports the symbols it needs from
this module instead of reaching into the vendored file directly. That
keeps call sites free of the vendored filename and gives us one place
to adjust if the backend layout shifts.

Hard-depends on torch — see the ``dqn`` extra in ``pyproject.toml``.
Only ``runners/dqn.py`` imports this module, so NNA-only workflows
keep working without torch installed.
"""

from __future__ import annotations

from ..rl_backend import rl_routing_wCUDA_wCheckP as _rl


DQN = _rl.DQN
HazardRoutingEnv = _rl.HazardRoutingEnv
select_action = _rl.select_action
load_config = _rl.load_config
apply_runtime_config = _rl.apply_runtime_config
activate_hazards = _rl.activate_hazards
set_seed = _rl.set_seed
RAIN_KEYS = _rl.RAIN_KEYS


__all__ = [
    "DQN",
    "HazardRoutingEnv",
    "select_action",
    "load_config",
    "apply_runtime_config",
    "activate_hazards",
    "set_seed",
    "RAIN_KEYS",
]
