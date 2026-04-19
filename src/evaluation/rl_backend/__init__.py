"""Vendored RL backend — DQN training/inference module used by the evaluation harness.

Importers should prefer the stable re-exports in
``src/evaluation/runners/_rl_backend.py`` rather than reaching into
``rl_routing_wCUDA_wCheckP`` directly; that keeps downstream code free
of the vendored filename.
"""
