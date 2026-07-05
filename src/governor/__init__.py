"""Overshoot Governor: token-budget admission control for multi-agent systems.

One policy core, two adapters. The core is a framework-agnostic policy
engine -- pure Python, no I/O, fully unit-testable:
  - ledger:    NaiveLedger (failing baseline), AtomicLedger (reserve/reconcile)
  - estimator: OutputEstimator (empirical p90 of output tokens per task key)
  - quota:     QuotaNode (hierarchical lease tree for subagent spawning)

Both adapters expose the SAME ledger, so enforcement semantics live in one
place and cannot drift between surfaces:
  - governor.adk_plugin: ADK 2.x Runner plugin (primary enforcement point --
    covers every agent and every spawned subagent of the Runner)
  - governor.mcp_server: the ledger over MCP (cross-runtime governance)

Adapters are imported lazily: the core works wherever Python does.
"""

from .appeals import AppealsDesk, AppealsLog
from .estimator import OutputEstimator
from .judge import MissionJudge
from .ledger import AtomicLedger, NaiveLedger, Reservation
from .quota import QuotaError, QuotaNode

__all__ = [
    "AppealsDesk",
    "AppealsLog",
    "MissionJudge",
    "AtomicLedger",
    "NaiveLedger",
    "Reservation",
    "OutputEstimator",
    "QuotaError",
    "QuotaNode",
]
