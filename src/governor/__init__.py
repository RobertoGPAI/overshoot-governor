"""Overshoot Governor: token-budget admission control for multi-agent systems.

Core pieces (no ADK dependency):
  - ledger:    NaiveLedger (failing baseline), AtomicLedger (reserve/reconcile)
  - estimator: OutputEstimator (empirical p90 of output tokens per task key)
  - quota:     QuotaNode (hierarchical lease tree for subagent spawning)

ADK 2.x integration lives in governor.adk_plugin (imported lazily so the core
stays usable in a plain Kaggle notebook).
"""

from .estimator import OutputEstimator
from .ledger import AtomicLedger, NaiveLedger, Reservation
from .quota import QuotaError, QuotaNode

__all__ = [
    "AtomicLedger",
    "NaiveLedger",
    "Reservation",
    "OutputEstimator",
    "QuotaError",
    "QuotaNode",
]
