"""Central token-budget ledgers.

Two implementations of the same interface:

- ``NaiveLedger``   -- check-then-act against *spent* tokens only. It ignores
  in-flight calls and its admission check is not atomic, so concurrent agents
  race past the limit. This is Meadows' leverage point #12: the budget is just
  a parameter, and the balancing feedback loop that should enforce it is weak
  and delayed. It exists to be the failing baseline.

- ``AtomicLedger``  -- reserve / execute / reconcile. Admission decisions are
  serialized under a lock and account for *committed* (in-flight) tokens, so
  the balancing loop acts before the money leaves the wallet (leverage points
  #9, shorten the delay, and #8, strengthen the balancing loop). A
  completion-reserve buffer (leverage point #11) keeps the mission finishable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Reservation:
    """A slice of budget committed to one in-flight call."""

    amount: int
    settled: bool = False


@dataclass
class LedgerStats:
    admitted: int = 0
    denied: int = 0
    timeline: list[tuple[float, int, int]] = field(default_factory=list)
    """(timestamp, spent, committed) samples, appended on every state change."""


class NaiveLedger:
    """Check-then-act admission with no in-flight accounting (the baseline)."""

    def __init__(self, budget: int, clock=None) -> None:
        self.budget = budget
        self.spent = 0
        self.stats = LedgerStats()
        self._clock = clock or time.monotonic

    @property
    def overshoot(self) -> int:
        return max(0, self.spent - self.budget)

    async def try_reserve(self, estimate: int) -> Reservation | None:
        # The check reads `spent` only -- the billing-dashboard pattern: keep
        # calling until the dashboard shows the limit reached. Calls already
        # admitted but not yet settled are invisible, and nothing prevents
        # interleaving between this check and the eventual settle.
        if self.spent >= self.budget:
            self.stats.denied += 1
            return None
        await asyncio.sleep(0)  # yield point: where concurrent checks interleave
        self.stats.admitted += 1
        return Reservation(amount=estimate)

    async def settle(self, reservation: Reservation, actual: int) -> None:
        reservation.settled = True
        self.spent += actual
        self._sample()

    def _sample(self) -> None:
        self.stats.timeline.append((self._clock(), self.spent, 0))


class AtomicLedger:
    """Reserve/execute/reconcile ledger with atomic admission.

    available = budget * (1 - reserve_fraction) - spent - committed

    ``try_reserve`` and ``settle`` are the only writers and both run under one
    lock, so admission decisions are never simultaneous even when the LLM
    calls themselves run in parallel. The ``reserve_fraction`` slice is only
    admissible once ``begin_finalization()`` is called, guaranteeing there is
    always budget left to land the mission.
    """

    def __init__(
        self,
        budget: int,
        reserve_fraction: float = 0.10,
        clock=None,
    ) -> None:
        self.budget = budget
        self.reserve_fraction = reserve_fraction
        self.spent = 0
        self.committed = 0
        self.finalizing = False
        self.stats = LedgerStats()
        self._lock = asyncio.Lock()
        self._clock = clock or time.monotonic

    @property
    def overshoot(self) -> int:
        return max(0, self.spent - self.budget)

    @property
    def usable_budget(self) -> int:
        if self.finalizing:
            return self.budget
        return int(self.budget * (1.0 - self.reserve_fraction))

    @property
    def available(self) -> int:
        return self.usable_budget - self.spent - self.committed

    def begin_finalization(self) -> None:
        """Release the completion reserve for landing the mission."""
        self.finalizing = True

    async def try_reserve(self, estimate: int) -> Reservation | None:
        async with self._lock:
            if estimate > self.available:
                self.stats.denied += 1
                return None
            self.committed += estimate
            self.stats.admitted += 1
            self._sample()
            return Reservation(amount=estimate)

    async def settle(self, reservation: Reservation, actual: int) -> None:
        async with self._lock:
            if not reservation.settled:
                self.committed -= reservation.amount
                reservation.settled = True
            self.spent += actual
            self._sample()

    async def cancel(self, reservation: Reservation) -> None:
        """Release a reservation whose call failed before consuming tokens."""
        await self.settle(reservation, actual=0)

    def _sample(self) -> None:
        self.stats.timeline.append((self._clock(), self.spent, self.committed))
