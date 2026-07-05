"""The right of appeal: voice for governed agents.

A hard denial turns the budget into a wall; a wall can strand the mission at
99%. The AppealsDesk turns the denial into a contestable administrative act
(Hirschman's *voice*): a denied agent may appeal by stating why the call is
critical to the overall mission, and a granted appeal admits the call as a
*priority* reservation -- allowed into the appeal tranche of the ledger, never
into the completion reserve. Enforcement stays inviolable while the case is
heard: appeals reallocate protected headroom, they do not raise the ceiling.

Legitimacy constraints, encoded:
  - an appeal without a justification is not an appeal (no silent overrides);
  - grants per agent are capped (appeal is a remedy, not a second budget);
  - every appeal -- granted or refused -- is logged with its justification
    (accountability; the STRIDE repudiation mitigation extends here).

The desk's policy is deliberately simple (non-empty justification + per-agent
cap). The `judge` hook accepts a richer arbiter -- e.g. a coordinator LLM
ruling on whether the justification actually serves the mission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .ledger import AtomicLedger, Reservation

Judge = Callable[[str, int, str], Awaitable[bool]]


@dataclass
class AppealRecord:
    agent: str
    estimate: int
    justification: str
    granted: bool


@dataclass
class AppealsLog:
    records: list[AppealRecord] = field(default_factory=list)

    @property
    def granted(self) -> int:
        return sum(1 for r in self.records if r.granted)

    @property
    def refused(self) -> int:
        return sum(1 for r in self.records if not r.granted)


class AppealsDesk:
    """Hears appeals against admission denials and grants priority reservations."""

    def __init__(
        self,
        ledger: AtomicLedger,
        max_grants_per_agent: int = 2,
        judge: Judge | None = None,
    ) -> None:
        self.ledger = ledger
        self.max_grants_per_agent = max_grants_per_agent
        self.judge = judge
        self.log = AppealsLog()
        self._grants: dict[str, int] = defaultdict(int)

    async def appeal(
        self, agent: str, estimate: int, justification: str
    ) -> Reservation | None:
        justification = (justification or "").strip()
        if not justification or self._grants[agent] >= self.max_grants_per_agent:
            self.log.records.append(AppealRecord(agent, estimate, justification, False))
            return None
        if self.judge is not None and not await self.judge(agent, estimate, justification):
            self.log.records.append(AppealRecord(agent, estimate, justification, False))
            return None
        reservation = await self.ledger.try_reserve(estimate, priority=True)
        granted = reservation is not None
        if granted:
            self._grants[agent] += 1
        self.log.records.append(AppealRecord(agent, estimate, justification, granted))
        return reservation
