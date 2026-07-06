"""MissionJudge: an arbiter agent for appeals -- judgment as a budgeted call.

Two design decisions carry the weight here:

1. **The judge is not the coordinator.** The coordinator allocates quotas; if
   it also ruled on appeals against the scarcity it created, it would be judge
   in its own cause (nemo iudex in causa sua). The arbiter is a separate,
   minimal model call whose only mandate is: does this justification actually
   serve the mission?

2. **Justice is not free.** Every hearing reserves and settles tokens on the
   SAME ledger it arbitrates (priority tranche, like the appeals it hears).
   If the system cannot afford the hearing, the appeal is refused outright --
   which also caps the blast radius of appeal spam: an attacker flooding the
   channel exhausts the hearing budget, not the mission's.

Offline behavior: with no model caller available (no API key), the judge
abstains and defers to the desk's mechanical policy (justification present +
ration). That keeps tests and simulations deterministic; the LLM ruling is an
upgrade, not a dependency.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable

from .ledger import AtomicLedger

# caller(prompt) -> (verdict_text, total_tokens_used)
Caller = Callable[[str], Awaitable[tuple[str, int]]]

RULING_PROMPT = """You are the arbiter of a token-budget governor for a \
multi-agent system. The overall mission is:

  {mission}

Agent '{agent}' was denied a model call (estimated {estimate} tokens) and \
appeals with this justification:

  "{justification}"

Grant the appeal ONLY if completing this specific call plausibly protects or \
advances the mission more than saving the tokens would. Spending already \
incurred is not a reason (sunk cost). Answer with exactly one word: \
GRANT or REFUSE."""


def genai_caller(model: str | None = None) -> Caller | None:
    """Default caller backed by google.genai; None when no API key is set.

    The hearing is a plain text prompt -- no tools, no system instruction --
    so even models without function-calling support (e.g. Gemma) can serve
    as arbiter. Override via GOVERNOR_JUDGE_MODEL to put justice on a small,
    frugal model while the governed agents run on a capable one.
    """
    # google-genai reads either name; AI Studio tutorials use GEMINI_API_KEY.
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return None
    model = model or os.environ.get("GOVERNOR_JUDGE_MODEL", "gemini-2.5-flash")
    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client()

    async def call(prompt: str) -> tuple[str, int]:
        # A transient 5xx must not decide a case: the hearing is one cheap
        # request, so retry briefly before failing closed as usual.
        for attempt in (1, 2, 3):
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=prompt
                )
                break
            except genai_errors.ServerError:
                if attempt == 3:
                    raise
                await asyncio.sleep(attempt)
        used = 0
        if response.usage_metadata and response.usage_metadata.total_token_count:
            used = response.usage_metadata.total_token_count
        return (response.text or ""), used

    return call


class MissionJudge:
    """Arbiter for AppealsDesk: plug ``.rule`` into the desk's judge hook."""

    def __init__(
        self,
        mission: str,
        ledger: AtomicLedger,
        caller: Caller | None = None,
        hearing_estimate: int = 400,
    ) -> None:
        self.mission = mission
        self.ledger = ledger
        self.caller = caller if caller is not None else genai_caller()
        self.hearing_estimate = hearing_estimate
        self.hearings = 0
        self.hearing_tokens = 0
        self.hearing_failures = 0
        self.last_error: Exception | None = None

    async def rule(self, agent: str, estimate: int, justification: str) -> bool:
        if self.caller is None:
            return True  # abstain: the desk's mechanical policy decides

        # The hearing itself must be admitted -- from the priority tranche,
        # like the appeal it examines. No budget for justice, no appeal.
        reservation = await self.ledger.try_reserve(
            self.hearing_estimate, priority=True
        )
        if reservation is None:
            return False

        try:
            verdict, used = await self.caller(
                RULING_PROMPT.format(
                    mission=self.mission,
                    agent=agent,
                    estimate=estimate,
                    justification=justification,
                )
            )
        except Exception as exc:
            self.hearing_failures += 1
            self.last_error = exc
            await self.ledger.cancel(reservation)
            return False  # a hearing that fails grants nothing

        await self.ledger.settle(reservation, used)
        self.hearings += 1
        self.hearing_tokens += used
        return verdict.strip().upper().startswith("GRANT")
