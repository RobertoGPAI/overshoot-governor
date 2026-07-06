"""ADK 2.x Runner plugin: the budget governor as a global balancing loop.

Registered once on the Runner, the plugin's callbacks apply to every LLM call
of every agent and subagent it manages -- which is exactly the semantics a
budget governor needs (a per-agent callback could be bypassed by a spawned
subagent; a Runner plugin cannot).

Two intervention levels, deliberately at different Meadows leverage points:

- Hard enforcement (#8, strengthen the balancing loop): before_model_callback
  atomically reserves ``input_estimate + p90(output)`` against the shared
  ledger and short-circuits the call with a refusal LlmResponse when the
  reservation is denied. after_model_callback reconciles the reservation with
  the actual usage_metadata and feeds the estimator.

- Information flow (#6, the meter in the hallway): when ``visibility`` is on,
  the plugin appends the live budget state -- and the overall ``mission`` --
  to the system instruction of every outgoing request, so the agent can
  economize *before* hitting the wall, and can weigh restraint against the
  goal its restraint must serve (#3).

- Voice (#5 rules in service of #3 goals): a denial is a contestable act, not
  a wall. The denied agent may reply 'APPEAL: <reason tied to the mission>'
  and retry once; the AppealsDesk may admit it into the protected appeal
  tranche. Rationed, logged, and never at the expense of the completion
  reserve.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .appeals import AppealsDesk
from .estimator import OutputEstimator
from .judge import MissionJudge
from .ledger import AtomicLedger, Reservation

APPEAL_MARKER = "APPEAL:"

DENIAL_TEXT = (
    "[BUDGET GOVERNOR] This model call was not admitted: the projected token "
    "cost exceeds the remaining budget. Wrap up with the information you "
    "already have. Do not start new tool calls or spawn new agents. "
    "Right of appeal: if THIS call is on the critical path to the overall "
    f"mission, reply with a single line '{APPEAL_MARKER} <one-line reason tied "
    "to the mission>' and retry once. Appeals are logged and rationed."
)


def estimate_input_tokens(llm_request: LlmRequest) -> int:
    """Deterministic-side estimate, computed offline (~4 chars/token).

    For exact pre-call counts swap this for the Gemini ``count_tokens``
    endpoint; the heuristic keeps the governor dependency-free and errs on
    the conservative side with a fixed per-part overhead.
    """
    chars = 0
    parts_seen = 0
    for content in llm_request.contents or []:
        for part in content.parts or []:
            parts_seen += 1
            if part.text:
                chars += len(part.text)
            elif part.function_call is not None:
                chars += len(str(part.function_call.args or ""))
            elif part.function_response is not None:
                chars += len(str(part.function_response.response or ""))
    config = llm_request.config
    if config is not None and config.system_instruction:
        chars += len(str(config.system_instruction))
    return chars // 4 + parts_seen * 8


class BudgetGovernorPlugin(BasePlugin):
    """Global admission control + budget visibility for an ADK Runner."""

    def __init__(
        self,
        budget: int,
        reserve_fraction: float = 0.10,
        appeal_fraction: float = 0.05,
        visibility: bool = True,
        mission: str | None = None,
        estimator: OutputEstimator | None = None,
        arbiter: bool = False,
        name: str = "budget_governor",
    ) -> None:
        super().__init__(name=name)
        self.ledger = AtomicLedger(
            budget=budget,
            reserve_fraction=reserve_fraction,
            appeal_fraction=appeal_fraction,
        )
        # With arbiter=True (and a mission), appeals are heard by a separate
        # judge agent -- deliberately NOT the coordinator (nemo iudex in causa
        # sua) -- whose hearings are themselves budgeted on this ledger.
        self.judge = (
            MissionJudge(mission=mission, ledger=self.ledger)
            if arbiter and mission
            else None
        )
        self.appeals = AppealsDesk(
            self.ledger, judge=self.judge.rule if self.judge else None
        )
        self.estimator = estimator or OutputEstimator()
        self.visibility = visibility
        self.mission = mission
        self._pending: dict[str, list[tuple[Reservation, str]]] = defaultdict(list)

    @staticmethod
    def _extract_appeal(llm_request: LlmRequest) -> str | None:
        """A line starting with 'APPEAL: <reason>' in the latest turn is a filed appeal.

        Line-anchored on purpose: the governor's own DENIAL_TEXT mentions the
        marker mid-sentence when explaining the right of appeal, and quoting
        the rules must not count as invoking them.
        """
        for content in reversed((llm_request.contents or [])[-2:]):
            for part in content.parts or []:
                if not part.text:
                    continue
                for line in part.text.splitlines():
                    line = line.strip()
                    if line.startswith(APPEAL_MARKER):
                        return line[len(APPEAL_MARKER):].strip()
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = callback_context.agent_name
        estimate = estimate_input_tokens(llm_request) + self.estimator.predict(key)

        reservation = await self.ledger.try_reserve(estimate)
        if reservation is None:
            # Right of appeal (voice): a justified retry may enter the appeal
            # tranche -- never the completion reserve. Rationed and logged.
            justification = self._extract_appeal(llm_request)
            if justification:
                reservation = await self.appeals.appeal(key, estimate, justification)
        if reservation is None:
            # Short-circuit: the model is never called. The refusal text tells
            # the agent to land with what it has, or appeal once with cause.
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=DENIAL_TEXT)]
                )
            )

        self._pending[callback_context.invocation_id].append((reservation, key))

        if self.visibility:
            mission_line = (
                f"Overall mission (the goal your restraint must serve): {self.mission}. "
                if self.mission
                else ""
            )
            llm_request.append_instructions([
                "[BUDGET GOVERNOR] " + mission_line + "Live budget state: "
                f"{self.ledger.available} tokens available, "
                f"{self.ledger.committed} committed to in-flight calls, "
                f"{self.ledger.spent} already spent of {self.ledger.budget}. "
                "Be economical: prefer short answers, avoid speculative tool "
                "calls, and do not spawn subagents unless strictly necessary. "
                "Forgo any action that does not serve the mission; if a "
                "critical call is denied, you may appeal with "
                f"'{APPEAL_MARKER} <reason>'."
            ])
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        stack = self._pending.get(callback_context.invocation_id)
        if not stack:
            return None
        reservation, key = stack.pop()

        usage = llm_response.usage_metadata
        if usage is not None and usage.total_token_count:
            actual = usage.total_token_count
            output = usage.candidates_token_count or 0
            self.estimator.update(key, output)
        else:
            actual = reservation.amount  # no metadata: charge the estimate
        await self.ledger.settle(reservation, actual)
        return None

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        stack = self._pending.get(callback_context.invocation_id)
        if stack:
            reservation, _ = stack.pop()
            await self.ledger.cancel(reservation)
        return None

    def report(self) -> str:
        led = self.ledger
        return (
            f"budget={led.budget} spent={led.spent} committed={led.committed} "
            f"available={led.available} overshoot={led.overshoot} "
            f"admitted={led.stats.admitted} denied={led.stats.denied} "
            f"appeals_granted={self.appeals.log.granted} "
            f"appeals_refused={self.appeals.log.refused}"
        )
