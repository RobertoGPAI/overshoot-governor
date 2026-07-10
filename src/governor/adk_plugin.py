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

- Landing (#8 bends the trajectory, it does not sever it): inside a single
  ADK invocation a short-circuited denial IS the agent's final message -- the
  agent never runs again to "wrap up" or appeal, so a bare denial decapitates
  the mission and strands the whole budget's work. And landing has a closing
  window: it costs one more read of the context, which grows every turn, so
  waiting for a denial can strand the mission past the point where not even
  the landing's input fits. The governor therefore keeps a *dynamic* runway
  -- before admitting an ordinary call it checks that enough would remain to
  land afterwards, and when the check fails it lands NOW: releases the
  completion reserve (``begin_finalization`` -- this is what that tranche is
  *for*), admits the call with ``max_output_tokens`` capped to what still
  fits, and instructs the model to deliver its final result within that
  allowance. Only when even the landing does not fit does the terminal
  denial fire.
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

# Below this many output tokens a landing is not worth admitting: the model
# could not say anything useful and the reserve is better left unspent.
LANDING_FLOOR = 256

LANDING_TEXT = (
    "[BUDGET GOVERNOR] FINAL ALLOWANCE. Your next step would have exceeded "
    "the remaining budget, so the completion reserve has been released to "
    "land the mission: this call is admitted with room for about {allowance} "
    "output tokens, and it is the last one. Deliver the mission's final "
    "result NOW, complete and self-contained, within that space, using only "
    "what you already know. Do not call tools. Do not spawn agents."
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
        self.landings = 0
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

    async def _try_landing(
        self, key: str, llm_request: LlmRequest
    ) -> tuple[Reservation | None, int]:
        """Admit one final, output-capped call against the completion reserve.

        Returns ``(reservation, allowance)``; a ``None`` reservation means not
        even the landing fits and the terminal denial should fire. Called at
        most once per ledger: ``begin_finalization`` is idempotent and the
        caller gates on ``ledger.finalizing``.
        """
        self.ledger.begin_finalization()
        # The landing instruction is itself appended to the request and
        # billed with it -- charge for it, or the allowance overshoots by
        # exactly the text that grants it.
        input_estimate = (
            estimate_input_tokens(llm_request) + len(LANDING_TEXT) // 4 + 8
        )
        headroom = self.ledger.available
        # The landing fills the ledger to the brim, so it has none of the
        # slack that quietly absorbs estimation error on ordinary calls: the
        # chars//4 heuristic can undercount, and reasoning models bill
        # thinking tokens the output cap does not govern. Cap the output
        # below the reservation by a margin that eats both -- zero overshoot
        # is the one number this project promises. Sizing is empirical: the
        # observed heuristic error on a live landing was +0.4% of the input
        # estimate, so 10% + 128 is ~20x that error while still leaving a
        # usable allowance when the context is large relative to headroom
        # (an oversized margin IS a decapitation, via the floor check).
        margin = input_estimate // 10 + 128
        allowance = headroom - input_estimate - margin
        if allowance < LANDING_FLOOR:
            return None, 0
        reservation = await self.ledger.try_reserve(headroom)
        return reservation, allowance if reservation else 0

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = callback_context.agent_name
        input_estimate = estimate_input_tokens(llm_request)
        output_estimate = self.estimator.predict(key)
        estimate = input_estimate + output_estimate

        reservation = None
        allowance = 0
        if not self.ledger.finalizing:
            # Runway check: landing costs one more read of the context, and
            # the context grows every turn -- waiting for a denial can strand
            # the mission past the point where not even the landing's input
            # fits (observed live: 3.1k left against a 6.5k context). Keep
            # enough fuel to reach the runway: if admitting this call would
            # leave less than the NEXT landing costs (today's context plus
            # this call's output, margin, floor), land now instead.
            headroom = self.ledger.budget - self.ledger.spent - self.ledger.committed
            next_input = input_estimate + output_estimate
            runway = next_input + next_input // 10 + 128 + LANDING_FLOOR
            if headroom - estimate < runway:
                reservation, allowance = await self._try_landing(key, llm_request)

        if reservation is None:
            reservation = await self.ledger.try_reserve(estimate)
        if reservation is None:
            # Right of appeal (voice): a justified retry may enter the appeal
            # tranche -- never the completion reserve. Rationed and logged.
            justification = self._extract_appeal(llm_request)
            if justification:
                reservation = await self.appeals.appeal(key, estimate, justification)
        if reservation is None and not self.ledger.finalizing:
            # Landing protocol: within one invocation a denial is terminal
            # (the short-circuit text becomes the agent's final message), so
            # before denying, release the completion reserve and admit one
            # last call capped to whatever output still fits.
            reservation, allowance = await self._try_landing(key, llm_request)
        if reservation is None:
            # Short-circuit: the model is never called. The refusal text tells
            # the agent to land with what it has, or appeal once with cause.
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=DENIAL_TEXT)]
                )
            )

        self._pending[callback_context.invocation_id].append((reservation, key))

        if allowance:
            self.landings += 1
            if llm_request.config is None:
                llm_request.config = types.GenerateContentConfig()
            # Obedience is not a plan: a landed model may spend its final
            # allowance calling another tool (observed live), and the call
            # after the tool round-trip meets an exhausted ledger -- terminal
            # denial, decapitation AFTER the landing. Remove the affordance
            # instead of requesting the behavior: with no tool declarations
            # the only possible output is text, and text ends the invocation
            # as the mission's actual deliverable.
            llm_request.config.tools = None
            llm_request.config.tool_config = None
            cap = llm_request.config.max_output_tokens
            llm_request.config.max_output_tokens = (
                min(cap, allowance) if cap else allowance
            )
            llm_request.append_instructions(
                [LANDING_TEXT.format(allowance=allowance)]
            )
        elif self.visibility:
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
            f"landings={self.landings} "
            f"appeals_granted={self.appeals.log.granted} "
            f"appeals_refused={self.appeals.log.refused}"
        )
