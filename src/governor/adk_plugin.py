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
  the plugin appends the live budget state to the system instruction of every
  outgoing request, so the agent can economize *before* hitting the wall.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .estimator import OutputEstimator
from .ledger import AtomicLedger, Reservation

DENIAL_TEXT = (
    "[BUDGET GOVERNOR] This model call was not admitted: the projected token "
    "cost exceeds the remaining budget. Wrap up with the information you "
    "already have. Do not start new tool calls or spawn new agents."
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
        visibility: bool = True,
        estimator: OutputEstimator | None = None,
        name: str = "budget_governor",
    ) -> None:
        super().__init__(name=name)
        self.ledger = AtomicLedger(budget=budget, reserve_fraction=reserve_fraction)
        self.estimator = estimator or OutputEstimator()
        self.visibility = visibility
        self._pending: dict[str, list[tuple[Reservation, str]]] = defaultdict(list)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = callback_context.agent_name
        estimate = estimate_input_tokens(llm_request) + self.estimator.predict(key)

        reservation = await self.ledger.try_reserve(estimate)
        if reservation is None:
            # Short-circuit: the model is never called. The refusal text tells
            # the agent to land with what it has instead of retrying.
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=DENIAL_TEXT)]
                )
            )

        self._pending[callback_context.invocation_id].append((reservation, key))

        if self.visibility:
            llm_request.append_instructions([
                "[BUDGET GOVERNOR] Live budget state: "
                f"{self.ledger.available} tokens available, "
                f"{self.ledger.committed} committed to in-flight calls, "
                f"{self.ledger.spent} already spent of {self.ledger.budget}. "
                "Be economical: prefer short answers, avoid speculative tool "
                "calls, and do not spawn subagents unless strictly necessary."
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
            f"admitted={led.stats.admitted} denied={led.stats.denied}"
        )
