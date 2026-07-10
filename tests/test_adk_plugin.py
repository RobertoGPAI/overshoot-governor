"""Plugin-level tests: the landing protocol observed by the observer game.

The game's wall cells showed that inside a single ADK invocation a
short-circuited denial IS the agent's final message: the mission dies with
the governor's own denial text as its "report" and the appeal right is
unreachable. These tests pin the fix -- deny only when not even a capped
landing fits.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from governor.adk_plugin import (
    DENIAL_TEXT,
    LANDING_FLOOR,
    LANDING_TEXT,
    BudgetGovernorPlugin,
    estimate_input_tokens,
)
from governor.estimator import OutputEstimator


class _Ctx:
    """Duck-typed CallbackContext: the plugin reads two attributes."""

    agent_name = "worker"
    invocation_id = "inv-1"


def _request(text: str = "x" * 400) -> LlmRequest:
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=text)])]
    )


def _plugin(budget: int = 2000, prior: int = 5000) -> BudgetGovernorPlugin:
    # A prior far above the budget guarantees the ordinary reservation is
    # denied on the very first call, forcing the landing decision immediately.
    return BudgetGovernorPlugin(
        budget=budget, estimator=OutputEstimator(prior=prior)
    )


def test_landing_admits_capped_final_call_instead_of_denying():
    async def scenario():
        plugin = _plugin()
        request = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        # Admitted (None = proceed to the model), not short-circuited.
        assert result is None
        assert plugin.landings == 1
        assert plugin.ledger.finalizing
        # Output capped below the reservation by a safety margin, with the
        # appended landing instruction itself billed (measure the input on
        # a pristine copy: `request` has the instruction added by now).
        input_estimate = (
            estimate_input_tokens(_request()) + len(LANDING_TEXT) // 4 + 8
        )
        margin = input_estimate // 4 + 256
        expected = plugin.ledger.budget - input_estimate - margin
        assert request.config.max_output_tokens == expected
        assert "FINAL ALLOWANCE" in str(request.config.system_instruction)
        # The reservation fills the ledger: nothing left to overshoot with.
        assert plugin.ledger.available == 0

    asyncio.run(scenario())


def test_denial_is_terminal_after_the_landing():
    async def scenario():
        plugin = _plugin()
        await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        # The landing call settles at its full reservation; budget exhausted.
        reservation, _ = plugin._pending["inv-1"].pop()
        await plugin.ledger.settle(reservation, reservation.amount)

        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        assert result is not None  # short-circuited this time
        assert result.content.parts[0].text == DENIAL_TEXT
        assert plugin.landings == 1  # one landing per ledger, ever
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_landing_denied_when_not_even_the_floor_fits():
    async def scenario():
        # Input alone nearly fills the budget: allowance < LANDING_FLOOR.
        plugin = _plugin(budget=300)
        request = _request(text="x" * 800)
        assert (
            plugin.ledger.budget - estimate_input_tokens(request)
            < LANDING_FLOOR
        )
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is not None
        assert result.content.parts[0].text == DENIAL_TEXT
        assert plugin.landings == 0

    asyncio.run(scenario())


def test_ordinary_admission_still_gets_the_meter_not_the_landing():
    async def scenario():
        plugin = BudgetGovernorPlugin(
            budget=50_000, estimator=OutputEstimator(prior=1024)
        )
        request = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None
        assert plugin.landings == 0
        assert not plugin.ledger.finalizing
        instructions = str(request.config.system_instruction)
        assert "Live budget state" in instructions
        assert "FINAL ALLOWANCE" not in instructions

    asyncio.run(scenario())
