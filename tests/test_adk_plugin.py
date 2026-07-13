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
from google.adk.models.llm_response import LlmResponse
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


def _request(text: str = "x" * 400, with_tools: bool = False) -> LlmRequest:
    kwargs = {}
    if with_tools:
        # Only override the framework's default config when the test needs
        # tool declarations; config=None breaks append_instructions.
        kwargs["config"] = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name="investigate")
            ])]
        )
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=text)])],
        **kwargs,
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
        request = _request(with_tools=True)
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        # Admitted (None = proceed to the model), not short-circuited.
        assert result is None
        assert plugin.landings == 1
        assert plugin.ledger.finalizing
        # The landed call cannot call tools: the affordance is removed, so
        # the only possible output is text -- the mission's deliverable.
        # (Observed live: a landed model spent its allowance on one more
        # tool call and was decapitated by the post-tool denial.)
        assert request.config.tools is None
        # Output capped below the reservation by a safety margin, with the
        # appended landing instruction itself billed (measure the input on
        # a pristine copy: `request` has the instruction added by now).
        input_estimate = (
            estimate_input_tokens(_request()) + len(LANDING_TEXT) // 4 + 8
        )
        margin = input_estimate // 10 + 128
        expected = plugin.ledger.budget - input_estimate - margin
        assert request.config.max_output_tokens == expected
        # The landing order rides as the last user message (max attention),
        # not as a system instruction.
        last = request.contents[-1]
        assert last.role == "user"
        assert "FINAL ALLOWANCE" in last.parts[0].text
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
        reservation, _, _ = plugin._pending["inv-1"].pop()
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


def test_landing_survives_a_large_context_regression():
    """Replay of the 2026-07-10 live run that regressed to decapitation.

    Budget 12000, ~4300 already spent, and a ~24k-char context at the denied
    call: an input-proportional margin sized at 25% pushed the allowance
    below LANDING_FLOOR and the terminal denial fired -- the margin itself
    decapitated the mission. The landing must survive contexts that are
    large relative to the remaining headroom.
    """

    async def scenario():
        plugin = _plugin(budget=12_000)
        r = await plugin.ledger.try_reserve(4294)
        await plugin.ledger.settle(r, 4294)
        request = _request(text="x" * 24_000)
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None  # admitted: landed, not denied
        assert plugin.landings == 1
        assert request.config.max_output_tokens >= LANDING_FLOOR
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_no_mission_dies_without_a_landing():
    """The runway invariant, replaying the second live regression.

    There, denial arrived with 3.1k of headroom against a 6.5k-token context:
    the landing's own input no longer fit and the terminal denial fired. The
    governor must land *before* the window closes: simulate a mission whose
    context grows every turn (as contexts do) and assert the trajectory ends
    in a landing -- never in a denial with no landing behind it.
    """

    async def scenario():
        plugin = _plugin(budget=12_000, prior=1024)
        chars = 2_000
        for turn in range(20):
            request = _request(text="x" * chars)
            result = await plugin.before_model_callback(
                callback_context=_Ctx(), llm_request=request
            )
            assert result is None, (
                f"terminal denial at turn {turn} with landings="
                f"{plugin.landings} -- the mission was decapitated"
            )
            if plugin.landings:
                assert request.config.max_output_tokens >= LANDING_FLOOR
                return
            # Settle at the full reservation (worst case) and grow the
            # context by roughly what a tool-using turn appends.
            reservation, _, _ = plugin._pending["inv-1"].pop()
            await plugin.ledger.settle(reservation, reservation.amount)
            chars += 6_000
        raise AssertionError("mission never landed and never ended")

    asyncio.run(scenario())


def test_wasted_landing_gets_a_shorter_second_runway():
    """Re-entrant landing: a model that spends its final allowance imitating
    tool calls (observed live on Nemotron -- stripping declarations does not
    strip the pattern in its own history) gets another, shorter landing
    instead of a terminal denial, until the floor ends it."""

    async def scenario():
        plugin = _plugin()  # budget 2000, prior 5000: first call must land
        request = _request()
        assert await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        ) is None
        first_cap = request.config.max_output_tokens
        # The landed call wastes its allowance on a cheap tool call.
        reservation, _, _ = plugin._pending["inv-1"].pop()
        await plugin.ledger.settle(reservation, 300)

        request2 = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request2
        )
        assert result is None, "second landing denied"
        assert plugin.landings == 2
        assert request2.config.max_output_tokens < first_cap
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_landing_reply_cannot_take_off_again():
    """A landing is a landing: function calls in the landed reply are
    stripped at settle time, so the invocation finalizes with the model's
    own words instead of executing one more tool round against an
    exhausted ledger (observed live on Nemotron: declaration-stripping
    does not stop a model whose history teaches the pattern)."""

    async def scenario():
        plugin = _plugin()
        await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        reply = LlmResponse(content=types.Content(role="model", parts=[
            types.Part(text="Preliminary notes on the findings."),
            types.Part(function_call=types.FunctionCall(
                name="investigate", args={"aspect": "more"}
            )),
        ]))
        replaced = await plugin.after_model_callback(
            callback_context=_Ctx(), llm_response=reply
        )
        assert replaced is not None
        texts = [p.text for p in replaced.content.parts if p.text]
        assert texts == ["Preliminary notes on the findings."]
        assert not any(p.function_call for p in replaced.content.parts)

        # And a call-only reply gets the governor's minimal epitaph rather
        # than an empty final message.
        plugin2 = _plugin()
        await plugin2.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        reply2 = LlmResponse(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(
                name="investigate", args={}
            )),
        ]))
        replaced2 = await plugin2.after_model_callback(
            callback_context=_Ctx(), llm_response=reply2
        )
        assert "landed at budget exhaustion" in replaced2.content.parts[-1].text

    asyncio.run(scenario())


def test_input_calibrator_learns_the_tokenizer():
    from governor.estimator import InputCalibrator

    cal = InputCalibrator(min_samples=2)
    assert cal.factor("a") == 1.0  # trust the heuristic until evidence
    # Takeoff samples (tiny estimate, overhead-dominated) must not train:
    # a 186-vs-585 first turn taught 3.14x and tripled every later estimate.
    cal.update("a", estimated=186, actual=585)
    cal.update("a", estimated=7000, actual=4500)  # Llama-family: overcounted
    cal.update("a", estimated=7100, actual=4507)
    assert 0.6 < cal.factor("a") < 0.7
    cal.update("b", estimated=3000, actual=3600)  # Spanish: undercounted
    cal.update("b", estimated=3000, actual=3500)
    assert cal.factor("b") > 1.1
    assert cal.factor("c") == 1.0  # keys are independent


def test_event_sink_records_the_decision_trail():
    """The governor's decisions leave a record: landing, settle, denial."""

    async def scenario():
        events = []
        plugin = BudgetGovernorPlugin(
            budget=2000,
            estimator=OutputEstimator(prior=5000),
            event_sink=events.append,
        )
        await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        await plugin.after_model_callback(
            callback_context=_Ctx(),
            llm_response=LlmResponse(),  # no usage: charged at the estimate
        )

        kinds = [e["event"] for e in events]
        assert kinds == ["landing", "settled"]
        assert events[0]["allowance"] > 0 and events[0]["cap"] > 0
        assert events[1]["was_landing"] and events[1]["actual"] > 0

    asyncio.run(scenario())


def test_landing_is_retryable_after_a_provider_error():
    """A gust on short final: the landed call is the mission's longest
    generation, hence the likeliest to be interrupted by a transient
    provider error. Observed live: the resumed invocation met
    finalizing=True, which walled off both ordinary admission and the
    landing gate -- terminal denial after a successful landing. A cancelled
    landing must reopen the approach."""

    async def scenario():
        plugin = _plugin()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        assert result is None and plugin.landings == 1

        # The provider drops the landed call; ADK cancels via the plugin.
        await plugin.on_model_error_callback(
            callback_context=_Ctx(),
            llm_request=_request(),
            error=RuntimeError("503 UNAVAILABLE"),
        )

        # The retry invocation must land again, not meet a terminal denial.
        request = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None, "retry after a dropped landing was denied"
        assert plugin.landings == 2
        assert request.config.max_output_tokens >= LANDING_FLOOR
        assert plugin.ledger.overshoot == 0

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
