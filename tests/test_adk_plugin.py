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
    output_cap,
)
from governor.estimator import OutputEstimator


class _Ctx:
    """Duck-typed CallbackContext: the plugin reads two attributes."""

    agent_name = "worker"

    def __init__(self, invocation_id: str = "inv-1") -> None:
        self.invocation_id = invocation_id


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


def test_landing_budgets_the_thinking_toll():
    """Replay of the 2026-07-15 staircase failures (gemma-4-26b-a4b-it).

    Landings failed 5 of 6 while ordinary runs passed 9 of 9: the landed
    call burned its entire allowance on thinking tokens (case #1: allowance
    2041, thoughts 2038, output 0) because on Gemini-family models thoughts
    bill against max_output_tokens before any response text exists. The
    landing that prevents decapitation was itself the decapitation. Two
    guarantees pin the fix: a reasoning agent lands EARLIER (the runway
    carries the toll), and no landing is ever admitted whose allowance the
    known toll consumes down past the floor.
    """

    async def scenario():
        # Same budget, same request; the only difference is a settle history
        # of ~2k thinking tokens. The reasoning twin must land NOW while the
        # non-reasoning twin still cruises on ordinary admission -- the toll
        # widened the runway.
        events = []
        reasoning = BudgetGovernorPlugin(
            budget=4000, estimator=OutputEstimator(prior=1024),
            event_sink=events.append,
        )
        reasoning.thoughts.update("worker", 2038)
        cruising = BudgetGovernorPlugin(
            budget=4000, estimator=OutputEstimator(prior=1024)
        )
        request = _request()
        assert await cruising.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        ) is None
        assert cruising.landings == 0  # old arithmetic: no landing yet
        assert await reasoning.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        ) is None
        assert reasoning.landings == 1  # toll-aware arithmetic: land now
        # The cap that reaches the model leaves the floor intact AFTER the
        # expected thinking: room to think and still room to speak.
        toll = reasoning.thoughts.predict("worker")
        assert request.config.max_output_tokens - toll >= LANDING_FLOOR
        # The toll is on the record: JSONL reconstructions must show it.
        landing = next(e for e in events if e["event"] == "landing")
        assert landing["thoughts_toll"] == 2038

        # Case #1's exact shape -- budget 6000, 2574 already spent, ~1400
        # input tokens, ~2k learned toll: the allowance that fits (~1.6k)
        # cannot survive the toll, so the landing must NOT be admitted to
        # fail. The terminal denial is the documented, honest outcome.
        doomed = _plugin(budget=6000)
        doomed.thoughts.update("worker", 2038)
        r = await doomed.ledger.try_reserve(2574)
        await doomed.ledger.settle(r, 2574)
        result = await doomed.before_model_callback(
            callback_context=_Ctx(), llm_request=_request(text="x" * 5600)
        )
        assert result is not None
        assert result.content.parts[0].text == DENIAL_TEXT
        assert doomed.landings == 0
        assert doomed.ledger.overshoot == 0

    asyncio.run(scenario())


def test_non_reasoning_models_pay_no_thinking_tax():
    """A prior of zero means zero: an agent whose settle history shows no
    thoughts (or no history at all) gets the pre-toll arithmetic
    byte-for-byte. DiffusionGemma-style models must not land one turn
    earlier, or one token poorer, for a toll they never billed."""

    async def scenario():
        # Zero-thoughts history is trained (not just absent) and still no tax.
        plugin = _plugin()
        for _ in range(5):
            plugin.thoughts.update("worker", 0)
        assert plugin.thoughts.predict("worker") == 0
        request = _request()
        assert await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        ) is None
        # Byte-for-byte the old allowance: same input estimate, same margin.
        input_estimate = (
            estimate_input_tokens(_request()) + len(LANDING_TEXT) // 4 + 8
        )
        margin = input_estimate // 10 + 128
        assert request.config.max_output_tokens == (
            plugin.ledger.budget - input_estimate - margin
        )
        # And no earlier landing: the runway is exactly the old runway.
        cruising = BudgetGovernorPlugin(
            budget=4000, estimator=OutputEstimator(prior=1024)
        )
        for _ in range(5):
            cruising.thoughts.update("worker", 0)
        assert await cruising.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        ) is None
        assert cruising.landings == 0

    asyncio.run(scenario())


def test_thoughts_estimator_learns_at_settle():
    """The toll is learned where the governor already looks: settle-time
    usage metadata. One sample trains (thinking is a property of the model,
    not of the lucky call), and providers that report thoughts as None
    (NVIDIA NIM) train zero, not garbage."""

    async def scenario():
        def settle(plugin, thoughts):
            return plugin.after_model_callback(
                callback_context=_Ctx(),
                llm_response=LlmResponse(
                    usage_metadata=types.GenerateContentResponseUsageMetadata(
                        total_token_count=500,
                        candidates_token_count=100,
                        thoughts_token_count=thoughts,
                    )
                ),
            )

        big = 50_000  # roomy: ordinary admissions, no landing in the way
        plugin = BudgetGovernorPlugin(
            budget=big, estimator=OutputEstimator(prior=1024)
        )
        assert plugin.thoughts.predict("worker") == 0  # prior: no tax
        await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        await settle(plugin, 333)
        assert plugin.thoughts.predict("worker") == 333  # one sample trains

        nim = BudgetGovernorPlugin(
            budget=big, estimator=OutputEstimator(prior=1024)
        )
        await nim.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        await settle(nim, None)
        assert nim.thoughts.predict("worker") == 0  # None is absence, not data

    asyncio.run(scenario())


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


def test_leaked_reservation_reconciled_on_the_retry_invocation():
    """Replay of the 2026-07-14 live leak (email #3, budget 2962).

    Reservation A's model call died during response streaming -- neither
    after_model_callback nor on_model_error_callback fired -- and the retry
    arrived as a NEW invocation. A's 546 committed tokens were held forever:
    reservation B came and was cancelled cleanly via on_model_error, but the
    landing later priced itself against headroom shrunk by the leak (964
    tokens of allowance where ~1500 fit). A new invocation of the same agent
    is proof the old one can no longer settle: A must be cancelled then."""

    async def scenario():
        events = []
        plugin = BudgetGovernorPlugin(
            budget=2962,
            estimator=OutputEstimator(prior=400),
            event_sink=events.append,
        )
        # inv-1: reservation A admitted, then its call dies silently -- no
        # settle, no error callback, nothing pops it.
        assert await plugin.before_model_callback(
            callback_context=_Ctx("inv-1"), llm_request=_request()
        ) is None
        leaked = plugin.ledger.committed
        assert leaked > 0

        # inv-2 (the retry): admitting B must first reconcile A away, so
        # committed reflects ONE live reservation, not two.
        assert await plugin.before_model_callback(
            callback_context=_Ctx("inv-2"), llm_request=_request()
        ) is None
        reconciled = [e for e in events if e["event"] == "reconciled"]
        assert [e["amount"] for e in reconciled] == [leaked]
        assert plugin.ledger.committed == plugin._pending["inv-2"][0][0].amount

        # B's call fails loudly; on_model_error cancels it as before.
        await plugin.on_model_error_callback(
            callback_context=_Ctx("inv-2"),
            llm_request=_request(),
            error=RuntimeError("ServerError"),
        )
        assert plugin.ledger.committed == 0

        # inv-3: a context large enough to trip the runway check. With the
        # leak reconciled the landing prices its allowance against the FULL
        # budget -- the exact expression the unleaked ledger would grant.
        request = _request(text="x" * 4000)
        assert await plugin.before_model_callback(
            callback_context=_Ctx("inv-3"), llm_request=request
        ) is None
        assert plugin.landings == 1
        input_estimate = (
            estimate_input_tokens(_request(text="x" * 4000))
            + len(LANDING_TEXT) // 4 + 8
        )
        expected = (
            plugin.ledger.budget - input_estimate
            - (input_estimate // 10 + 128)
        )
        assert request.config.max_output_tokens == expected
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_after_run_sweeps_what_the_invocation_left_pending():
    """The leak can also happen on the mission's LAST call, where no later
    admission exists to reconcile it: invocation end must sweep whatever is
    still pending (and drop the invocation's _pending slot)."""

    async def scenario():
        plugin = BudgetGovernorPlugin(
            budget=50_000, estimator=OutputEstimator(prior=1024)
        )
        assert await plugin.before_model_callback(
            callback_context=_Ctx("inv-1"), llm_request=_request()
        ) is None
        assert plugin.ledger.committed > 0

        await plugin.after_run_callback(invocation_context=_Ctx("inv-1"))
        assert plugin.ledger.committed == 0
        assert "inv-1" not in plugin._pending
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_ordinary_admission_caps_realized_spend():
    """The wall must be physical, not statistical: an ordinary admission
    reserves input + p90(output), but the p90 is an estimate and the model
    bills what it actually writes (observed live: 11 of 26 runs overshot
    with denied=0 once the estimator warmed up). Every ordinary admission
    therefore carries max_output_tokens = estimate + tail margin."""

    async def scenario():
        prior = 1024
        plugin = BudgetGovernorPlugin(
            budget=50_000, estimator=OutputEstimator(prior=prior)
        )
        request = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None
        assert plugin.landings == 0
        # Far from the wall the headroom clamp is slack: the cap is the
        # reserved output estimate plus the tail margin, nothing else.
        assert request.config.max_output_tokens == output_cap(prior)
        assert output_cap(prior) == prior + prior // 2 + 128
        # The meter still rides along with the cap.
        assert "Live budget state" in str(request.config.system_instruction)

    asyncio.run(scenario())


def test_ordinary_cap_never_exceeds_remaining_headroom():
    """Near the wall the margin yields to the ledger: even a reply that
    fills the whole cap must settle inside the ordinary tranche, leaving
    the appeal tranche and the completion reserve intact."""

    async def scenario():
        plugin = BudgetGovernorPlugin(
            budget=20_000, estimator=OutputEstimator(prior=1024)
        )
        r = await plugin.ledger.try_reserve(15_500)
        await plugin.ledger.settle(r, 15_500)
        headroom = plugin.ledger.available  # ordinary tranche, pre-admission
        input_estimate = estimate_input_tokens(_request())

        request = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None
        assert plugin.landings == 0
        cap = request.config.max_output_tokens
        assert cap < output_cap(1024)  # the clamp bit
        assert input_estimate + cap == headroom  # ... exactly at the wall
        # A maximal reply settles with zero overshoot and the tranche intact.
        reservation, _, _ = plugin._pending["inv-1"].pop()
        await plugin.ledger.settle(reservation, input_estimate + cap)
        assert plugin.ledger.available >= 0
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_appealed_cap_clamps_to_the_priority_tranche():
    """An appealed call may spend the appeal tranche, never the completion
    reserve: its cap clamps against priority headroom, not the (already
    exhausted) ordinary tranche."""

    async def scenario():
        plugin = BudgetGovernorPlugin(
            budget=20_000, estimator=OutputEstimator(prior=1024)
        )
        r = await plugin.ledger.try_reserve(16_500)
        await plugin.ledger.settle(r, 16_500)
        text = ("APPEAL: this call is the mission's critical path" + " x" * 200)[:400]
        input_estimate = estimate_input_tokens(_request(text=text))
        assert input_estimate + 1024 > plugin.ledger.available  # ordinary: denied
        headroom = plugin.ledger.priority_available

        request = _request(text=text)
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None
        assert plugin.appeals.log.granted == 1
        cap = request.config.max_output_tokens
        assert input_estimate + cap == headroom  # reserve stays inviolate
        reservation, _, _ = plugin._pending["inv-1"].pop()
        await plugin.ledger.settle(reservation, input_estimate + cap)
        assert plugin.ledger.overshoot == 0

    asyncio.run(scenario())


def test_landing_cap_is_the_allowance_not_the_ordinary_expression():
    """The landing path already caps at whatever still fits; the ordinary
    cap must not touch it."""

    async def scenario():
        events = []
        plugin = BudgetGovernorPlugin(
            budget=2000,
            estimator=OutputEstimator(prior=5000),
            event_sink=events.append,
        )
        request = _request()
        result = await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=request
        )
        assert result is None
        assert [e["event"] for e in events] == ["landing"]  # no "admitted"
        assert request.config.max_output_tokens == events[0]["allowance"]

    asyncio.run(scenario())


def test_estimator_learns_billed_output_including_thoughts():
    """Thinking bills as output and (on Gemini) the output cap governs
    thoughts + text together: the estimator must learn the billed sum, or
    every thinking call under-reserves by its thoughts (observed live on
    Gemma: text ~21, thoughts ~150+, both billed -- the exact size of the
    recorded overshoots)."""

    async def scenario():
        plugin = BudgetGovernorPlugin(
            budget=500_000,
            estimator=OutputEstimator(prior=1024, min_samples=1),
        )
        await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=600,
            candidates_token_count=21,
            thoughts_token_count=150,
            total_token_count=771,
        )
        await plugin.after_model_callback(
            callback_context=_Ctx(),
            llm_response=LlmResponse(usage_metadata=usage),
        )
        assert plugin.estimator.predict("worker") == 171

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


def test_estimator_does_not_learn_the_landing():
    """The self-reinforcing cliff, replayed: a landing's bill is
    ceiling-shaped (capped by the allowance, inflated by the deliberation
    the FINAL ALLOWANCE order provokes). Feeding it to the output estimator
    teaches landing-sized costs, fattens the runway, and forces the next
    landing -- observed live as 22/22 turn-1 landings at a budget the
    staircase had validated. Cruise calls train; landings do not. The
    thoughts estimator is the exception: the toll is best measured where
    it bites."""

    async def scenario():
        plugin = _plugin()  # budget 2000, prior 5000: first call must land
        await plugin.before_model_callback(
            callback_context=_Ctx(), llm_request=_request()
        )
        assert plugin.landings == 1
        before = plugin.estimator.predict("worker")
        usage = types.GenerateContentResponseUsageMetadata(
            total_token_count=1866,
            candidates_token_count=329,
            thoughts_token_count=1214,
            prompt_token_count=600,
        )
        await plugin.after_model_callback(
            callback_context=_Ctx(),
            llm_response=LlmResponse(usage_metadata=usage),
        )
        # The landed bill (329 + 1214) must NOT train the output estimator --
        # checked on the HISTORY, not the prediction: below min_samples the
        # predictor answers the prior either way, and a poisoned sample would
        # hide behind it until it is 5 samples deep and self-reinforcing.
        assert plugin.estimator.predict("worker") == before
        assert len(plugin.estimator._history.get("worker", [])) == 0
        # ...but the thoughts estimator must learn the toll it just measured.
        assert plugin.thoughts.predict("worker") == 1214

    asyncio.run(scenario())
