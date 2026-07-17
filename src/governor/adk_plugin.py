"""ADK 2.x Runner plugin: the budget governor as a global balancing loop.

Registered once on the Runner, the plugin's callbacks apply to every LLM call
of every agent and subagent it manages -- which is exactly the semantics a
budget governor needs (a per-agent callback could be bypassed by a spawned
subagent; a Runner plugin cannot).

Two intervention levels, deliberately at different Meadows leverage points:

- Hard enforcement (#8, strengthen the balancing loop): before_model_callback
  atomically reserves ``input_estimate + p90(output)`` against the shared
  ledger and short-circuits the call with a refusal LlmResponse when the
  reservation is denied. The reservation bounds *expected* spend; the same
  admission also bounds *realized* spend by setting ``max_output_tokens`` to
  the reserved output estimate plus a tail margin, clamped to the tranche
  headroom left after the input -- an output that runs past its estimate
  bills anyway, and a ledger that can only record the excess is a meter, not
  a wall. after_model_callback reconciles the reservation with the actual
  usage_metadata and feeds the estimator.

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
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from .appeals import AppealsDesk
from .estimator import InputCalibrator, OutputEstimator, ThoughtsEstimator
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
    "what you already know. Do not call tools. Do not spawn agents. If you "
    "reason before writing, your reasoning bills against this same "
    "allowance: think briefly or not at all -- write the deliverable "
    "directly."
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


def output_cap(output_estimate: int) -> int:
    """Physical output cap for an ordinary admission.

    The estimator predicts a p90, so by construction ~1 call in 10 runs past
    it -- capping AT the estimate would truncate exactly those calls. The
    margin buys out the tail: half again the estimate covers the p90-to-max
    spread of a well-behaved output distribution, and the 128-token floor
    keeps small estimates (a warmed-up estimator on a terse agent) from
    producing caps so tight that ordinary variance truncates the reply.
    Truncation is still possible -- that is the point of a wall -- but rare,
    and a truncated turn settles normally, so the mission continues.
    """
    return output_estimate + output_estimate // 2 + 128


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
        event_sink=None,
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
        self.calibrator = InputCalibrator()
        # Reasoning models bill thinking tokens inside the output cap
        # (observed live: a landing whose whole allowance went to thoughts,
        # zero response tokens -- the landing became the failure). The toll
        # is learned per agent from settle-time usage, prior zero.
        self.thoughts = ThoughtsEstimator()
        self.visibility = visibility
        self.mission = mission
        self.landings = 0
        # The landed call is the mission's longest generation, hence the one
        # a provider hiccup is most likely to interrupt. If it dies, its
        # reservation is cancelled and the landing must be attemptable again
        # on the resumed invocation -- otherwise finalizing=True walls off
        # both ordinary admission AND the landing gate, and the retry meets
        # a terminal denial (observed live as post-landing decapitations
        # correlated with transient provider errors).
        self._landing_reservation: Reservation | None = None
        # Telemetry: every admission decision, emitted as a dict to an
        # optional sink. A governor whose decisions leave no record is not
        # an institution -- and every debugging session of this plugin so
        # far has consisted of reconstructing exactly these events from
        # aggregate arithmetic.
        self._event_sink = event_sink
        # (reservation, agent key, raw chars-based input estimate) -- the raw
        # estimate is kept so settle time can calibrate it against the
        # provider's true prompt count. Keyed by invocation_id, because a
        # reservation can only be settled by the invocation that made it:
        # anything still pending once that invocation is over is a leak and
        # must be reconciled (see _reconcile_stale / after_run_callback).
        self._pending: dict[str, list[tuple[Reservation, str, int]]] = defaultdict(list)

    def _emit(self, kind: str, **data) -> None:
        if self._event_sink is None:
            return
        led = self.ledger
        self._event_sink({"event": kind, "spent": led.spent,
                          "committed": led.committed, **data})

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
    ) -> tuple[Reservation | None, int, int]:
        """Admit one final, output-capped call against the completion reserve.

        Returns ``(reservation, allowance, thoughts_toll)``; a ``None``
        reservation means not even the landing fits and the terminal denial
        should fire. Called at most once per ledger: ``begin_finalization``
        is idempotent and the caller gates on ``ledger.finalizing``.
        """
        self.ledger.begin_finalization()
        # The landing instruction is itself appended to the request and
        # billed with it -- charge for it, or the allowance overshoots by
        # exactly the text that grants it. Calibrated like every estimate:
        # an overcounting heuristic closes the landing window early.
        input_estimate = self.calibrator.calibrate(
            key, estimate_input_tokens(llm_request) + len(LANDING_TEXT) // 4 + 8
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
        # On reasoning models the allowance is not all speech: thinking
        # tokens bill against the same output cap before any response text
        # exists (observed live: allowance 2041, thoughts 2038, output 0 --
        # the landing became the failure). The floor must survive the toll:
        # a runway you cannot speak on is not a runway, and a landing whose
        # allowance the known toll consumes is better spent as the terminal
        # denial. Non-reasoning agents have a toll of zero and keep the old
        # arithmetic exactly.
        thoughts_toll = self.thoughts.predict(key)
        if allowance - thoughts_toll < LANDING_FLOOR:
            return None, 0, thoughts_toll
        reservation = await self.ledger.try_reserve(headroom)
        return reservation, allowance if reservation else 0, thoughts_toll

    async def _reconcile_stale(self, key: str, current_invocation: str) -> None:
        """Cancel reservations this agent left pending in past invocations.

        A model call can die without firing ANY settlement hook -- observed
        live: the provider error surfaced while the response stream was being
        iterated, outside on_model_error_callback's reach, and the retry
        arrived as a fresh invocation. The orphaned reservation then holds
        its committed tokens until process exit, and the landing pays for it
        (964 tokens of allowance granted where ~1500 actually fit). An agent
        runs one invocation at a time, so this agent showing up under a new
        invocation_id is proof the old invocation's pending reservations can
        never settle: cancel them before this admission is priced. Should a
        cancelled call somehow settle after all, the ledger still records
        the spend -- the settled flag only guards the committed release.
        """
        for inv_id, entries in list(self._pending.items()):
            if inv_id == current_invocation:
                continue
            stale = [e for e in entries if e[1] == key]
            if not stale:
                continue
            kept = [e for e in entries if e[1] != key]
            if kept:
                self._pending[inv_id] = kept
            else:
                del self._pending[inv_id]
            for reservation, _, _ in stale:
                await self.ledger.cancel(reservation)
                self._emit("reconciled", agent=key, amount=reservation.amount,
                           stale_invocation=inv_id)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = callback_context.agent_name
        await self._reconcile_stale(key, callback_context.invocation_id)
        raw_input = estimate_input_tokens(llm_request)
        input_estimate = self.calibrator.calibrate(key, raw_input)
        output_estimate = self.estimator.predict(key)
        estimate = input_estimate + output_estimate

        reservation = None
        allowance = 0
        thoughts_toll = self.thoughts.predict(key)
        # Runway check, on every admission: landing costs one more read of
        # the context, and the context grows every turn -- waiting for a
        # denial can strand the mission past the point where not even the
        # landing's input fits (observed live: 3.1k left against a 6.5k
        # context). Keep enough fuel to reach the runway: if admitting this
        # call would leave less than the NEXT landing costs (today's context
        # plus this call's output, margin, floor -- and the thinking toll,
        # for agents whose settle history shows one: reasoning models bill
        # thoughts against the landing's own output cap, so the runway must
        # carry room to think AND room to speak). A reasoning agent lands
        # earlier than a non-reasoning one, which is the correct trade:
        # better early with room to speak than on time with room only to
        # think.
        headroom = self.ledger.budget - self.ledger.spent - self.ledger.committed
        # The next call's context grows by this call's output AND, on
        # ADK/Gemini, by its thoughts: thought parts re-enter the history
        # and get re-read (observed live: a 1.311-thought call grew the
        # next input by ~2.5k; the projection that ignored it admitted on a
        # 32-token margin and the mission was decapitated two turns later
        # -- the window did not jump, the projection was blind to half the
        # growth). The toll therefore appears TWICE below, for two
        # different reasons: once as context growth of the next call, once
        # as the landing's own room to think.
        next_input = input_estimate + output_estimate + thoughts_toll
        runway = next_input + next_input // 10 + 128 + LANDING_FLOOR + thoughts_toll
        if headroom - estimate < runway:
            reservation, allowance, thoughts_toll = await self._try_landing(
                key, llm_request
            )

        if reservation is None:
            reservation = await self.ledger.try_reserve(estimate)
        appealed = False
        if reservation is None:
            # Right of appeal (voice): a justified retry may enter the appeal
            # tranche -- never the completion reserve. Rationed and logged.
            justification = self._extract_appeal(llm_request)
            if justification:
                reservation = await self.appeals.appeal(key, estimate, justification)
                appealed = reservation is not None
        if reservation is None:
            # Landing protocol: within one invocation a denial is terminal
            # (the short-circuit text becomes the agent's final message), so
            # before denying, admit one last call capped to whatever output
            # still fits. Re-entrant on purpose: a model that spends its
            # allowance imitating tool calls from its own history (observed
            # live -- stripping declarations does not strip the pattern)
            # gets a shorter runway each time, until the floor ends it.
            # Headroom shrinks monotonically, so this terminates.
            reservation, allowance, thoughts_toll = await self._try_landing(
                key, llm_request
            )
        if reservation is None:
            # Short-circuit: the model is never called. The refusal text tells
            # the agent to land with what it has, or appeal once with cause.
            self._emit("denied_terminal", agent=key, estimate=estimate)
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=DENIAL_TEXT)]
                )
            )

        self._pending[callback_context.invocation_id].append(
            (reservation, key, raw_input)
        )

        if allowance:
            self.landings += 1
            self._landing_reservation = reservation
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
            # The landing order rides as the LAST user message, not as a
            # system instruction: the end of the context is where attention
            # actually lands, and models that shrugged off the instruction
            # obeyed the message.
            llm_request.contents = list(llm_request.contents or []) + [
                types.Content(
                    role="user",
                    parts=[types.Part(
                        text=LANDING_TEXT.format(allowance=allowance)
                    )],
                )
            ]
            self._emit("landing", agent=key, allowance=allowance,
                       cap=llm_request.config.max_output_tokens,
                       reserved=reservation.amount,
                       thoughts_toll=thoughts_toll)
        else:
            # The reservation bounded EXPECTED spend; nothing yet bounds
            # REALIZED spend -- an output that runs past its p90 estimate
            # bills anyway and the ledger can only record the excess
            # (observed live: once the estimator warmed to ~200 tokens,
            # 11 of 26 runs overshot, median 337, with denied=0). Make the
            # wall physical: cap the output at the reserved estimate plus
            # the tail margin, clamped so even a maximal reply fits in the
            # tranche this call was admitted from -- an ordinary call must
            # leave the appeal tranche and the completion reserve intact,
            # an appealed call only the completion reserve. The tranche
            # slack left after this reservation is exactly what the actual
            # may exceed the estimate by, so cap = estimate + slack is the
            # wall itself; far from it, the margin governs.
            slack = (
                self.ledger.priority_available if appealed
                else self.ledger.available
            )
            cap = max(1, min(output_cap(output_estimate),
                             output_estimate + slack))
            if llm_request.config is None:
                llm_request.config = types.GenerateContentConfig()
            prior_cap = llm_request.config.max_output_tokens
            llm_request.config.max_output_tokens = (
                min(prior_cap, cap) if prior_cap else cap
            )
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
            self._emit("admitted", agent=key, estimate=estimate,
                       cap=llm_request.config.max_output_tokens)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        stack = self._pending.get(callback_context.invocation_id)
        if not stack:
            return None
        reservation, key, raw_input = stack.pop()

        usage = llm_response.usage_metadata
        thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
        was_landing = reservation is self._landing_reservation
        if usage is not None and usage.total_token_count:
            actual = usage.total_token_count
            output = usage.candidates_token_count or 0
            # Thinking bills as output, and on Gemini max_output_tokens
            # governs thoughts + text together. The estimator must learn
            # that billed sum: a p90 of the visible text alone leaves every
            # thinking call under-reserved by its thoughts (observed live
            # on Gemma: text ~21, thoughts ~150+, both billed -- the exact
            # size of the recorded overshoots), and a cap sized from text
            # alone would strangle the thinking before the answer. The
            # thoughts estimator learns the toll separately: the landing
            # needs the SPLIT (room to think vs room to speak), the wall
            # needs the SUM.
            #
            # But only CRUISE calls train the output estimator. A landing's
            # bill is ceiling-shaped -- capped by the allowance, inflated by
            # the deliberation the FINAL ALLOWANCE order provokes -- and
            # feeding it back teaches the estimator landing-sized costs,
            # which fattens the runway, which forces the next landing: a
            # self-reinforcing cliff (observed live: 22/22 turn-1 landings
            # at a budget the staircase had validated, prediction climbing
            # 700 -> ~1.5k on nothing but its own landings). The calibrator
            # must not learn the takeoff; the estimator must not learn the
            # landing. The thoughts estimator DOES learn it: the landing
            # toll is best measured where it bites.
            if not was_landing:
                self.estimator.update(key, output + (thoughts or 0))
            self.thoughts.update(key, thoughts)
            prompt = getattr(usage, "prompt_token_count", None)
            if prompt:
                self.calibrator.update(key, raw_input, prompt)
        else:
            actual = reservation.amount  # no metadata: charge the estimate
            output = None
            thoughts = None
        await self.ledger.settle(reservation, actual)
        self._emit(
            "settled", agent=key, actual=actual, output=output,
            thoughts=thoughts, was_landing=was_landing,
        )

        # A landing is a landing: nothing takes off after it. Stripping tool
        # *declarations* does not stop a model whose own history teaches the
        # pattern (observed live: the landed call kept tool-calling and its
        # follow-up died at the exhausted ledger). So the guarantee moves to
        # the response side: drop any function calls from the landed reply,
        # and the invocation finalizes with whatever the model actually
        # wrote -- its words, however thin, not the governor's.
        if was_landing and llm_response.content and llm_response.content.parts:
            kept = [p for p in llm_response.content.parts
                    if p.function_call is None]
            stripped = len(llm_response.content.parts) - len(kept)
            if stripped:
                if not any(p.text for p in kept):
                    kept.append(types.Part(text=(
                        "[BUDGET GOVERNOR] The mission landed at budget "
                        "exhaustion before a final report was written."
                    )))
                self._emit("landing_enforced", agent=key,
                           calls_stripped=stripped)
                return LlmResponse(
                    content=types.Content(role="model", parts=kept),
                    usage_metadata=llm_response.usage_metadata,
                )
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
            reservation, _, _ = stack.pop()
            await self.ledger.cancel(reservation)
            self._emit("cancelled", amount=reservation.amount,
                       was_landing=reservation is self._landing_reservation,
                       error=type(error).__name__)
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Invocation end: whatever it left pending, it can no longer settle.

        The belt to _reconcile_stale's suspenders -- that path only fires if
        the agent runs again, and the leak may happen on the mission's last
        call. Also drops the invocation's (usually empty) _pending slot, so
        the dict does not grow one key per invocation forever.
        """
        entries = self._pending.pop(invocation_context.invocation_id, [])
        for reservation, key, _ in entries:
            await self.ledger.cancel(reservation)
            self._emit("reconciled", agent=key, amount=reservation.amount,
                       stale_invocation=invocation_context.invocation_id)

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
