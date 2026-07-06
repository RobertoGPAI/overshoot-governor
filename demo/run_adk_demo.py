"""Live ADK 2.x demo: a governed multi-agent team doing real Gemini calls.

Runs the same research task twice under the same budget -- once with budget
visibility injected into every request (the meter in the hallway) and once
blind -- and prints both ledger reports. Requires GOOGLE_API_KEY.

    python demo/run_adk_demo.py [--budget 20000]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from governor.adk_plugin import BudgetGovernorPlugin

# flash-lite has far higher free-tier rate limits; use --model to switch
# when the default's requests-per-minute/day quota runs out.
DEFAULT_MODEL = "gemini-2.5-flash"
MODEL = DEFAULT_MODEL

TASK = (
    "Research the concept of 'overshoot and collapse' in system dynamics and "
    "write a 3-paragraph summary connecting it to resource limits in "
    "multi-agent AI systems."
)


def build_team() -> LlmAgent:
    researcher = LlmAgent(
        name="researcher",
        model=MODEL,
        description="Gathers and condenses background facts for the team.",
        instruction=(
            "You are the researcher. Produce concise factual notes on the "
            "topic you are given. No prose polish; bullet points. "
            "Never transfer to yourself; when done, hand off to the writer "
            "or back to the coordinator."
        ),
    )
    writer = LlmAgent(
        name="writer",
        model=MODEL,
        description="Turns the researcher's notes into polished prose.",
        instruction=(
            "You are the writer. Turn the notes you receive into the "
            "requested deliverable, exactly as specified. "
            "Never transfer to yourself; deliver your text directly."
        ),
    )
    return LlmAgent(
        name="coordinator",
        model=MODEL,
        description="Coordinates the team and delivers the final answer.",
        instruction=(
            "You coordinate a research team. Delegate research to the "
            "researcher and drafting to the writer, then deliver the result."
        ),
        sub_agents=[researcher, writer],
    )


async def run_once(
    budget: int,
    visibility: bool,
    *,
    reserve_fraction: float = 0.10,
    appeal_fraction: float = 0.05,
    appeal_round: bool = False,
) -> BudgetGovernorPlugin:
    governor = BudgetGovernorPlugin(
        budget=budget,
        reserve_fraction=reserve_fraction,
        appeal_fraction=appeal_fraction,
        visibility=visibility,
        mission=TASK,
        arbiter=True,  # appeals heard by a budgeted judge agent, not the coordinator
    )
    app = App(name="overshoot_demo", root_agent=build_team(), plugins=[governor])
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="overshoot_demo", user_id="capstone"
    )

    async def send(text: str) -> None:
        # Provider errors (429/503) or runtime hiccups must not vaporize the
        # run: the governor cancels the affected reservation via
        # on_model_error_callback, and the ledger report still prints.
        message = types.Content(role="user", parts=[types.Part(text=text)])
        try:
            async for event in runner.run_async(
                user_id="capstone", session_id=session.id, new_message=message
            ):
                if event.content and event.content.parts and event.content.parts[0].text:
                    preview = event.content.parts[0].text.strip().replace("\n", " ")[:110]
                    print(f"  [{event.author}] {preview}")
        except Exception as exc:  # noqa: BLE001 -- demo resilience, not policy
            print(f"  [demo] turn aborted by provider/runtime error: "
                  f"{type(exc).__name__}: {str(exc)[:140]}")

    await send(TASK)

    # A denial silences its addressee: the refusal short-circuits the model
    # call, so the denied agent never hears the verdict -- to appeal it would
    # need the very call it was denied. The driver therefore acts as public
    # prosecutor (fiscalia), filing the appeal ex officio on the mission's
    # behalf, not the agent's; the AppealsDesk + MissionJudge then decide.
    if appeal_round and governor.ledger.stats.denied:
        print("  -- denial detected; filing an appeal and retrying --")
        # The plea must be forward-looking (marginal value vs marginal cost):
        # the ruling prompt instructs the judge to reject sunk-cost reasoning,
        # and "we already spent so much" pleas get refused -- correctly.
        await send(
            "APPEAL: granting this single call produces the mission's entire "
            "deliverable (the 3-paragraph summary); refusing it means the "
            "mission returns nothing at all."
        )
    return governor


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=20_000)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model for the team (try gemini-2.5-flash-lite when the "
        "free-tier quota of the default runs out)",
    )
    parser.add_argument(
        "--appeal-demo",
        action="store_true",
        help="single run tuned so the budget bites mid-mission, then an appeal "
        "is filed and heard by the MissionJudge (uses --budget, default 12000)",
    )
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    # AI Studio and most tutorials call it GEMINI_API_KEY; the google-genai SDK
    # accepts either name. Accept both so the demo runs whichever you set.
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit(
            "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY). "
            "Get a free key at https://aistudio.google.com/apikey and run:  "
            "export GEMINI_API_KEY=...  (Linux/WSL/Kaggle)."
        )

    if args.appeal_demo:
        # Ceilings tuned so the ordinary tranche affords roughly ONE call (a
        # fresh reservation is ~input + the 1024-token p90 prior): the first
        # call is admitted, the next is denied mid-mission, and the appeal
        # tranche can still afford the hearing (400) plus the appealed call.
        # Ordinary 30%, appeals 25%, completion reserve 45%.
        budget = args.budget if args.budget != 20_000 else 5_000
        print(f"=== Appeal demo: budget {budget}, ordinary ceiling {int(budget * 0.30)} ===")
        gov = await run_once(
            budget,
            visibility=True,
            reserve_fraction=0.45,
            appeal_fraction=0.25,
            appeal_round=True,
        )
        print("ledger:", gov.report())
        for rec in gov.appeals.log.records:
            verdict = "GRANTED" if rec.granted else "REFUSED"
            print(f"appeal [{verdict}] {rec.agent}: {rec.justification!r}")
        if gov.judge:
            print(
                f"judge: {gov.judge.hearings} hearing(s), "
                f"{gov.judge.hearing_tokens} tokens spent on justice"
            )
            if gov.judge.hearing_failures:
                print(
                    f"judge: {gov.judge.hearing_failures} FAILED hearing(s) -- "
                    f"last error: {type(gov.judge.last_error).__name__}: "
                    f"{str(gov.judge.last_error)[:140]}"
                )
        return

    print(f"=== Condition A: blind (no meter), budget {args.budget} tokens ===")
    blind = await run_once(args.budget, visibility=False)
    print("ledger:", blind.report())

    print(f"\n=== Condition B: sighted (meter visible), budget {args.budget} tokens ===")
    sighted = await run_once(args.budget, visibility=True)
    print("ledger:", sighted.report())

    print("\nSame task, same enforcement; the only difference is whether the")
    print("agents can see the meter. Compare `spent` between the two reports.")


if __name__ == "__main__":
    asyncio.run(main())
