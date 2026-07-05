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

MODEL = "gemini-2.5-flash"

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
            "topic you are given. No prose polish; bullet points."
        ),
    )
    writer = LlmAgent(
        name="writer",
        model=MODEL,
        description="Turns the researcher's notes into polished prose.",
        instruction=(
            "You are the writer. Turn the notes you receive into the "
            "requested deliverable, exactly as specified."
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


async def run_once(budget: int, visibility: bool) -> BudgetGovernorPlugin:
    governor = BudgetGovernorPlugin(
        budget=budget,
        reserve_fraction=0.10,
        visibility=visibility,
        mission=TASK,
        arbiter=True,  # appeals heard by a budgeted judge agent, not the coordinator
    )
    app = App(name="overshoot_demo", root_agent=build_team(), plugins=[governor])
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="overshoot_demo", user_id="capstone"
    )
    message = types.Content(role="user", parts=[types.Part(text=TASK)])
    async for event in runner.run_async(
        user_id="capstone", session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            preview = event.content.parts[0].text.strip().replace("\n", " ")[:110]
            print(f"  [{event.author}] {preview}")
    return governor


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=20_000)
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit(
            "GOOGLE_API_KEY is not set. Get a key at https://aistudio.google.com/ "
            "and run:  set GOOGLE_API_KEY=...  (or export on Linux/Kaggle)."
        )

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
