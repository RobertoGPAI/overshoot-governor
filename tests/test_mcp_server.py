"""Smoke test: the MCP server's tools drive the shared ledger correctly."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governor import mcp_server


def test_reserve_settle_roundtrip():
    async def scenario():
        status = await mcp_server.budget_status()
        assert status["spent"] == 0

        admitted = await mcp_server.reserve(agent="tester", input_tokens=500)
        assert admitted["admitted"] is True

        done = await mcp_server.settle(
            admitted["reservation_id"], actual_total_tokens=900, output_tokens=400
        )
        assert done["ok"] is True
        assert done["spent"] == 900

        # settling twice is rejected
        again = await mcp_server.settle(admitted["reservation_id"], 900)
        assert again["ok"] is False

    asyncio.run(scenario())


def test_denial_when_budget_exhausted():
    async def scenario():
        huge = mcp_server._ledger.usable_budget * 2
        denied = await mcp_server.reserve(agent="tester", input_tokens=huge)
        assert denied["admitted"] is False

    asyncio.run(scenario())
