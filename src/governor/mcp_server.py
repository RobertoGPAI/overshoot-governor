"""MCP server exposing the governor's ledger as tools.

Why MCP here: the ADK plugin governs one Runner, but real fleets mix runtimes
(ADK teams, Claude Code sessions, custom scripts). Running ONE governor as an
MCP server gives them all the same atomic ledger -- cross-runtime budget
governance over the standard protocol. An ADK agent can also mount these tools
via MCPToolset, turning the meter into something the agent can *ask* for
instead of something injected into its context.

Run (stdio transport, budget from env):

    set GOVERNOR_BUDGET=100000
    python -m governor.mcp_server

Wire into ADK:

    from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
    tools = MCPToolset(connection_params=StdioConnectionParams(
        server_params={"command": "python", "args": ["-m", "governor.mcp_server"]}))
"""

from __future__ import annotations

import os
import uuid

from mcp.server.fastmcp import FastMCP

from .estimator import OutputEstimator
from .ledger import AtomicLedger, Reservation

mcp = FastMCP("overshoot-governor")

_ledger = AtomicLedger(
    budget=int(os.environ.get("GOVERNOR_BUDGET", "100000")),
    reserve_fraction=float(os.environ.get("GOVERNOR_RESERVE", "0.10")),
)
_estimator = OutputEstimator()
_reservations: dict[str, tuple[Reservation, str]] = {}


@mcp.tool()
async def budget_status() -> dict:
    """Read the live budget meter: available, committed, spent, overshoot."""
    return {
        "budget": _ledger.budget,
        "available": _ledger.available,
        "committed_in_flight": _ledger.committed,
        "spent": _ledger.spent,
        "overshoot": _ledger.overshoot,
        "finalizing": _ledger.finalizing,
    }


@mcp.tool()
async def reserve(agent: str, input_tokens: int) -> dict:
    """Ask admission for one model call. Returns a reservation_id if admitted.

    The reserved amount is input_tokens plus the p90 of this agent's observed
    output tokens (a conservative prior until enough history accumulates).
    Call settle() with the actual usage afterwards -- unsettled reservations
    hold budget forever.
    """
    estimate = input_tokens + _estimator.predict(agent)
    reservation = await _ledger.try_reserve(estimate)
    if reservation is None:
        return {
            "admitted": False,
            "reason": "projected cost exceeds remaining budget",
            "available": _ledger.available,
        }
    rid = uuid.uuid4().hex[:12]
    _reservations[rid] = (reservation, agent)
    return {"admitted": True, "reservation_id": rid, "reserved": estimate}


@mcp.tool()
async def settle(reservation_id: str, actual_total_tokens: int, output_tokens: int = 0) -> dict:
    """Reconcile a reservation with the actual token usage of the call."""
    entry = _reservations.pop(reservation_id, None)
    if entry is None:
        return {"ok": False, "reason": "unknown or already-settled reservation_id"}
    reservation, agent = entry
    await _ledger.settle(reservation, actual_total_tokens)
    if output_tokens:
        _estimator.update(agent, output_tokens)
    return {"ok": True, "spent": _ledger.spent, "available": _ledger.available}


@mcp.tool()
async def cancel(reservation_id: str) -> dict:
    """Release a reservation whose call failed before consuming tokens."""
    entry = _reservations.pop(reservation_id, None)
    if entry is None:
        return {"ok": False, "reason": "unknown or already-settled reservation_id"}
    await _ledger.cancel(entry[0])
    return {"ok": True, "available": _ledger.available}


@mcp.tool()
async def begin_finalization() -> dict:
    """Unlock the completion reserve so the mission can land."""
    _ledger.begin_finalization()
    return {"finalizing": True, "available": _ledger.available}


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
