"""CLI Wrapper for the Budget Governor Ledger.
Allows the parent agent to manage a persistent budget across sub-agent calls.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path to use the actual project logic
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governor.ledger import AtomicLedger

STATE_FILE = ".governor_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None

def save_state(ledger: AtomicLedger, mission: str | None = None):
    # AtomicLedger isn't naturally JSON serializable, so we save its key metrics
    state = {
        "budget": ledger.budget,
        "spent": ledger.spent,
        "committed": ledger.committed,
        "overshoot": ledger.overshoot,
        "reserve_fraction": ledger.reserve_fraction,
        "appeal_fraction": ledger.appeal_fraction,
        "mission": mission,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def init_budget(budget: int, mission: str, reserve: float, appeal: float):
    ledger = AtomicLedger(budget=budget, reserve_fraction=reserve, appeal_fraction=appeal)
    save_state(ledger, mission=mission)
    print(f"Budget initialized: {budget} tokens. Mission: {mission}")

def get_status():
    state = load_state()
    if not state:
        print("No budget initialized. Use 'init' first.")
        return
    
    # Reconstruct a temporary ledger to get calculated properties
    ledger = AtomicLedger(
        budget=state["budget"], 
        reserve_fraction=state["reserve_fraction"], 
        appeal_fraction=state["appeal_fraction"]
    )
    ledger.spent = state["spent"]
    ledger.committed = state["committed"]
    
    print(f"Mission: {state.get('mission', 'Not set')}")
    print(f"Budget: {ledger.budget} | Spent: {ledger.spent} | Available: {ledger.available} | Committed: {ledger.committed}")

def reserve_tokens(amount: int):
    state = load_state()
    if not state: return
    
    ledger = AtomicLedger(budget=state["budget"], reserve_fraction=state["reserve_fraction"], appeal_fraction=state["appeal_fraction"])
    ledger.spent = state["spent"]
    ledger.committed = state["committed"]
    
    res = ledger.try_reserve(amount)
    if res:
        save_state(ledger)
        print(f"Reserved {amount} tokens. New Available: {ledger.available}")
    else:
        print("DENIED: Insufficient budget.")
        sys.exit(1)

def settle_tokens(actual: int):
    state = load_state()
    if not state: return
    
    ledger = AtomicLedger(budget=state["budget"], reserve_fraction=state["reserve_fraction"], appeal_fraction=state["appeal_fraction"])
    ledger.spent = state["spent"]
    ledger.committed = state["committed"]
    
    # We simplify settlement for the CLI: just add to spent and clear commitment
    # In a real scenario we'd track reservation IDs.
    ledger.spent += actual
    # This is a naive settlement for the CLI tool
    # we assume the commitment was for the previous call
    # We'll just let the parent agent handle the balance.
    save_state(ledger)
    print(f"Settled {actual} tokens. Total Spent: {ledger.spent}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    init_p = subparsers.add_parser("init")
    init_p.add_argument("budget", type=int)
    init_p.add_argument("--reserve", type=float, default=0.1)
    init_p.add_argument("--appeal", type=float, default=0.05)

    subparsers.add_parser("status")
    
    res_p = subparsers.add_parser("reserve")
    res_p.add_argument("amount", type=int)
    
    set_p = subparsers.add_parser("settle")
    set_p.add_argument("actual", type=int)

    args = parser.parse_args()

    if args.command == "init":
        init_budget(args.budget, args.reserve, args.appeal)
    elif args.command == "status":
        get_status()
    elif args.command == "reserve":
        reserve_tokens(args.amount)
    elif args.command == "settle":
        settle_tokens(args.actual)
